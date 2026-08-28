"""Public entrypoint, shell, and documentation checks."""

from __future__ import annotations

import http.server
import importlib.util
import os
import shutil
import subprocess
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
RUN_VALIDATOR = SCRIPTS / "run_validator.py"


def _load_run_validator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_run_validator_under_test", RUN_VALIDATOR
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_validator_fails_closed_without_problem_source(monkeypatch):
    module = _load_run_validator()
    settings = types.SimpleNamespace(problem_server_url="")
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    with pytest.raises(SystemExit, match="PROBLEM_SERVER_URL"):
        module.main()


SHELL_SCRIPTS = [
    REPO_ROOT / "setup_validator.sh",
    REPO_ROOT / "start_validator.sh",
    REPO_ROOT / "start_demo_miner.sh",
    SCRIPTS / "preflight_validator.sh",
    SCRIPTS / "register_testnet.sh",
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_are_valid_bash(script: Path):
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_start_script_launches_only_the_validator():
    text = (REPO_ROOT / "start_validator.sh").read_text()
    assert ".venv/bin/activate" in text
    assert "preflight_validator.sh" in text
    assert "scripts/run_validator.py" in text
    assert "run_miner" not in text
    assert "problem_server" not in text


def test_start_script_fails_closed_without_setup():
    text = (REPO_ROOT / "start_validator.sh").read_text()
    assert "run ./setup_validator.sh first" in text
    assert "using current Python" not in text


def test_setup_pulls_the_configured_immutable_sandbox_image():
    setup = (REPO_ROOT / "setup_validator.sh").read_text()
    preflight = (SCRIPTS / "preflight_validator.sh").read_text()
    assert 'preflight_validator.sh" --pull' in setup
    assert "docker pull" in preflight
    assert "RELEASE_POLICY.v3_image" in preflight
    assert "run_timed 20 docker info" in preflight
    assert "run_timed 45 docker run" in preflight


def test_preflight_probes_the_protected_lease_route_without_consuming_a_problem():
    preflight = (SCRIPTS / "preflight_validator.sh").read_text()
    assert 'f"{base}/v3/challenges/lease"' in preflight
    assert 'data=b"{}"' in preflight
    assert 'method="POST"' in preflight
    assert "status != 401" in preflight
    assert 'headers.get("Date", "")' in preflight
    assert "/v1/health" not in preflight


@pytest.mark.parametrize(
    "status, valid_date, succeeds",
    [(401, True, True), (200, True, False), (302, True, False), (401, False, False)],
)
def test_preflight_requires_an_unsigned_lease_rejection(
    status, valid_date, succeeds, tmp_path
):
    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received["path"] = self.path
            received["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.end_headers()

        def log_message(self, *_args):
            pass

        def date_time_string(self, timestamp=None):
            if valid_date:
                return super().date_time_string(timestamp)
            return "not-a-date"

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        docker = tmp_path / "docker"
        docker.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = run ]; then printf rlvr-preflight-ok; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        docker.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PROBLEM_SERVER_URL": f"http://127.0.0.1:{server.server_port}",
            "SUBTENSOR_NETWORK": "test",
        }
        result = subprocess.run(
            ["bash", str(SCRIPTS / "preflight_validator.sh")],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (result.returncode == 0) is succeeds
    assert received == {"path": "/v3/challenges/lease", "body": b"{}"}


def test_register_script_is_runbook_only():
    text = (SCRIPTS / "register_testnet.sh").read_text()
    assert "exit 0" in text
    for keyword in ("new_coldkey", "new_hotkey", "faucet", "subnet register"):
        assert keyword in text


def test_docs_describe_the_validator_and_demo_miner_boundary():
    readme = (REPO_ROOT / "README.md").read_text().lower()
    for keyword in (
        "problem server",
        "validator",
        "demo miner",
        "sandbox",
        "weights",
        "v3",
    ):
        assert keyword in readme


# --------------------------------------------------------------------------- #
# Stage-2 Rust activation package (docs/RUST_CHALLENGES.md): reproducible
# sandbox image, honest digest handling, staged preflight, benchmark matrix.
# --------------------------------------------------------------------------- #
RUST_DOCKERFILE = REPO_ROOT / "docker" / "rust-sandbox" / "Dockerfile"
RUST_BUILD_SCRIPT = SCRIPTS / "build_rust_sandbox.sh"


def test_rust_sandbox_dockerfile_is_digest_pinned_and_minimal():
    """The canonical image recipe: digest-pinned official Rust base, the
    Python 3 supervisor runtime, nothing else. The toolchain version is
    asserted AT BUILD TIME against the release policy's pin, so an image
    built from a drifted base fails the build instead of skewing verdicts."""
    from rlvr.policy import RELEASE_POLICY

    assert RUST_DOCKERFILE.exists()
    text = RUST_DOCKERFILE.read_text(encoding="utf-8")

    from_lines = [l for l in text.splitlines() if l.startswith("FROM ")]
    assert from_lines, "Dockerfile has no FROM line"
    assert all("@sha256:" in line for line in from_lines), (
        "base image must be digest-pinned, not tag-floating"
    )
    assert "python3" in text
    assert "--no-install-recommends" in text
    assert "rm -rf /var/lib/apt/lists" in text
    # Build-time toolchain assertion, version-locked to policy.
    assert RELEASE_POLICY.rustc_version in text
    assert "rustc --version" in text


def test_build_script_never_invents_a_digest():
    """The ONLY digest that may enter RELEASE_POLICY is a registry
    RepoDigest read back after a real push. The build script therefore
    contains no digest literal of its own, reads the pinned toolchain
    version from policy, self-checks the built image, and prints the
    published digest with the policy line to update."""
    assert RUST_BUILD_SCRIPT.exists()
    result = subprocess.run(
        ["bash", "-n", str(RUST_BUILD_SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    text = RUST_BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "@sha256:" not in text, "digests are read from the registry, never typed"
    assert "RepoDigests" in text
    assert "RELEASE_POLICY" in text
    assert "rustc --version" in text
    assert "--push" in text
    assert "policy" in text


def test_preflight_validates_the_v3_image():
    preflight = (SCRIPTS / "preflight_validator.sh").read_text(encoding="utf-8")

    assert "RELEASE_POLICY.v3_image" in preflight
    assert "git --version" in preflight
    assert "--require-rust" not in preflight


def test_benchmark_covers_the_rust_matrix():
    """The deployment benchmark grows a language axis plus the knobs the
    canonical matrix needs: source-size padding for the compile axis and
    the adversarial cases, all synthetic."""
    result = subprocess.run(
        [os.fspath(REPO_ROOT / ".venv" / "bin" / "python"),
         str(SCRIPTS / "benchmark_grading.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--language" in result.stdout
    assert "rust" in result.stdout
    assert "--source-pad-bytes" in result.stdout
    assert "--adversarial" in result.stdout
    # BENCHMARK-ONLY image override: the validator's executor is policy-pinned
    # (and must stay so), which means a locally built, not-yet-published image
    # is unbenchmarkable without a dev-tool override. The flag lets benchmark
    # data exist BEFORE the registry push decision, is echoed in the output
    # configuration block, and touches nothing in validator runtime.
    assert "--image" in result.stdout


def test_env_example_carries_no_rust_knobs():
    """Everything rust is release policy; nothing rust is operator env.
    Green today and must stay green — this is the B3 boundary expressed at
    the operator-facing file."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "RUST" not in text.upper()


def test_rust_sandbox_image_provides_the_full_supervisor_stdlib():
    """MEASURED on the benchmark host: python3-minimal lacks shutil under
    `python3 -I -S`, so the supervisor's tempfile import dies before the
    first case runs. The image must install the FULL python3 stdlib, and —
    the durable half of the pin — the build must PROVE the supervisor's
    import surface under the exact isolated flags the container uses, so
    any future Debian stdlib split fails the build, not the fleet."""
    text = RUST_DOCKERFILE.read_text(encoding="utf-8")

    assert "python3-minimal" not in text
    assert "python3 -I -S" in text
    assert "tempfile" in text, (
        "build must import-check the supervisor's stdlib surface"
    )
