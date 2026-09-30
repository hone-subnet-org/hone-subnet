"""One real terminal task, graded exactly as a validator would.

The fixture under tests/fixtures/v3-terminal-urlooker-651460ec is a task the
problem server issued once and retired: a pinned urlooker checkout plus
incident material, six hidden checks, and a reference bash script. It is not
in the mainnet task pool. The fast tests need no Docker; the grading tests run
in the pinned sandbox image and are skipped without it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from rlvr.policy import RELEASE_POLICY
from rlvr.v3.archive import extract_archive
from rlvr.v3.artifacts import ArtifactRef
from rlvr.v3.grading import evaluate_terminal
from rlvr.v3.identity import TerminalScriptTaskIdentity, compute_task_id
from rlvr.v3.manifest import load_manifest
from rlvr.v3.reasons import MinerReason, Stage
from rlvr.v3.release import round_policy
from rlvr.v3.script import validate_script
from tests.test_v3_fixture_yamlcpp import needs_image

FIXTURE = Path(__file__).parent / "fixtures" / "v3-terminal-urlooker-651460ec"
CHECK_IDS = [
    "00-summary-and-receipt-account", "01-complete-dispatch-ledger", "02-receiver-streams-and-exhaust",
    "03-complete-alarm-emissions-and", "04-all-notification-previews", "05-preserved-recovery-material",
]


def refs() -> dict[str, ArtifactRef]:
    raw = json.loads((FIXTURE / "artifact_refs.json").read_text())
    return {name: ArtifactRef(**value) for name, value in raw.items()}


def identity() -> TerminalScriptTaskIdentity:
    return TerminalScriptTaskIdentity(**json.loads((FIXTURE / "identity.json").read_text()))


def policy():
    return round_policy(RELEASE_POLICY, dispatch_concurrency=1)


def unpack(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    scratch = tmp_path / f"scratch-{name}"
    scratch.mkdir(parents=True, exist_ok=True)
    limits = policy().workspace_archive if name == "workspace" else policy().verifier_archive
    extract_archive(FIXTURE / f"{name}.tar.zst", refs()[name], target, limits, scratch_dir=scratch)
    return target


def try_task():
    spec = importlib.util.spec_from_file_location("try_task", Path(__file__).parent.parent / "scripts" / "try_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# fast: the fixture is internally consistent and within release limits
# --------------------------------------------------------------------------- #
def test_identity_hashes_to_the_task_id():
    assert compute_task_id(identity()) == (FIXTURE / "task_id.txt").read_text().strip()


def test_archives_match_their_references():
    for name, ref in refs().items():
        blob = (FIXTURE / f"{name}.tar.zst").read_bytes()
        assert hashlib.sha256(blob).hexdigest() == ref.sha256
        assert len(blob) == ref.compressed_size_bytes
    assert identity().environment_sha256 == refs()["workspace"].sha256
    assert identity().verifier_sha256 == refs()["verifier"].sha256


def test_archives_extract_under_release_limits_and_the_manifest_has_six_inspection_checks(tmp_path):
    workspace = unpack(tmp_path, "workspace")
    verifier = unpack(tmp_path, "verifier")
    assert (workspace / "recovery" / "README.md").is_file()
    assert (workspace / "go.mod").is_file()
    manifest = load_manifest(verifier, task_type="terminal_script_v1")
    assert [check.check_id for check in manifest.checks] == CHECK_IDS
    assert {check.kind for check in manifest.checks} == {"inspection"}
    assert manifest.setup is None
    assert identity().result_tree_path == "."


def test_reference_script_passes_the_script_rules():
    assert validate_script((FIXTURE / "reference.sh").read_bytes(), policy().script).status == "accepted"


# --------------------------------------------------------------------------- #
# grading in the pinned sandbox image
# --------------------------------------------------------------------------- #
def grade(tmp_path: Path, script: bytes, tag: str):
    workspace = unpack(tmp_path / tag, "workspace")
    verifier = unpack(tmp_path / tag, "verifier")
    manifest = load_manifest(verifier, task_type="terminal_script_v1")
    rp = policy()
    return evaluate_terminal(
        workspace, script, identity(), manifest, verifier,
        rp.supervisor, rp.tree, rp.script, rp.docker_binary, f"fixture-{tag}",
    )


@needs_image
def test_empty_script_fails_the_first_check(tmp_path):
    result = grade(tmp_path, b"", "empty")
    assert result.status == "failed"
    assert result.reason_code is MinerReason.CHECK_FAILED and result.stage is Stage.CHECK
    assert [(c.check_id, c.outcome) for c in result.checks][:2] == [(CHECK_IDS[0], "failed"), (CHECK_IDS[1], "skipped")]
    assert result.failed_check is None  # inspection checks have no display


@needs_image
def test_reference_script_passes_all_six_checks(tmp_path):
    result = grade(tmp_path, (FIXTURE / "reference.sh").read_bytes(), "reference")
    assert result.status == "passed", result.reason
    assert [c.outcome for c in result.checks] == ["passed"] * 6


@needs_image
def test_touching_the_recovery_inputs_fails_at_the_last_check(tmp_path):
    script = (FIXTURE / "reference.sh").read_bytes() + b"\necho tampered >> recovery/README.md\n"
    result = grade(tmp_path, script, "tampered")
    assert result.status == "failed"
    outcomes = [(c.check_id, c.outcome) for c in result.checks]
    assert outcomes[:5] == [(check_id, "passed") for check_id in CHECK_IDS[:5]]
    assert outcomes[5] == (CHECK_IDS[5], "failed")


# --------------------------------------------------------------------------- #
# the miner-facing script
# --------------------------------------------------------------------------- #
def test_try_task_extracts_the_terminal_workspace_and_hints_a_script(tmp_path, capsys):
    module = try_task()
    assert module.main(["--fixture", str(FIXTURE), "extract", str(tmp_path / "work")]) == 0
    out = capsys.readouterr().out
    assert (tmp_path / "work" / "recovery" / "README.md").is_file()
    assert "terminal script" in out and "alarm ledger" in out
    hint = next(line.strip() for line in out.splitlines() if "grade <your.sh>" in line)
    assert hint.split()[2:] == ["--fixture", str(FIXTURE), "grade", "<your.sh>"]


def test_try_task_rejects_an_oversized_script_after_reading_at_most_the_limit(tmp_path, capsys):
    module = try_task()
    big = tmp_path / "big.sh"
    with big.open("wb") as handle:
        handle.truncate(RELEASE_POLICY.v3_script_bytes * 8)  # sparse: 8x the limit on disk
    assert module.main(["--fixture", str(FIXTURE), "grade", str(big)]) == 1
    assert "script exceeds" in capsys.readouterr().out


@needs_image
def test_try_task_grades_a_script_like_a_validator(tmp_path, capsys):
    module = try_task()
    assert module.main(["--fixture", str(FIXTURE), "grade", str(FIXTURE / "reference.sh")]) == 0
    assert "status: passed" in capsys.readouterr().out
    empty = tmp_path / "empty.sh"
    empty.write_bytes(b"")
    assert module.main(["--fixture", str(FIXTURE), "grade", str(empty)]) == 1
    out = capsys.readouterr().out
    assert "status: failed (check_failed)" in out and f"{CHECK_IDS[0]}: failed" in out
