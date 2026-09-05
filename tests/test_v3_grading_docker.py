from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rlvr.v3.grading import evaluate_repository, evaluate_terminal
from rlvr.v3.identity import RepositoryTaskIdentity, TerminalScriptTaskIdentity
from rlvr.v3.manifest import load_manifest
from rlvr.v3.patch import PatchLimits
from rlvr.v3.script import ScriptLimits
from rlvr.v3.supervisor import SupervisorPolicy
from rlvr.v3.tree import TreeLimits

IMAGE = (
    "public.ecr.aws/t3h1r6x1/hone-subnet/polyglot-sandbox@sha256:"
    "87f7ea823a2ffde124040db0f271e59110e666afecfc68106c4b07fddb4eee08"
)
HEX = "ab" * 32


def docker_ready() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    probe = subprocess.run(
        [docker, "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not docker_ready(), reason="pinned Docker image unavailable")


def policy() -> SupervisorPolicy:
    uid = os.getuid()
    gid = os.getgid()
    return SupervisorPolicy(
        image=IMAGE,
        candidate_uid=uid,
        candidate_gid=gid,
        trusted_uid=uid,
        trusted_gid=gid,
        memory_bytes=256 << 20,
        cpus=1,
        pids_limit=64,
        tmpfs_bytes=64 << 20,
        max_file_bytes=16 << 20,
        watchdog_slack_s=5,
    )


def tree_limits() -> TreeLimits:
    return TreeLimits(128, 1 << 20, 4 << 20, 512)


def write_manifest(root: Path, document: dict, files: dict[str, bytes]) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(document))
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    return root


def expectation(stdout: str) -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": None}


def inspection(check_id: str, checker: str, stdout: str) -> dict:
    return {
        "check_id": check_id,
        "kind": "inspection",
        "argv": ("/opt/rlvr-venv/bin/python3", "-I", "-S", f"/verify/{checker}"),
        "timeout_s": 10,
        "max_stdout_bytes": 65_536,
        "max_stderr_bytes": 65_536,
        "expect": expectation(stdout),
    }


def test_repository_patch_runs_hidden_invocation_and_inspection(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "sub" / "greet.py"
    source.parent.mkdir(parents=True)
    source.write_text("name = input().strip()\nprint(f'Hello, {name}')\n")
    verifier = write_manifest(
        tmp_path / "verifier",
        {
            "manifest_version": 1,
            "task_type": "repository_patch_v1",
            "setup": None,
            "checks": [
                {
                    "check_id": "greet",
                    "kind": "invocation",
                    "argv": ("/opt/rlvr-venv/bin/python3", "-I", "-S", "/work/sub/greet.py"),
                    "cwd": ".",
                    "stdin": "inputs/name.txt",
                    "timeout_s": 10,
                    "max_stdout_bytes": 65_536,
                    "max_stderr_bytes": 65_536,
                    "expect": expectation("gold/greeting.txt"),
                },
                inspection("layout", "layout.py", "gold/layout.txt"),
            ],
        },
        {
            "inputs/name.txt": b"world\n",
            "gold/greeting.txt": b"Hello, world!\n",
            "gold/layout.txt": b"sub\nsub/greet.py\n",
            "checks/layout.py": (
                b"from pathlib import Path\n"
                b"r=Path('/result')\n"
                b"print('\\n'.join(sorted(str(p.relative_to(r)) for p in r.rglob('*'))))\n"
            ),
        },
    )
    manifest = load_manifest(verifier, task_type="repository_patch_v1")
    identity = RepositoryTaskIdentity(
        task_kind="bug_fix",
        instruction="Add punctuation",
        primary_language="python",
        workspace_sha256=HEX,
        verifier_sha256=HEX,
        execution_profile_id="proto-python-v1",
        working_directory="sub",
        verifier_policy="black-box-v1",
        authoring_version="fixture-1",
    )
    patch = (
        b"diff --git a/sub/greet.py b/sub/greet.py\n"
        b"--- a/sub/greet.py\n+++ b/sub/greet.py\n"
        b"@@ -1,2 +1,2 @@\n name = input().strip()\n"
        b"-print(f'Hello, {name}')\n+print(f'Hello, {name}!')\n"
    )
    result = evaluate_repository(
        workspace,
        patch,
        identity,
        manifest,
        verifier,
        policy(),
        tree_limits(),
        PatchLimits(1 << 20, 10),
        shutil.which("docker") or "/usr/bin/docker",
        f"hone-repo-{os.getpid()}",
    )
    assert result.status == "passed", result
    assert [check.outcome for check in result.checks] == ["passed", "passed"]


def test_terminal_script_is_graded_only_from_result_state(tmp_path):
    environment = tmp_path / "environment"
    (environment / "data").mkdir(parents=True)
    (environment / "out").mkdir()
    (environment / "data" / "input.txt").write_text("pear\napple\npear\n")
    verifier = write_manifest(
        tmp_path / "verifier",
        {
            "manifest_version": 1,
            "task_type": "terminal_script_v1",
            "setup": None,
            "checks": [inspection("report", "report.py", "gold/report.txt")],
        },
        {
            "gold/report.txt": b"apple\npear\n",
            "checks/report.py": b"from pathlib import Path\nprint(Path('/result/report.txt').read_text(), end='')\n",
        },
    )
    manifest = load_manifest(verifier, task_type="terminal_script_v1")
    identity = TerminalScriptTaskIdentity(
        instruction="Create a sorted unique report",
        environment_sha256=HEX,
        verifier_sha256=HEX,
        execution_profile_id="proto-python-v1",
        result_tree_path="out",
        verifier_policy="black-box-v1",
        authoring_version="fixture-1",
    )
    result = evaluate_terminal(
        environment,
        b"sort -u ../data/input.txt > report.txt\nexit 3\n",
        identity,
        manifest,
        verifier,
        policy(),
        tree_limits(),
        ScriptLimits(1 << 20),
        shutil.which("docker") or "/usr/bin/docker",
        f"hone-term-{os.getpid()}",
        script_timeout_s=30,
        script_max_output_bytes=65_536,
    )
    assert result.status == "passed"
    assert result.script_exit_code == 3
    assert (environment / "out" / "report.txt").read_bytes() == b"apple\npear\n"
