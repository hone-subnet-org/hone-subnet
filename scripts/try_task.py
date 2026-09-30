#!/usr/bin/env python
"""Work a real task locally and grade your answer exactly as a validator would.

The fixtures under tests/fixtures/ are tasks the problem server issued once
and retired: a repository task (default) and a terminal task. Use them to
check your miner end to end:

    # 1. get the workspace and the instruction
    python scripts/try_task.py extract /path/to/work

    # 2. produce your answer, however you like: a unified diff for a
    #    repository task, a bash script for a terminal task

    # 3. grade it in the pinned sandbox image (needs Docker)
    python scripts/try_task.py grade /path/to/fix.diff

    # the terminal task
    python scripts/try_task.py --fixture tests/fixtures/v3-terminal-urlooker-651460ec extract /path/to/work
    python scripts/try_task.py --fixture tests/fixtures/v3-terminal-urlooker-651460ec grade /path/to/solve.sh

Grading prints the status, the reason code, every check's outcome, and for a
failed check the same display a failure notice carries. The pinned image is
pulled with:

    docker pull "$(python -c 'from rlvr.policy import RELEASE_POLICY; print(RELEASE_POLICY.v3_image)')"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from rlvr.policy import RELEASE_POLICY
from rlvr.v3.archive import extract_archive
from rlvr.v3.artifacts import ArtifactRef
from rlvr.v3.grading import evaluate_repository, evaluate_terminal
from rlvr.v3.identity import RepositoryTaskIdentity, TerminalScriptTaskIdentity
from rlvr.v3.manifest import load_manifest
from rlvr.v3.release import round_policy

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "v3-yamlcpp-e6225d04"
TaskIdentity = RepositoryTaskIdentity | TerminalScriptTaskIdentity


def load(fixture: Path) -> tuple[TaskIdentity, dict[str, ArtifactRef]]:
    raw = json.loads((fixture / "identity.json").read_text())
    model = TerminalScriptTaskIdentity if raw.get("task_type") == "terminal_script_v1" else RepositoryTaskIdentity
    identity = model(**raw)
    refs = {name: ArtifactRef(**value) for name, value in json.loads((fixture / "artifact_refs.json").read_text()).items()}
    return identity, refs


def is_terminal(identity: TaskIdentity) -> bool:
    return identity.task_type == "terminal_script_v1"


def unpack(fixture: Path, name: str, target: Path, scratch: Path) -> Path:
    policy = round_policy(RELEASE_POLICY, dispatch_concurrency=1)
    limits = policy.workspace_archive if name == "workspace" else policy.verifier_archive
    _, refs = load(fixture)
    extract_archive(fixture / f"{name}.tar.zst", refs[name], target, limits, scratch_dir=scratch)
    return target


def extract(fixture: Path, destination: Path) -> int:
    if destination.exists() or destination.is_symlink():
        print(f"refusing to overwrite {destination}", file=sys.stderr)
        return 2
    identity, _ = load(fixture)
    with tempfile.TemporaryDirectory(prefix="hone-task-") as scratch:
        unpack(fixture, "workspace", destination, Path(scratch))
    print(f"workspace: {destination}")
    if is_terminal(identity):
        print(f"task type: terminal script, run with bash in {identity.result_tree_path!r} under the workspace")
    else:
        print(f"working directory: {identity.working_directory}")
        print(f"language: {identity.primary_language}")
    print("instruction:")
    print(f"  {identity.instruction}")
    if is_terminal(identity):
        print("Write a bash script that produces the required results in that workspace, then:")
        print(f"  python scripts/try_task.py --fixture {fixture} grade <your.sh>")
    else:
        print("Produce a unified diff with paths relative to the workspace root, then:")
        print(f"  python scripts/try_task.py --fixture {fixture} grade <your.diff>")
    return 0


def grade(fixture: Path, answer_path: Path) -> int:
    identity, _ = load(fixture)
    terminal = is_terminal(identity)
    limit = RELEASE_POLICY.v3_script_bytes if terminal else RELEASE_POLICY.v3_patch_bytes
    with answer_path.open("rb") as handle:
        answer = handle.read(limit + 1)  # never read more than the validator would accept
    if len(answer) > limit:
        print(f"status: rejected ({'script' if terminal else 'patch'} exceeds the {limit} byte limit)")
        return 1
    if shutil.which("docker") is None:
        print("docker is required to grade", file=sys.stderr)
        return 2
    policy = round_policy(RELEASE_POLICY, dispatch_concurrency=1)
    with tempfile.TemporaryDirectory(prefix="hone-grade-") as temporary:
        root = Path(temporary)
        scratch = root / "scratch"
        scratch.mkdir()
        workspace = unpack(fixture, "workspace", root / "workspace", scratch)
        verifier = unpack(fixture, "verifier", root / "verifier", scratch)
        manifest = load_manifest(verifier, task_type=identity.task_type)
        if terminal:
            result = evaluate_terminal(
                workspace, answer, identity, manifest, verifier,
                policy.supervisor, policy.tree, policy.script, policy.docker_binary, "try-task",
            )
        else:
            result = evaluate_repository(
                workspace, answer, identity, manifest, verifier,
                policy.supervisor, policy.tree, policy.patch, policy.docker_binary, "try-task",
            )
    code = getattr(result.reason_code, "value", None)
    print(f"status: {result.status}" + (f" ({code})" if code else ""))
    if result.reason:
        print(f"reason: {result.reason}")
    for check in result.checks:
        print(f"  {check.check_id}: {check.outcome}")
    if result.failed_check:
        print("failed check:")
        for line in result.failed_check.splitlines():
            print(f"  {line}")
    if result.status == "abandoned":
        print("the validator could not grade this; check Docker and the pinned image", file=sys.stderr)
        return 2
    return 0 if result.status == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="task fixture directory")
    commands = parser.add_subparsers(dest="command", required=True)
    extract_cmd = commands.add_parser("extract", help="write the task workspace to a new directory")
    extract_cmd.add_argument("destination", type=Path)
    grade_cmd = commands.add_parser("grade", help="grade a unified diff or a bash script against the task")
    grade_cmd.add_argument("answer", type=Path)
    args = parser.parse_args(argv)
    if args.command == "extract":
        return extract(args.fixture, args.destination)
    return grade(args.fixture, args.answer)


if __name__ == "__main__":
    sys.exit(main())
