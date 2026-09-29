#!/usr/bin/env python
"""Work a real task locally and grade your patch exactly as a validator would.

The fixture under tests/fixtures/v3-yamlcpp-e6225d04 is a task the problem
server issued once and retired. Use it to check your miner end to end:

    # 1. get the workspace and the instruction
    python scripts/try_task.py extract /path/to/work

    # 2. produce a unified diff against that workspace, however you like

    # 3. grade it in the pinned sandbox image (needs Docker)
    python scripts/try_task.py grade /path/to/fix.diff

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
from rlvr.v3.grading import evaluate_repository
from rlvr.v3.identity import RepositoryTaskIdentity
from rlvr.v3.manifest import load_manifest
from rlvr.v3.release import round_policy

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "v3-yamlcpp-e6225d04"


def load(fixture: Path) -> tuple[RepositoryTaskIdentity, dict[str, ArtifactRef]]:
    identity = RepositoryTaskIdentity(**json.loads((fixture / "identity.json").read_text()))
    refs = {name: ArtifactRef(**value) for name, value in json.loads((fixture / "artifact_refs.json").read_text()).items()}
    return identity, refs


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
    print(f"working directory: {identity.working_directory}")
    print(f"language: {identity.primary_language}")
    print("instruction:")
    print(f"  {identity.instruction}")
    print("Produce a unified diff with paths relative to the workspace root, then:")
    print(f"  python scripts/try_task.py --fixture {fixture} grade <your.diff>")
    return 0


def grade(fixture: Path, patch_path: Path) -> int:
    limit = RELEASE_POLICY.v3_patch_bytes
    with patch_path.open("rb") as handle:
        patch = handle.read(limit + 1)  # never read more than the validator would accept
    if len(patch) > limit:
        print(f"status: rejected (patch exceeds the {limit} byte limit)")
        return 1
    if shutil.which("docker") is None:
        print("docker is required to grade", file=sys.stderr)
        return 2
    identity, _ = load(fixture)
    policy = round_policy(RELEASE_POLICY, dispatch_concurrency=1)
    with tempfile.TemporaryDirectory(prefix="hone-grade-") as temporary:
        root = Path(temporary)
        scratch = root / "scratch"
        scratch.mkdir()
        workspace = unpack(fixture, "workspace", root / "workspace", scratch)
        verifier = unpack(fixture, "verifier", root / "verifier", scratch)
        manifest = load_manifest(verifier, task_type=identity.task_type)
        result = evaluate_repository(
            workspace, patch, identity, manifest, verifier,
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
    grade_cmd = commands.add_parser("grade", help="grade a unified diff against the task")
    grade_cmd.add_argument("patch", type=Path)
    args = parser.parse_args(argv)
    if args.command == "extract":
        return extract(args.fixture, args.destination)
    return grade(args.fixture, args.patch)


if __name__ == "__main__":
    sys.exit(main())
