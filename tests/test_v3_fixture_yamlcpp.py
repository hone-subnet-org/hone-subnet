"""One real repository task, graded exactly as a validator would.

The fixture under tests/fixtures/v3-yamlcpp-e6225d04 is a task the problem
server issued once and retired: yaml-cpp with a private EventArchive feature
and two planted defects, six hidden checks, and a reference fix. It is not in
the mainnet task pool. The fast tests need no Docker; the grading tests run
in the pinned sandbox image and are skipped without it.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rlvr.policy import RELEASE_POLICY
from rlvr.v3.archive import extract_archive
from rlvr.v3.artifacts import ArtifactRef
from rlvr.v3.grading import evaluate_repository
from rlvr.v3.identity import RepositoryTaskIdentity, compute_task_id
from rlvr.v3.manifest import load_manifest
from rlvr.v3.patch import PatchLimits, static_rejection
from rlvr.v3.reasons import MinerReason, Stage
from rlvr.v3.release import round_policy

FIXTURE = Path(__file__).parent / "fixtures" / "v3-yamlcpp-e6225d04"
CHECK_IDS = [
    "00-d1-alias-binary-pick", "01-d1-alias-binary-lone", "02-d1-alias-key-no-prefix",
    "03-d2-nul-trigger-rtrip", "04-d2-nul-quota-admit", "05-d2-nul-handmade-ckpt",
]


def refs() -> dict[str, ArtifactRef]:
    raw = json.loads((FIXTURE / "artifact_refs.json").read_text())
    return {name: ArtifactRef(**value) for name, value in raw.items()}


def identity() -> RepositoryTaskIdentity:
    return RepositoryTaskIdentity(**json.loads((FIXTURE / "identity.json").read_text()))


def policy():
    return round_policy(RELEASE_POLICY, dispatch_concurrency=1)


def unpack(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    scratch = tmp_path / f"scratch-{name}"
    scratch.mkdir(parents=True, exist_ok=True)
    limits = policy().workspace_archive if name == "workspace" else policy().verifier_archive
    extract_archive(FIXTURE / f"{name}.tar.zst", refs()[name], target, limits, scratch_dir=scratch)
    return target


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
    assert identity().workspace_sha256 == refs()["workspace"].sha256
    assert identity().verifier_sha256 == refs()["verifier"].sha256


def test_archives_extract_under_release_limits_and_the_manifest_has_six_checks(tmp_path):
    workspace = unpack(tmp_path, "workspace")
    verifier = unpack(tmp_path, "verifier")
    assert (workspace / "src" / "eventarchive_store.cpp").is_file()
    manifest = load_manifest(verifier, task_type="repository_patch_v1")
    assert [check.check_id for check in manifest.checks] == CHECK_IDS
    assert manifest.setup is None


def test_reference_diff_passes_the_static_rules():
    patch = (FIXTURE / "reference.diff").read_bytes()
    assert static_rejection(patch, PatchLimits(RELEASE_POLICY.v3_patch_bytes, 30)) is None


# --------------------------------------------------------------------------- #
# grading in the pinned sandbox image
# --------------------------------------------------------------------------- #
def docker_ready() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    probe = subprocess.run(
        [docker, "image", "inspect", RELEASE_POLICY.v3_image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False,
    )
    return probe.returncode == 0


needs_image = pytest.mark.skipif(not docker_ready(), reason="pinned Docker image unavailable")


def grade(tmp_path: Path, patch: bytes, tag: str):
    workspace = unpack(tmp_path / tag, "workspace")
    verifier = unpack(tmp_path / tag, "verifier")
    manifest = load_manifest(verifier, task_type="repository_patch_v1")
    rp = policy()
    return evaluate_repository(
        workspace, patch, identity(), manifest, verifier,
        rp.supervisor, rp.tree, rp.patch, rp.docker_binary, f"fixture-{tag}",
    )


def harmless_patch(tmp_path: Path) -> bytes:
    """A valid patch that changes nothing the checks look at."""
    workspace = unpack(tmp_path / "probe", "workspace")
    before = (workspace / "CONTRIBUTING.md").read_text().splitlines(keepends=True)
    after = before + ["\nNothing to see here.\n"]
    return "".join(difflib.unified_diff(before, after, "a/CONTRIBUTING.md", "b/CONTRIBUTING.md")).encode()


def d1_only_patch() -> bytes:
    """The reference diff's first file only: fixes defect d1, leaves d2."""
    text = (FIXTURE / "reference.diff").read_text()
    second = text.index("--- a/src/eventarchive_store.cpp")
    return text[:second].encode()


@needs_image
def test_unfixed_workspace_fails_the_first_check(tmp_path):
    result = grade(tmp_path, harmless_patch(tmp_path), "unfixed")
    assert result.status == "failed"
    assert result.reason_code is MinerReason.CHECK_FAILED and result.stage is Stage.CHECK
    assert [(c.check_id, c.outcome) for c in result.checks][:2] == [(CHECK_IDS[0], "failed"), (CHECK_IDS[1], "skipped")]


@needs_image
def test_reference_fix_passes_all_six_checks(tmp_path):
    result = grade(tmp_path, (FIXTURE / "reference.diff").read_bytes(), "reference")
    assert result.status == "passed", result.reason
    assert [c.outcome for c in result.checks] == ["passed"] * 6


@needs_image
def test_fixing_only_the_first_defect_fails_at_the_first_d2_check(tmp_path):
    result = grade(tmp_path, d1_only_patch(), "d1only")
    assert result.status == "failed"
    outcomes = [(c.check_id, c.outcome) for c in result.checks]
    assert outcomes[:3] == [(CHECK_IDS[0], "passed"), (CHECK_IDS[1], "passed"), (CHECK_IDS[2], "passed")]
    assert outcomes[3] == (CHECK_IDS[3], "failed")


# --------------------------------------------------------------------------- #
# the miner-facing script
# --------------------------------------------------------------------------- #
def test_try_task_extracts_the_workspace_and_prints_the_instruction(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("try_task", Path(__file__).parent.parent / "scripts" / "try_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["extract", str(tmp_path / "work")]) == 0
    out = capsys.readouterr().out
    assert (tmp_path / "work" / "src" / "eventarchive_store.cpp").is_file()
    assert "EventArchive" in out
    hint = next(line.strip() for line in out.splitlines() if "grade <your.diff>" in line)
    argv = hint.split()[2:]  # drop "python scripts/try_task.py"
    assert argv[:2] == ["--fixture", str(FIXTURE)] and argv[2:] == ["grade", "<your.diff>"]
    # the hinted option order parses with the real parser (grade itself needs Docker)
    assert module.main(argv[:2] + ["extract", str(tmp_path / "work2")]) == 0
    assert module.main(["extract", str(tmp_path / "work")]) == 2  # never overwrites
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    assert module.main(["extract", str(dangling)]) == 2  # nor follows a symlink
    assert not (tmp_path / "missing").exists()


def test_try_task_rejects_an_oversized_patch_without_reading_it(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("try_task", Path(__file__).parent.parent / "scripts" / "try_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    limit = RELEASE_POLICY.v3_patch_bytes
    big = tmp_path / "big.diff"
    with big.open("wb") as handle:
        handle.truncate(limit * 8)  # sparse: 8x the limit on disk, nothing to read
    assert module.main(["grade", str(big)]) == 1
    assert "exceeds" in capsys.readouterr().out


@needs_image
def test_try_task_grades_a_patch_like_a_validator(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("try_task", Path(__file__).parent.parent / "scripts" / "try_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["grade", str(FIXTURE / "reference.diff")]) == 0
    assert "status: passed" in capsys.readouterr().out
    half = tmp_path / "half.diff"
    half.write_bytes(d1_only_patch())
    assert module.main(["grade", str(half)]) == 1
    out = capsys.readouterr().out
    assert "status: failed (check_failed)" in out and f"{CHECK_IDS[3]}: failed" in out
