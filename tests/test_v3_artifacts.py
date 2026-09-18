"""V3 shared artifact and upload-slot seam.

Contract for module ``rlvr.v3.artifacts``:

    ArtifactRef            server archive: artifact_role in
                           workspace|terminal_environment|dependency_bundle|
                           verifier, artifact_format exactly tar_zst_v1,
                           sha256, compressed_size_bytes, expanded_size_bytes
    MinerArtifactRef       miner upload: artifact_role patch|script|trajectory
                           with the exact format mapping unified_diff_v1|
                           bash_script_v1|trajectory_v1, upload_id, sha256,
                           size_bytes
    UploadSlot             challenge_id, task_id, uid, hotkey, artifact_role,
                           artifact_format (same mapping), upload_id (write-once
                           object identity), upload_url, expires_at, max_bytes
    MinerSlotSet           submission (patch|script slot) + trajectory slot,
                           identical challenge/task/uid/hotkey binding
    ArtifactGrant          validator-only read grant for patch|script:
                           challenge_id, task_id, uid, hotkey, upload_id,
                           artifact_role, artifact_format, sha256, size_bytes,
                           read_url
    CommittedSubmissionDisposition        disposition "committed", grant
    MinerFaultSubmissionDisposition       disposition missing_upload|
                                          digest_mismatch|malformed_trajectory|
                                          expired_slot, no grant
    InfrastructureFailureSubmissionDisposition  "infrastructure_failure"
    SubmissionDisposition  discriminated union of the three on "disposition"

All inherit WireModel (extra forbidden, frozen, strict). Bounds: uid 0..1023,
IDs and hotkey 1..128 characters, task_id and digests lowercase 64 hex, URLs
1..8192 opaque characters, times and sizes 0..2^53-1, slot max_bytes
1..2^53-1.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

SAFE_MAX = 2**53 - 1
HEX = "a" * 64
HEX2 = "b" * 64


def _mod():
    from rlvr.v3 import artifacts

    return artifacts


def archive(**over):
    base = {
        "artifact_role": "workspace",
        "artifact_format": "tar_zst_v1",
        "sha256": HEX,
        "compressed_size_bytes": 10,
        "expanded_size_bytes": 20,
    }
    base.update(over)
    return base


def upload_ref(**over):
    base = {
        "artifact_role": "patch",
        "artifact_format": "unified_diff_v1",
        "upload_id": "up-1",
        "sha256": HEX,
        "size_bytes": 5,
    }
    base.update(over)
    return base


def binding(**over):
    base = {"challenge_id": "chal-1", "task_id": HEX, "uid": 7, "hotkey": "hk-7"}
    base.update(over)
    return base


def slot(**over):
    base = {
        **binding(),
        "artifact_role": "patch",
        "artifact_format": "unified_diff_v1",
        "upload_id": "up-patch",
        "upload_url": "https://uploads.example/one",
        "expires_at": 1000,
        "max_bytes": 1024,
    }
    base.update(over)
    return base


def trajectory_slot(**over):
    base = slot(
        artifact_role="trajectory",
        artifact_format="trajectory_v1",
        upload_id="up-traj",
        upload_url="https://uploads.example/two",
    )
    base.update(over)
    return base


def grant(**over):
    base = {
        "uid": 7,
        "hotkey": "hk-7",
        "upload_id": "up-patch",
        "sha256": HEX,
        "size_bytes": 5,
        "format": "unified_diff_v1",
        "read_url": "https://reads.example/one",
    }
    base.update(over)
    return base


def failure(**over):
    base = {"uid": 7, "hotkey": "hk-7", "reason": "artifact_invalid"}
    base.update(over)
    return base


def slot_set(**over):
    base = {"submission": slot(), "trajectory": trajectory_slot()}
    base.update(over)
    return base


BINDING_BOUNDS = [
    pytest.param({"uid": -1}, id="uid-neg"),
    pytest.param({"uid": 1024}, id="uid-1024"),
    pytest.param({"hotkey": ""}, id="hotkey-empty"),
    pytest.param({"hotkey": "h" * 129}, id="hotkey-129"),
    pytest.param({"challenge_id": ""}, id="chal-empty"),
    pytest.param({"challenge_id": "c" * 129}, id="chal-129"),
    pytest.param({"task_id": HEX.upper()}, id="task-upper"),
    pytest.param({"task_id": HEX[:-1]}, id="task-63"),
]

DIGEST_BOUNDS = [
    pytest.param({"sha256": HEX.upper()}, id="digest-upper"),
    pytest.param({"sha256": HEX[:-1]}, id="digest-63"),
    pytest.param({"sha256": HEX + "a"}, id="digest-65"),
]


# --------------------------------------------------------------------------- #
# Valid models and exact serialization
# --------------------------------------------------------------------------- #
def test_archive_ref_round_trips_and_canonicalizes_exactly():
    m = _mod()
    from rlvr.v3.canonical import canonical_json_bytes

    ref = m.ArtifactRef(**archive())
    assert ref.model_dump(mode="json") == archive()

    expected = (
        b'{"artifact_format":"tar_zst_v1",'
        b'"artifact_role":"workspace",'
        b'"compressed_size_bytes":10,'
        b'"expanded_size_bytes":20,'
        b'"sha256":"' + HEX.encode() + b'"}'
    )
    assert canonical_json_bytes(ref) == expected


@pytest.mark.parametrize(
    "role", ["workspace", "terminal_environment", "dependency_bundle", "verifier"]
)
def test_archive_roles_accept_tar_zst(role):
    m = _mod()
    assert m.ArtifactRef(**archive(artifact_role=role)).artifact_role == role


@pytest.mark.parametrize(
    "role, fmt",
    [
        ("patch", "unified_diff_v1"),
        ("script", "bash_script_v1"),
        ("trajectory", "trajectory_v1"),
    ],
)
def test_upload_ref_accepts_exact_role_format_pairs(role, fmt):
    m = _mod()
    payload = upload_ref(artifact_role=role, artifact_format=fmt)
    assert m.MinerArtifactRef(**payload).model_dump(mode="json") == payload


def test_slot_set_and_grant_round_trip():
    m = _mod()
    slots = m.MinerSlotSet(**slot_set())
    assert slots.submission.upload_id == "up-patch"
    assert slots.trajectory.artifact_role == "trajectory"
    assert m.ArtifactGrant(**grant()).model_dump(mode="json") == grant()


def test_script_submission_slot_is_valid():
    m = _mod()
    script_slot = slot(artifact_role="script", artifact_format="bash_script_v1")
    slots = m.MinerSlotSet(**slot_set(submission=script_slot))
    assert slots.submission.artifact_format == "bash_script_v1"


# --------------------------------------------------------------------------- #
# Inherited WireModel behavior: extra forbidden, frozen, strict input
# --------------------------------------------------------------------------- #
FROZEN_TABLE = [
    pytest.param("ArtifactRef", archive, "artifact_role", "verifier", id="ArtifactRef"),
    pytest.param("MinerArtifactRef", upload_ref, "upload_id", "up-2", id="MinerArtifactRef"),
    pytest.param("UploadSlot", slot, "max_bytes", 2048, id="UploadSlot"),
    pytest.param("MinerSlotSet", slot_set, "trajectory", None, id="MinerSlotSet"),
    pytest.param("ArtifactGrant", grant, "read_url", "https://reads.example/two", id="ArtifactGrant"),
    pytest.param("ArtifactFailure", failure, "uid", 8, id="ArtifactFailure"),
]


@pytest.mark.parametrize("name, build, field, new_value", FROZEN_TABLE)
def test_models_forbid_extra_and_are_frozen(name, build, field, new_value):
    model = getattr(_mod(), name)
    with pytest.raises(ValidationError):
        model(**build(), unexpected=1)
    instance = model(**build())
    assert field in type(instance).model_fields
    with pytest.raises(ValidationError):
        setattr(instance, field, new_value)


def test_slot_set_rejects_coerced_or_foreign_inputs():
    m = _mod()
    with pytest.raises(ValidationError):
        m.MinerSlotSet(**slot_set(submission="not-a-slot"))
    with pytest.raises(ValidationError):
        m.MinerSlotSet(**slot_set(submission=slot(uid="7")))
    with pytest.raises(ValidationError):
        m.MinerSlotSet(**slot_set(trajectory=trajectory_slot(max_bytes=True)))


# --------------------------------------------------------------------------- #
# Role/format cross-use is rejected by construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "over",
    [
        {"artifact_format": "unified_diff_v1"},
        {"artifact_role": "patch"},
        {"artifact_role": "trajectory"},
        {"artifact_format": "tar_zst_v2"},
    ],
    ids=["upload-format-on-archive", "patch-role-on-archive", "trajectory-role-on-archive", "unknown-format"],
)
def test_archive_ref_rejects_foreign_roles_and_formats(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.ArtifactRef(**archive(**over))


@pytest.mark.parametrize(
    "role, fmt",
    [
        ("patch", "trajectory_v1"),
        ("patch", "bash_script_v1"),
        ("script", "unified_diff_v1"),
        ("trajectory", "unified_diff_v1"),
        ("workspace", "tar_zst_v1"),
        ("patch", "tar_zst_v1"),
    ],
)
def test_upload_ref_rejects_role_format_mismatch(role, fmt):
    m = _mod()
    with pytest.raises(ValidationError):
        m.MinerArtifactRef(**upload_ref(artifact_role=role, artifact_format=fmt))


@pytest.mark.parametrize(
    "role, fmt",
    [("script", "unified_diff_v1"), ("trajectory", "bash_script_v1"), ("workspace", "tar_zst_v1")],
)
def test_upload_slot_rejects_role_format_mismatch(role, fmt):
    m = _mod()
    with pytest.raises(ValidationError):
        m.UploadSlot(**slot(artifact_role=role, artifact_format=fmt))


@pytest.mark.parametrize("value", ["trajectory_v1", "shell_script_v1", "tar_zst_v1"])
def test_grant_accepts_only_submission_formats(value):
    m = _mod()
    with pytest.raises(ValidationError):
        m.ArtifactGrant(**grant(format=value))


# --------------------------------------------------------------------------- #
# Bounds and literal strictness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "over",
    DIGEST_BOUNDS
    + [
        pytest.param({"compressed_size_bytes": -1}, id="compressed-neg"),
        pytest.param({"expanded_size_bytes": SAFE_MAX + 1}, id="expanded-over"),
    ],
)
def test_archive_ref_bounds(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.ArtifactRef(**archive(**over))


@pytest.mark.parametrize(
    "over",
    DIGEST_BOUNDS
    + [
        pytest.param({"upload_id": ""}, id="upload-empty"),
        pytest.param({"upload_id": "u" * 129}, id="upload-129"),
        pytest.param({"size_bytes": -1}, id="size-neg"),
        pytest.param({"size_bytes": SAFE_MAX + 1}, id="size-over"),
    ],
)
def test_upload_ref_bounds(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.MinerArtifactRef(**upload_ref(**over))


def test_upload_ref_boundary_values_accepted():
    m = _mod()
    assert m.MinerArtifactRef(**upload_ref(size_bytes=0)).size_bytes == 0
    ok = m.MinerArtifactRef(**upload_ref(size_bytes=SAFE_MAX, upload_id="u" * 128))
    assert ok.size_bytes == SAFE_MAX and len(ok.upload_id) == 128


@pytest.mark.parametrize(
    "build, model_name",
    [
        pytest.param(slot, "UploadSlot", id="slot"),
        pytest.param(grant, "ArtifactGrant", id="grant"),
        pytest.param(failure, "ArtifactFailure", id="failure"),
    ],
)
@pytest.mark.parametrize("over", BINDING_BOUNDS)
def test_binding_bounds_apply_to_every_bound_model(build, model_name, over):
    model = getattr(_mod(), model_name)
    with pytest.raises(ValidationError):
        model(**build(**over))


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"upload_id": ""}, id="upload-empty"),
        pytest.param({"upload_id": "u" * 129}, id="upload-129"),
        pytest.param({"upload_url": ""}, id="url-empty"),
        pytest.param({"upload_url": "u" * 8193}, id="url-8193"),
        pytest.param({"expires_at": -1}, id="expires-neg"),
        pytest.param({"expires_at": SAFE_MAX + 1}, id="expires-over"),
        pytest.param({"max_bytes": 0}, id="max-bytes-zero"),
        pytest.param({"max_bytes": SAFE_MAX + 1}, id="max-bytes-over"),
    ],
)
def test_upload_slot_specific_bounds(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.UploadSlot(**slot(**over))


def test_upload_slot_boundary_values_accepted():
    m = _mod()
    ok = slot(uid=1023, hotkey="h" * 128, upload_url="u" * 8192, expires_at=SAFE_MAX, max_bytes=SAFE_MAX)
    assert m.UploadSlot(**ok).uid == 1023


@pytest.mark.parametrize(
    "over",
    DIGEST_BOUNDS
    + [
        pytest.param({"size_bytes": -1}, id="size-neg"),
        pytest.param({"size_bytes": SAFE_MAX + 1}, id="size-over"),
        pytest.param({"upload_id": ""}, id="upload-empty"),
        pytest.param({"read_url": ""}, id="url-empty"),
        pytest.param({"read_url": "r" * 8193}, id="url-8193"),
    ],
)
def test_grant_specific_bounds(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.ArtifactGrant(**grant(**over))


# --------------------------------------------------------------------------- #
# Slot-set binding and role constraints
# --------------------------------------------------------------------------- #
def test_slot_set_requires_submission_and_trajectory_roles():
    m = _mod()
    with pytest.raises(ValidationError):
        m.MinerSlotSet(**slot_set(submission=trajectory_slot()))
    with pytest.raises(ValidationError):
        m.MinerSlotSet(**slot_set(trajectory=slot()))


@pytest.mark.parametrize(
    "over",
    [{"challenge_id": "chal-2"}, {"task_id": HEX2}, {"uid": 8}, {"hotkey": "hk-8"}],
    ids=["challenge", "task", "uid", "hotkey"],
)
def test_slot_set_rejects_binding_mismatch_between_slots(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.MinerSlotSet(**slot_set(trajectory=trajectory_slot(**over)))


def test_slot_set_rejects_shared_upload_id():
    m = _mod()
    with pytest.raises(ValidationError):
        m.MinerSlotSet(**slot_set(trajectory=trajectory_slot(upload_id="up-patch")))


# --------------------------------------------------------------------------- #
# Commit artifact failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reason", ["slot_mismatch", "artifact_invalid", "trajectory_invalid"])
def test_artifact_failure_accepts_exact_reasons(reason):
    assert _mod().ArtifactFailure(**failure(reason=reason)).reason == reason


@pytest.mark.parametrize("reason", ["missing_upload", "infrastructure_failure", "unknown"])
def test_artifact_failure_rejects_unknown_reasons(reason):
    with pytest.raises(ValidationError):
        _mod().ArtifactFailure(**failure(reason=reason))
