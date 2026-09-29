from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import model_validator

from .wire import (
    BoundedIdentifier,
    BoundedSize,
    BoundedURL,
    HexDigest,
    PositiveBoundedSize,
    Timestamp,
    UID,
    WireModel,
)

ArchiveRole: TypeAlias = Literal[
    "workspace",
    "terminal_environment",
    "dependency_bundle",
    "verifier",
]
UploadRole: TypeAlias = Literal["patch", "script", "trajectory"]
SubmissionRole: TypeAlias = Literal["patch", "script"]
UploadFormat: TypeAlias = Literal[
    "unified_diff_v1",
    "bash_script_v1",
    "trajectory_v1",
]

_UPLOAD_FORMATS: dict[str, str] = {
    "patch": "unified_diff_v1",
    "script": "bash_script_v1",
    "trajectory": "trajectory_v1",
}


def _validate_upload_format(role: str, artifact_format: str) -> None:
    if _UPLOAD_FORMATS.get(role) != artifact_format:
        raise ValueError("artifact role and format do not match")


class ArtifactRef(WireModel):
    artifact_role: ArchiveRole
    artifact_format: Literal["tar_zst_v1"]
    sha256: HexDigest
    compressed_size_bytes: BoundedSize
    expanded_size_bytes: BoundedSize


class MinerArtifactRef(WireModel):
    artifact_role: UploadRole
    artifact_format: UploadFormat
    upload_id: BoundedIdentifier
    sha256: HexDigest
    size_bytes: BoundedSize

    @model_validator(mode="after")
    def validate_role_format(self) -> MinerArtifactRef:
        _validate_upload_format(self.artifact_role, self.artifact_format)
        return self


class UploadSlot(WireModel):
    challenge_id: BoundedIdentifier
    task_id: HexDigest
    uid: UID
    hotkey: BoundedIdentifier
    artifact_role: UploadRole
    artifact_format: UploadFormat
    upload_id: BoundedIdentifier
    upload_url: BoundedURL
    expires_at: Timestamp
    max_bytes: PositiveBoundedSize

    @model_validator(mode="after")
    def validate_role_format(self) -> UploadSlot:
        _validate_upload_format(self.artifact_role, self.artifact_format)
        return self


class MinerSlotSet(WireModel):
    submission: UploadSlot
    trajectory: UploadSlot

    @model_validator(mode="after")
    def validate_slots(self) -> MinerSlotSet:
        if self.submission.artifact_role not in ("patch", "script"):
            raise ValueError("submission slot must contain a patch or script")
        if self.trajectory.artifact_role != "trajectory":
            raise ValueError("trajectory slot must contain a trajectory")
        for field in ("challenge_id", "task_id", "uid", "hotkey"):
            if getattr(self.submission, field) != getattr(self.trajectory, field):
                raise ValueError(f"slot binding mismatch: {field}")
        if self.submission.upload_id == self.trajectory.upload_id:
            raise ValueError("upload slots must have distinct object identities")
        return self


class ArtifactGrant(WireModel):
    uid: UID
    hotkey: BoundedIdentifier
    upload_id: BoundedIdentifier
    sha256: HexDigest
    size_bytes: BoundedSize
    format: Literal["unified_diff_v1", "bash_script_v1"]
    read_url: BoundedURL


class ArtifactFailure(WireModel):
    uid: UID
    hotkey: BoundedIdentifier
    reason: Literal[
        "slot_mismatch",
        "artifact_invalid",
        "trajectory_invalid",
    ]
