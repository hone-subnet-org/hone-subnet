from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .artifacts import (
    ArtifactFailure,
    ArtifactGrant,
    ArtifactRef,
    MinerArtifactRef,
    MinerSlotSet,
)
from .canonical import SAFE_INTEGER_MAX, validate_protocol_string
from .identity import TaskIdentity, compute_task_id
from .wire import (
    BoundedIdentifier,
    BoundedURL,
    HexDigest,
    Timestamp,
    UID,
    WireModel,
    exact_int_literal,
)

ProtocolVersion = exact_int_literal(3)
VerifierPolicy = Literal["command-gold-digest-v1"]

EPISTULA_HEADERS = (
        "Epistula-Version",
        "Epistula-Timestamp",
        "Epistula-Uuid",
        "Epistula-Signed-By",
        "Epistula-Signed-For",
        "Epistula-Request-Signature",
)


def _utf8_size(value: str, maximum: int, field: str) -> str:
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{field} exceeds its UTF-8 byte limit")
    return value


class LeaseRequest(WireModel):
    request_id: BoundedIdentifier


class LeaseResponse(WireModel):
    protocol_version: ProtocolVersion
    challenge_id: BoundedIdentifier
    task_id: HexDigest
    identity: TaskIdentity
    workspace: ArtifactRef
    workspace_url: BoundedURL
    verifier: ArtifactRef
    issued_at: Timestamp
    expires_at: Timestamp
    slot_pool: Annotated[list[MinerSlotSet], Field(min_length=1, max_length=1_024)]
    commit_min_signed_responses: Annotated[int, Field(gt=0, le=1_024)]

    @model_validator(mode="after")
    def validate_bindings(self) -> LeaseResponse:
        if self.task_id != compute_task_id(self.identity):
            raise ValueError("task identity does not match task_id")
        expected_role = (
            "workspace"
            if self.identity.task_type == "repository_patch_v1"
            else "terminal_environment"
        )
        expected_digest = (
            self.identity.workspace_sha256
            if self.identity.task_type == "repository_patch_v1"
            else self.identity.environment_sha256
        )
        if self.workspace.artifact_role != expected_role or self.workspace.sha256 != expected_digest:
            raise ValueError("workspace does not match task identity")
        if self.verifier.artifact_role != "verifier" or self.verifier.sha256 != self.identity.verifier_sha256:
            raise ValueError("verifier does not match task identity")
        if self.expires_at <= self.issued_at:
            raise ValueError("lease must expire after issuance")
        if self.commit_min_signed_responses > len(self.slot_pool):
            raise ValueError("commit quorum exceeds the slot pool")
        upload_ids: set[str] = set()
        uids: set[int] = set()
        hotkeys: set[str] = set()
        expected_submission = "patch" if self.identity.task_type == "repository_patch_v1" else "script"
        for slots in self.slot_pool:
            if slots.submission.challenge_id != self.challenge_id or slots.submission.task_id != self.task_id:
                raise ValueError("slot does not match lease")
            if slots.submission.artifact_role != expected_submission:
                raise ValueError("submission slot does not match task type")
            if slots.submission.uid in uids or slots.submission.hotkey in hotkeys:
                raise ValueError("slot pool contains a duplicate miner")
            uids.add(slots.submission.uid)
            hotkeys.add(slots.submission.hotkey)
            for upload_id in (slots.submission.upload_id, slots.trajectory.upload_id):
                if upload_id in upload_ids:
                    raise ValueError("slot pool contains a duplicate upload_id")
                upload_ids.add(upload_id)
        return self


class MinerTaskRequest(WireModel):
    protocol_version: ProtocolVersion
    challenge_id: BoundedIdentifier
    task_id: HexDigest
    identity: TaskIdentity
    workspace: ArtifactRef
    workspace_url: BoundedURL
    expires_at: Timestamp
    slots: MinerSlotSet

    @model_validator(mode="after")
    def validate_slot_binding(self) -> MinerTaskRequest:
        if self.task_id != compute_task_id(self.identity):
            raise ValueError("task identity does not match task_id")
        expected_workspace_role = (
            "workspace"
            if self.identity.task_type == "repository_patch_v1"
            else "terminal_environment"
        )
        expected_workspace_digest = (
            self.identity.workspace_sha256
            if self.identity.task_type == "repository_patch_v1"
            else self.identity.environment_sha256
        )
        if (
            self.workspace.artifact_role != expected_workspace_role
            or self.workspace.sha256 != expected_workspace_digest
        ):
            raise ValueError("workspace does not match task identity")
        if self.slots.submission.challenge_id != self.challenge_id:
            raise ValueError("dispatch slots do not match challenge")
        if self.slots.submission.task_id != self.task_id:
            raise ValueError("dispatch slots do not match task")
        expected_submission = (
            "patch" if self.identity.task_type == "repository_patch_v1" else "script"
        )
        if self.slots.submission.artifact_role != expected_submission:
            raise ValueError("submission slot does not match task type")
        return self


class MinerTaskResponse(WireModel):
    protocol_version: ProtocolVersion
    challenge_id: BoundedIdentifier
    task_id: HexDigest
    response_type: Literal["repository_patch_v1", "terminal_script_v1"]
    submission: MinerArtifactRef
    trajectory: MinerArtifactRef

    @model_validator(mode="after")
    def validate_artifacts(self) -> MinerTaskResponse:
        expected = "patch" if self.response_type == "repository_patch_v1" else "script"
        if self.submission.artifact_role != expected:
            raise ValueError("submission does not match response_type")
        if self.trajectory.artifact_role != "trajectory":
            raise ValueError("trajectory artifact is required")
        if self.submission.upload_id == self.trajectory.upload_id:
            raise ValueError("submission and trajectory upload IDs must differ")
        return self


class MinerSubmission(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )

    uid: UID
    hotkey: BoundedIdentifier
    request_id: BoundedIdentifier
    response_body: str
    response_headers: dict[str, str]
    error: str
    latency_ms: Annotated[int, Field(ge=0, le=SAFE_INTEGER_MAX)]

    @field_validator("hotkey", "request_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        validate_protocol_string(value)
        return value

    @field_validator("response_body")
    @classmethod
    def validate_body_size(cls, value: str) -> str:
        return _utf8_size(value, 16_384, "response_body")

    @field_validator("error")
    @classmethod
    def validate_error_size(cls, value: str) -> str:
        return _utf8_size(value, 4_096, "error")

    @field_validator("response_headers")
    @classmethod
    def validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        for header_value in value.values():
            _utf8_size(header_value, 512, "response header")
        return value

    @model_validator(mode="after")
    def validate_response_shape(self) -> MinerSubmission:
        if self.response_body:
            if self.error or frozenset(self.response_headers) != frozenset(EPISTULA_HEADERS):
                raise ValueError("signed response has an invalid envelope")
        elif not self.error or self.response_headers:
            raise ValueError("failed response has an invalid envelope")
        return self


class ChallengeCommitRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )

    protocol_version: ProtocolVersion
    challenge_id: BoundedIdentifier
    submissions: Annotated[list[MinerSubmission], Field(max_length=1_024)]

    @field_validator("challenge_id")
    @classmethod
    def validate_challenge_id(cls, value: str) -> str:
        validate_protocol_string(value)
        return value

    @model_validator(mode="after")
    def reject_duplicates(self) -> ChallengeCommitRequest:
        uids = [item.uid for item in self.submissions]
        hotkeys = [item.hotkey for item in self.submissions]
        if len(uids) != len(set(uids)) or len(hotkeys) != len(set(hotkeys)):
            raise ValueError("commit contains duplicate miner identities")
        return self


class CommitRevealResponse(WireModel):
    protocol_version: ProtocolVersion
    challenge_id: BoundedIdentifier
    task_id: HexDigest
    verifier: ArtifactRef
    verifier_policy: VerifierPolicy
    verifier_url: BoundedURL
    grading_expires_at: Timestamp
    submission_grants: Annotated[list[ArtifactGrant], Field(max_length=1_024)]
    artifact_failures: Annotated[list[ArtifactFailure], Field(max_length=1_024)]

    @model_validator(mode="after")
    def validate_results(self) -> CommitRevealResponse:
        if self.verifier.artifact_role != "verifier":
            raise ValueError("commit response must contain a verifier")
        results = [*self.submission_grants, *self.artifact_failures]
        if len(results) > 1_024:
            raise ValueError("commit response contains too many results")
        uids = [item.uid for item in results]
        hotkeys = [item.hotkey for item in results]
        if len(uids) != len(set(uids)) or len(hotkeys) != len(set(hotkeys)):
            raise ValueError("commit response contains duplicate miner identities")
        return self


class FeedbackVerdict(WireModel):
    uid: UID
    hotkey: BoundedIdentifier
    passed: StrictBool
    grading_duration_ms: Annotated[int, Field(ge=0, le=SAFE_INTEGER_MAX)]


class ChallengeFeedbackRequest(WireModel):
    protocol_version: ProtocolVersion
    challenge_id: BoundedIdentifier
    task_id: HexDigest
    verdicts: Annotated[list[FeedbackVerdict], Field(min_length=1, max_length=1_024)]

    @model_validator(mode="after")
    def reject_duplicates(self) -> ChallengeFeedbackRequest:
        uids = [item.uid for item in self.verdicts]
        hotkeys = [item.hotkey for item in self.verdicts]
        if len(uids) != len(set(uids)) or len(hotkeys) != len(set(hotkeys)):
            raise ValueError("feedback contains duplicate miner identities")
        return self


class ChallengeFeedbackResponse(WireModel):
    protocol_version: ProtocolVersion
    challenge_id: BoundedIdentifier
    task_id: HexDigest


def serialize_commit_request(request: ChallengeCommitRequest) -> bytes:
    if type(request) is not ChallengeCommitRequest:
        raise TypeError("request must be a V3 commit request")
    validated = ChallengeCommitRequest.model_validate(request.model_dump(mode="python"))
    return validated.model_dump_json().encode("utf-8")


def serialize_feedback_request(request: ChallengeFeedbackRequest) -> bytes:
    if type(request) is not ChallengeFeedbackRequest:
        raise TypeError("request must be a V3 feedback request")
    validated = ChallengeFeedbackRequest.model_validate(request.model_dump(mode="python"))
    return validated.model_dump_json().encode("utf-8")


def derive_miner_request_id(challenge_id: str, uid: int, hotkey: str) -> str:
    material = f"hone-v3-miner-request\0{challenge_id}\0{uid}\0{hotkey}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def validate_miner_response(
    task: MinerTaskRequest, response: MinerTaskResponse
) -> None:
    if type(task) is not MinerTaskRequest or type(response) is not MinerTaskResponse:
        raise TypeError("validated V3 miner task and response are required")
    if response.challenge_id != task.challenge_id or response.task_id != task.task_id:
        raise ValueError("miner response does not match its task")
    if response.response_type != task.identity.task_type:
        raise ValueError("miner response type does not match its task")
    for artifact, slot in (
        (response.submission, task.slots.submission),
        (response.trajectory, task.slots.trajectory),
    ):
        if (
            artifact.upload_id != slot.upload_id
            or artifact.artifact_role != slot.artifact_role
            or artifact.artifact_format != slot.artifact_format
            or artifact.size_bytes > slot.max_bytes
        ):
            raise ValueError("miner artifact does not match its upload slot")


def validate_commit_reveal(
    lease: LeaseResponse,
    request: ChallengeCommitRequest,
    response: CommitRevealResponse,
) -> None:
    if type(lease) is not LeaseResponse or type(request) is not ChallengeCommitRequest or type(response) is not CommitRevealResponse:
        raise TypeError("validated V3 lease, commit, and reveal models are required")
    if request.challenge_id != lease.challenge_id or response.challenge_id != lease.challenge_id:
        raise ValueError("commit challenge binding mismatch")
    if response.task_id != lease.task_id:
        raise ValueError("commit task binding mismatch")
    if response.verifier.sha256 != lease.verifier.sha256:
        raise ValueError("commit verifier binding mismatch")
    if response.verifier_policy != lease.identity.verifier_policy:
        raise ValueError("commit verifier policy mismatch")
    expected = {(item.uid, item.hotkey) for item in request.submissions}
    actual = {
        (item.uid, item.hotkey)
        for item in [*response.submission_grants, *response.artifact_failures]
    }
    if actual != expected:
        raise ValueError("commit results do not match submitted miners")
