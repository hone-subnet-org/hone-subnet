from __future__ import annotations

import pytest
from pydantic import ValidationError

from rlvr.v3.api import (
    EPISTULA_HEADERS,
    ChallengeCommitRequest,
    ChallengeFeedbackRequest,
    ChallengeFeedbackResponse,
    CommitRevealResponse,
    FeedbackVerdict,
    LeaseResponse,
    MinerSubmission,
    MinerTaskRequest,
    MinerTaskResponse,
    serialize_commit_request,
    serialize_feedback_request,
    validate_commit_reveal,
)
from rlvr.v3.artifacts import ArtifactRef, MinerSlotSet, UploadSlot
from rlvr.v3.identity import RepositoryTaskIdentity, compute_task_id

HEX_A = "a" * 64
HEX_B = "b" * 64


def identity():
    return RepositoryTaskIdentity(
        task_kind="bug_fix",
        instruction="Fix it",
        primary_language="python",
        workspace_sha256=HEX_A,
        verifier_sha256=HEX_B,
        execution_profile_id="repo-polyglot-v1",
        working_directory=".",
        verifier_policy="command-gold-digest-v1",
        authoring_version="fixture-v1",
    )


def archive(role, digest):
    return ArtifactRef(
        artifact_role=role,
        artifact_format="tar_zst_v1",
        sha256=digest,
        compressed_size_bytes=10,
        expanded_size_bytes=10240,
    )


def slot(role, fmt, upload_id, *, uid=7, hotkey="hk-7", **over):
    ident = identity()
    fields = dict(
        challenge_id="chal-1",
        task_id=compute_task_id(ident),
        uid=uid,
        hotkey=hotkey,
        artifact_role=role,
        artifact_format=fmt,
        upload_id=upload_id,
        upload_url=f"https://uploads.invalid/{upload_id}",
        expires_at=200,
        max_bytes=100,
    )
    fields.update(over)
    return UploadSlot(**fields)


def slot_set(*, uid=7, hotkey="hk-7", prefix="u"):
    return MinerSlotSet(
        submission=slot("patch", "unified_diff_v1", f"{prefix}-submission", uid=uid, hotkey=hotkey),
        trajectory=slot("trajectory", "trajectory_v1", f"{prefix}-trajectory", uid=uid, hotkey=hotkey),
    )


def lease(**over):
    ident = identity()
    fields = dict(
        protocol_version=3,
        challenge_id="chal-1",
        task_id=compute_task_id(ident),
        identity=ident,
        workspace=archive("workspace", HEX_A),
        workspace_url="https://uploads.invalid/workspace",
        verifier=archive("verifier", HEX_B),
        issued_at=100,
        expires_at=200,
        slot_pool=[slot_set()],
        commit_min_signed_responses=1,
    )
    fields.update(over)
    return fields


def headers():
    return {key: "value" for key in EPISTULA_HEADERS}


def submission(**over):
    fields = dict(
        uid=7,
        hotkey="hk-7",
        request_id="req-7",
        response_body='{"protocol_version":3}',
        response_headers=headers(),
        error="",
        latency_ms=12,
    )
    fields.update(over)
    return fields


def test_lease_enforces_identity_artifacts_quorum_and_unique_upload_ids():
    assert LeaseResponse(**lease()).workspace.artifact_role == "workspace"
    with pytest.raises(ValidationError):
        LeaseResponse(**lease(task_id="c" * 64))
    with pytest.raises(ValidationError):
        LeaseResponse(**lease(workspace=archive("terminal_environment", HEX_A)))
    with pytest.raises(ValidationError):
        LeaseResponse(**lease(commit_min_signed_responses=2))
    duplicate = slot_set(uid=8, hotkey="hk-8", prefix="other")
    duplicate = duplicate.model_copy(
        update={"submission": duplicate.submission.model_copy(update={"upload_id": "u-submission"})}
    )
    with pytest.raises(ValidationError):
        LeaseResponse(**lease(slot_pool=[slot_set(), duplicate]))


@pytest.mark.parametrize("duplicate", ["uid", "hotkey"])
def test_lease_rejects_duplicate_uid_or_hotkey(duplicate):
    values = {"uid": 8, "hotkey": "hk-8"}
    values[duplicate] = 7 if duplicate == "uid" else "hk-7"
    with pytest.raises(ValidationError):
        LeaseResponse(**lease(slot_pool=[slot_set(), slot_set(prefix="x", **values)]))


def test_miner_task_request_binds_assigned_slots():
    current = lease()
    fields = {key: current[key] for key in (
        "protocol_version", "challenge_id", "task_id", "identity", "workspace", "workspace_url", "expires_at"
    )}
    assert MinerTaskRequest(**fields, slots=slot_set())
    foreign = MinerSlotSet(
        submission=slot("patch", "unified_diff_v1", "f-s", challenge_id="foreign"),
        trajectory=slot("trajectory", "trajectory_v1", "f-t", challenge_id="foreign"),
    )
    with pytest.raises(ValidationError):
        MinerTaskRequest(**fields, slots=foreign)


def test_miner_response_binds_type_formats_and_distinct_uploads():
    slots = slot_set()
    response = MinerTaskResponse(
        protocol_version=3,
        challenge_id="chal-1",
        task_id=compute_task_id(identity()),
        response_type="repository_patch_v1",
        submission={
            "artifact_role": "patch",
            "artifact_format": "unified_diff_v1",
            "upload_id": slots.submission.upload_id,
            "sha256": HEX_A,
            "size_bytes": 0,
        },
        trajectory={
            "artifact_role": "trajectory",
            "artifact_format": "trajectory_v1",
            "upload_id": slots.trajectory.upload_id,
            "sha256": HEX_B,
            "size_bytes": 1,
        },
    )
    assert response.protocol_version == 3
    with pytest.raises(ValidationError):
        MinerTaskResponse(**response.model_dump(exclude={"response_type"}), response_type="terminal_script_v1")
    payload = response.model_dump()
    payload["trajectory"]["upload_id"] = payload["submission"]["upload_id"]
    with pytest.raises(ValidationError):
        MinerTaskResponse(**payload)


@pytest.mark.parametrize(
    "field,limit",
    [("response_body", 16_384), ("error", 4_096)],
)
def test_opaque_text_limits_count_utf8_bytes(field, limit):
    if field == "response_body":
        base = submission(response_body="é" * (limit // 2), error="")
    else:
        base = submission(response_body="", response_headers={}, error="é" * (limit // 2))
    assert MinerSubmission(**base)
    base[field] += "a"
    with pytest.raises(ValidationError):
        MinerSubmission(**base)


def test_opaque_fields_are_not_nfc_normalized():
    body = "e\u0301"
    result = MinerSubmission(**submission(response_body=body))
    assert result.response_body == body
    request = ChallengeCommitRequest(protocol_version=3, challenge_id="chal-1", submissions=[result])
    encoded = serialize_commit_request(request)
    assert body.encode("utf-8") in encoded


def test_signed_and_failed_submission_shapes_are_exclusive():
    assert MinerSubmission(**submission())
    assert MinerSubmission(**submission(response_body="", response_headers={}, error="timeout"))
    for bad in (
        submission(response_headers={}),
        submission(error="also failed"),
        submission(response_body="", response_headers={}, error=""),
    ):
        with pytest.raises(ValidationError):
            MinerSubmission(**bad)


def test_headers_are_exact_and_values_use_utf8_byte_limit():
    missing = headers()
    missing.pop("Epistula-Signed-For")
    with pytest.raises(ValidationError):
        MinerSubmission(**submission(response_headers=missing))
    extra = headers() | {"Other": "x"}
    with pytest.raises(ValidationError):
        MinerSubmission(**submission(response_headers=extra))
    oversized = headers()
    oversized["Epistula-Version"] = "é" * 257
    with pytest.raises(ValidationError):
        MinerSubmission(**submission(response_headers=oversized))


@pytest.mark.parametrize("key", ["uid", "hotkey"])
def test_commit_rejects_duplicate_miner_identity(key):
    second = submission(uid=8, hotkey="hk-8", request_id="req-8")
    second[key] = submission()[key]
    with pytest.raises(ValidationError):
        ChallengeCommitRequest(protocol_version=3, challenge_id="chal-1", submissions=[submission(), second])


def grant(**over):
    fields = dict(
        uid=7,
        hotkey="hk-7",
        upload_id="upload-7",
        sha256=HEX_A,
        size_bytes=1,
        format="unified_diff_v1",
        read_url="https://uploads.invalid/read",
    )
    fields.update(over)
    return fields


def reveal(**over):
    fields = dict(
        protocol_version=3,
        challenge_id="chal-1",
        task_id=compute_task_id(identity()),
        verifier=archive("verifier", HEX_B),
        verifier_policy="command-gold-digest-v1",
        verifier_url="https://uploads.invalid/verifier",
        grading_expires_at=300,
        submission_grants=[grant()],
        artifact_failures=[],
    )
    fields.update(over)
    return fields


def test_commit_reveal_rejects_duplicates_across_lists():
    assert CommitRevealResponse(**reveal()).protocol_version == 3
    failure = {"uid": 7, "hotkey": "other", "reason": "artifact_invalid"}
    with pytest.raises(ValidationError):
        CommitRevealResponse(**reveal(artifact_failures=[failure]))
    failure = {"uid": 8, "hotkey": "hk-7", "reason": "artifact_invalid"}
    with pytest.raises(ValidationError):
        CommitRevealResponse(**reveal(artifact_failures=[failure]))


def test_commit_reveal_requires_verifier_role_and_exact_policy():
    with pytest.raises(ValidationError):
        CommitRevealResponse(**reveal(verifier=archive("workspace", HEX_B)))
    with pytest.raises(ValidationError):
        CommitRevealResponse(**reveal(verifier_policy="binary-pass-v1"))


def test_commit_reveal_results_match_every_submitted_miner_once():
    request = ChallengeCommitRequest(protocol_version=3, challenge_id="chal-1", submissions=[submission()])
    response = CommitRevealResponse(**reveal())
    validate_commit_reveal(LeaseResponse(**lease()), request, response)
    missing = CommitRevealResponse(**reveal(submission_grants=[]))
    with pytest.raises(ValueError):
        validate_commit_reveal(LeaseResponse(**lease()), request, missing)
    foreign = CommitRevealResponse(**reveal(submission_grants=[grant(uid=8, hotkey="hk-8")]))
    with pytest.raises(ValueError):
        validate_commit_reveal(LeaseResponse(**lease()), request, foreign)


def test_feedback_is_strict_bounded_unique_and_serialized_once():
    verdict = FeedbackVerdict(
        uid=7,
        hotkey="hk-7",
        passed=False,
        grading_duration_ms=123,
    )
    request = ChallengeFeedbackRequest(
        protocol_version=3,
        challenge_id="chal-1",
        task_id=compute_task_id(identity()),
        verdicts=[verdict],
    )
    assert serialize_feedback_request(request) == request.model_dump_json().encode()
    with pytest.raises(ValidationError):
        FeedbackVerdict(uid=7, hotkey="hk-7", passed=1, grading_duration_ms=0)
    with pytest.raises(ValidationError):
        ChallengeFeedbackRequest(
            protocol_version=3,
            challenge_id="chal-1",
            task_id=compute_task_id(identity()),
            verdicts=[verdict, verdict.model_copy(update={"hotkey": "other"})],
        )


def test_incoming_v3_models_require_explicit_protocol_version():
    for model, payload in (
        (LeaseResponse, lease()),
        (CommitRevealResponse, reveal()),
        (
            ChallengeFeedbackResponse,
            {"protocol_version": 3, "challenge_id": "chal-1", "task_id": "a" * 64},
        ),
    ):
        payload = dict(payload)
        payload.pop("protocol_version")
        with pytest.raises(ValidationError):
            model(**payload)
