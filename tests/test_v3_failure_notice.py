from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rlvr.v3.api import (
    FAILURE_NOTICE_MAX_BYTES,
    FailureExplanation,
    MinerFailureNotice,
    serialize_failure_notice,
)
from rlvr.v3.reasons import MinerReason


def notice(**changes):
    return MinerFailureNotice(
        **({
            "protocol_version": 3,
            "message_type": "failure_notice_v1",
            "challenge_id": "challenge",
            "task_id": "a" * 64,
            "uid": 7,
            "hotkey": "miner",
            "failure": FailureExplanation(
                version=1,
                reason_code=MinerReason.CHECK_FAILED,
                failed_check='Required stdout: "red-fox\\n"',
            ),
        } | changes)
    )


def test_notice_round_trips_with_exact_self_describing_shape():
    original = notice()
    body = serialize_failure_notice(original)
    assert MinerFailureNotice.model_validate_json(body) == original
    assert json.loads(body) == {
        "protocol_version": 3,
        "message_type": "failure_notice_v1",
        "challenge_id": "challenge",
        "task_id": "a" * 64,
        "uid": 7,
        "hotkey": "miner",
        "failure": {
            "version": 1,
            "reason_code": "check_failed",
            "failed_check": 'Required stdout: "red-fox\\n"',
        },
    }


@pytest.mark.parametrize("reason", [*MinerReason, "evaluation_failed"])
def test_notice_supports_reason_only(reason):
    original = notice(failure=FailureExplanation(version=1, reason_code=reason))
    assert MinerFailureNotice.model_validate_json(serialize_failure_notice(original)) == original
    assert json.loads(serialize_failure_notice(original))["failure"]["failed_check"] is None


@pytest.mark.parametrize("changes", [
    {"protocol_version": True},
    {"protocol_version": 3.0},
    {"protocol_version": "3"},
    {"protocol_version": 4},
    {"message_type": "solve"},
    {"uid": True},
    {"uid": -1},
    {"uid": "7"},
    {"hotkey": ""},
    {"challenge_id": ""},
    {"task_id": "not-a-digest"},
    {"failure": None},
    {"passed": False},
    {"actual_stdout": "private"},
    {"receipt": {}},
    {"check_id": "private"},
])
def test_notice_rejects_noncontract_data(changes):
    with pytest.raises(ValidationError):
        notice(**changes)


@pytest.mark.parametrize("changes", [
    {"version": True},
    {"reason_code": "arbitrary private reason"},
    {"failed_check": "x" * 2047},
    {"reason_code": MinerReason.TIMEOUT},
])
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_notice_serializer_revalidates_unchecked_nested_copies(changes):
    original = notice()
    forged = original.model_copy(update={
        "failure": original.failure.model_copy(update=changes),
    })
    with pytest.raises(ValidationError):
        serialize_failure_notice(forged)


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_notice_serializer_revalidates_outer_copy_and_requires_model():
    with pytest.raises(ValidationError):
        serialize_failure_notice(notice().model_copy(update={"uid": True}))
    with pytest.raises(TypeError):
        serialize_failure_notice(notice().model_dump())


def test_maximum_escaped_identifiers_and_display_fit_request_limit():
    original = notice(
        challenge_id="\x00" * 128,
        hotkey="\x00" * 128,
        failure=FailureExplanation(
            version=1, reason_code=MinerReason.CHECK_FAILED, failed_check="x" * 2046,
        ),
    )
    body = serialize_failure_notice(original)
    assert len(body) < FAILURE_NOTICE_MAX_BYTES == 8192
    assert MinerFailureNotice.model_validate_json(body) == original
