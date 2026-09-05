import base64
import hashlib

import pytest
import rfc8785

from rlvr.v3.trajectory import ModelFailure, Trajectory, TrajectoryError, parse_trajectory, serialize_trajectory


EMPTY_HASH = hashlib.sha256(b"").hexdigest()


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def fixture() -> dict:
    alternative = {"token_bytes_b64": b64(b"x"), "token_id": None, "logprob": 0}
    return {
        "schema_version": 1,
        "task_id": "0" * 64,
        "challenge_id": "c",
        "miner_hotkey": "h",
        "submission_sha256": EMPTY_HASH,
        "harness_name": "h",
        "harness_version": "1",
        "model_provider": "p",
        "model_name": "m",
        "events": [
            {
                "sequence": 0,
                "event_type": "model_turn",
                "request_body_b64": b64(b"{}"),
                "response_body_b64": b64(b"{}"),
                "generated_bytes_b64": b64(b"x"),
                "reasoning": "x",
                "output": "x",
                "tokens": [{"token_bytes_b64": b64(b"x"), "token_id": None, "logprob": 0, "top_logprobs": [alternative] * 5}],
            },
            {"sequence": 1, "event_type": "tool_call", "call_id": "1", "tool_name": "shell", "input_body_b64": b64(b"true")},
            {"sequence": 2, "event_type": "tool_result", "call_id": "1", "output_body_b64": "", "is_error": False},
            {"sequence": 3, "event_type": "final_submission", "submission_sha256": EMPTY_HASH},
        ],
    }


def test_contract_fixture_round_trips_as_canonical_jcs():
    trajectory = Trajectory.model_validate(fixture())
    raw = serialize_trajectory(trajectory)
    assert not raw.endswith(b"\n")
    assert parse_trajectory(raw) == trajectory


@pytest.mark.parametrize("mutation", ["sequence", "hash", "base64", "unknown", "tool"])
def test_rejects_invalid_trajectory(mutation):
    value = fixture()
    if mutation == "sequence":
        value["events"][1]["sequence"] = 8
    elif mutation == "hash":
        value["events"][-1]["submission_sha256"] = "1" * 64
    elif mutation == "base64":
        value["events"][0]["generated_bytes_b64"] = "eA"
    elif mutation == "unknown":
        value["extra"] = True
    else:
        value["events"][2]["call_id"] = "other"
    with pytest.raises(ValueError):
        Trajectory.model_validate(value)


def test_parser_rejects_noncanonical_json_and_duplicate_keys():
    raw = serialize_trajectory(Trajectory.model_validate(fixture()))
    with pytest.raises(TrajectoryError):
        parse_trajectory(raw + b"\n")
    with pytest.raises(TrajectoryError):
        parse_trajectory(b'{"schema_version":1,"schema_version":1}')


def test_failure_response_fields_match_the_failure_kind():
    ModelFailure(
        sequence=0,
        event_type="model_failure",
        request_body_b64="",
        response_body_b64=None,
        failure_kind="transport_error",
        status_code=None,
    )
    with pytest.raises(ValueError):
        ModelFailure(
            sequence=0,
            event_type="model_failure",
            request_body_b64="",
            response_body_b64=None,
            failure_kind="http_error",
            status_code=500,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.0, b'{"n":0}'),
        (1e21, b'{"n":1e+21}'),
        (5e-324, b'{"n":5e-324}'),
        (1.2345678901234567, b'{"n":1.2345678901234567}'),
    ],
)
def test_pinned_jcs_float_vectors(value, expected):
    assert rfc8785.dumps({"n": value}) == expected


def test_schema_version_rejects_boolean_one():
    value = fixture()
    value["schema_version"] = True
    with pytest.raises(ValueError):
        Trajectory.model_validate(value)
