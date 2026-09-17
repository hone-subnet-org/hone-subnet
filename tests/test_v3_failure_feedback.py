from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from rlvr.config import Settings
from rlvr.policy import RELEASE_POLICY
from rlvr.v3 import round as round_module
from rlvr.v3.api import (
    ChallengeFeedbackRequest,
    ChallengeFeedbackResponse,
    FailureExplanation,
    FeedbackVerdict,
    serialize_feedback_request,
)
from rlvr.v3.artifacts import ArtifactGrant
from rlvr.v3.client import V3ProblemServerClient
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.release import round_policy
from rlvr.v3.round import MinerEvaluation, RoundResult, compute_round_payments
from tests.test_v3_round import policy

DISPLAY = 'Command (argv): ["/usr/bin/python3","main.py"]\nRequired stdout: "pass\\n"'
PRIVATE = "private candidate output must never reach feedback"


def verdict(**changes):
    return FeedbackVerdict(
        **(
            {"uid": 7, "hotkey": "hk-7", "passed": False, "grading_duration_ms": 23}
            | changes
        )
    )


def request(item):
    return ChallengeFeedbackRequest(
        protocol_version=3, challenge_id="chal-1", task_id="a" * 64, verdicts=[item]
    )


def evaluation(
    uid=7,
    hotkey="hk-7",
    *,
    status="failed",
    code=MinerReason.CHECK_FAILED,
    stage=Stage.CHECK,
    display=DISPLAY,
):
    result = EvaluationResult(
        status, "" if status == "passed" else PRIVATE, (), None, code, stage
    )
    if display is not None:
        object.__setattr__(result, "failed_check", display)
    return MinerEvaluation(uid, hotkey, 1, result, 23)


def grant(uid=7, hotkey="hk-7"):
    return ArtifactGrant(
        uid=uid,
        hotkey=hotkey,
        upload_id=f"upload-{uid}",
        sha256="b" * 64,
        size_bytes=1,
        format="unified_diff_v1",
        read_url="https://uploads.invalid/read",
    )


class RecordingClient:
    def __init__(self, *, fails=False):
        self.bodies = []
        self.fails = fails

    async def feedback(self, payload):
        self.bodies.append(serialize_feedback_request(payload))
        if self.fails:
            raise httpx.ConnectError("offline")
        return True


def send(client, evaluations, *, grants=None, **kwargs):
    return asyncio.run(
        round_module._send_diagnostic_feedback(
            client,
            "chal-1",
            "a" * 64,
            [grant()] if grants is None else grants,
            evaluations,
            **kwargs,
        )
    )


@pytest.mark.parametrize("explicit_null", [False, True])
def test_absent_failure_keeps_legacy_bytes_in_all_serialization_paths(explicit_null):
    item = verdict(**({"failure": None} if explicit_null else {}))
    expected = {"uid": 7, "hotkey": "hk-7", "passed": False, "grading_duration_ms": 23}
    assert item.model_dump() == expected
    assert json.loads(item.model_dump_json()) == expected
    old_wire = (
        '{"protocol_version":3,"challenge_id":"chal-1","task_id":"'
        + "a" * 64
        + '","verdicts":[{"uid":7,"hotkey":"hk-7","passed":false,"grading_duration_ms":23}]}'
    ).encode()
    assert request(item).model_dump_json().encode() == old_wire
    assert serialize_feedback_request(request(item)) == old_wire


@pytest.mark.parametrize("reason", [*MinerReason, "evaluation_failed"])
def test_public_failure_reasons_round_trip_with_explicit_null_test(reason):
    failure = FailureExplanation(version=1, reason_code=reason)
    encoded = serialize_feedback_request(request(verdict(failure=failure)))
    payload = json.loads(encoded)["verdicts"][0]["failure"]
    assert payload == {"version": 1, "reason_code": reason, "failed_check": None}
    assert (
        ChallengeFeedbackRequest.model_validate_json(encoded).verdicts[0].failure
        == failure
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 1.0},
        {"version": "1"},
        {"version": 2},
        {"reason_code": PRIVATE},
        {"reason_code": RoundReason.VALIDATOR_ERROR},
        {"actual_stdout": PRIVATE},
        {"receipt": {}},
        {"check_id": "secret"},
        {"failed_check": b"bytes"},
        {"failed_check": 1},
        {"failed_check": ""},
        {"reason_code": MinerReason.TIMEOUT, "failed_check": DISPLAY},
    ],
)
def test_failure_schema_rejects_noncontract_data(changes):
    fields = {"version": 1, "reason_code": MinerReason.CHECK_FAILED} | changes
    with pytest.raises(ValidationError):
        FailureExplanation(**fields)


def test_passing_verdict_cannot_carry_failure():
    with pytest.raises(ValidationError):
        verdict(
            passed=True,
            failure=FailureExplanation(version=1, reason_code=MinerReason.CHECK_FAILED),
        )


@pytest.mark.parametrize("unit", ["x", "\n", "é", "😀"])
def test_failed_check_cap_counts_json_escaping_and_quotes(unit):
    per_unit = len(json.dumps(unit, ensure_ascii=True).encode("ascii")) - 2
    fitting = unit * (2046 // per_unit)
    FailureExplanation(
        version=1, reason_code=MinerReason.CHECK_FAILED, failed_check=fitting
    )
    with pytest.raises(ValidationError):
        FailureExplanation(
            version=1, reason_code=MinerReason.CHECK_FAILED, failed_check=fitting + unit
        )


@pytest.mark.parametrize(
    "update",
    [
        {"version": True},
        {"reason_code": PRIVATE},
        {"failed_check": "x" * 2047},
        {"reason_code": MinerReason.TIMEOUT, "failed_check": DISPLAY},
    ],
)
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_serializer_revalidates_nested_failure_after_unchecked_copy(update):
    valid = FailureExplanation(version=1, reason_code=MinerReason.CHECK_FAILED)
    forged = valid.model_copy(update=update)
    item = verdict().model_copy(update={"failure": forged})
    payload = request(verdict()).model_copy(update={"verdicts": [item]})
    with pytest.raises((ValidationError, ValueError)):
        serialize_feedback_request(payload)


def test_serializer_rejects_unchecked_failure_added_to_passing_verdict():
    payload = request(verdict(passed=True))
    item = payload.verdicts[0].model_copy(
        update={
            "failure": FailureExplanation(
                version=1, reason_code=MinerReason.CHECK_FAILED
            ),
        }
    )
    with pytest.raises((ValidationError, ValueError)):
        serialize_feedback_request(payload.model_copy(update={"verdicts": [item]}))


def test_server_feedback_keeps_legacy_payload_despite_available_display():
    client = RecordingClient()
    assert send(client, [evaluation()])
    assert client.bodies == [serialize_feedback_request(request(verdict()))]


def test_server_feedback_binds_exact_registration_and_preserves_grant_order():
    client = RecordingClient()
    evaluations = [
        evaluation(8, "hk-8", display="test for eight"),
        evaluation(7, "hk-7", display="test for seven"),
        evaluation(7, "retired-hotkey", status="passed", display=None),
        evaluation(9, "hk-9", status="passed", display=None),
    ]
    assert send(client, evaluations, grants=[grant(9, "hk-9"), grant(), grant(8, "hk-8")])
    items = json.loads(client.bodies[0])["verdicts"]
    assert [item["uid"] for item in items] == [9, 7, 8]
    assert [item["passed"] for item in items] == [True, False, False]
    assert all("failure" not in item for item in items)
    assert PRIVATE.encode() not in client.bodies[0]
    assert b"test for" not in client.bodies[0]


def test_feedback_failure_cannot_change_result_or_payments():
    evaluations = [evaluation(), evaluation(8, "hk-8", status="passed", display=None)]
    result = RoundResult("completed", "", tuple(evaluations))
    before = repr(result)
    client = RecordingClient(fails=True)
    assert not send(
        client,
        evaluations,
        grants=[grant(), grant(8, "hk-8")],
    )
    assert repr(result) == before
    assert compute_round_payments(
        result, speed_half_life_ms=180_000, speed_floor=0.95
    ) == {7: 0.0, 8: 1.0}


def test_enriched_feedback_retries_identical_serialized_bytes():
    payload = request(
        verdict(
            failure=FailureExplanation(
                version=1,
                reason_code=MinerReason.CHECK_FAILED,
                failed_check=DISPLAY,
            )
        )
    )
    bodies = []

    async def handler(http_request):
        bodies.append(await http_request.aread())
        if len(bodies) == 1:
            return httpx.Response(503)
        response = ChallengeFeedbackResponse(
            protocol_version=3, challenge_id="chal-1", task_id="a" * 64
        )
        return httpx.Response(200, content=response.model_dump_json())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = V3ProblemServerClient(
                "https://problems.invalid", "validator", http, retries=2
            )
            return await client.feedback(payload)

    assert asyncio.run(run())
    assert bodies == [serialize_feedback_request(payload)] * 2


def test_direct_notices_and_details_default_on_with_no_server_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("VALIDATOR_FAILURE_NOTICES", raising=False)
    monkeypatch.delenv("VALIDATOR_FAILED_CHECK_DETAILS", raising=False)
    monkeypatch.setenv("VALIDATOR_FAILURE_EXPLANATIONS", "true")
    settings = Settings(_env_file=None)
    assert settings.validator_failure_notices is True
    assert settings.validator_failed_check_details is True
    assert not hasattr(settings, "validator_failure_explanations")
    assert not hasattr(policy(tmp_path), "failure_explanations")
    monkeypatch.setattr("rlvr.v3.release.os.getuid", lambda: 1000)
    monkeypatch.setattr("rlvr.v3.release.os.getgid", lambda: 1000)
    monkeypatch.setattr("rlvr.v3.release.shutil.which", lambda _: "/usr/bin/docker")
    released = round_policy(RELEASE_POLICY, dispatch_concurrency=4)
    assert not hasattr(released, "failure_explanations")
    with pytest.raises(TypeError):
        round_policy(RELEASE_POLICY, dispatch_concurrency=4, failure_explanations=True)
    monkeypatch.setenv("VALIDATOR_FAILURE_NOTICES", "false")
    monkeypatch.setenv("VALIDATOR_FAILED_CHECK_DETAILS", "false")
    disabled = Settings(_env_file=None)
    assert disabled.validator_failure_notices is False
    assert disabled.validator_failed_check_details is False


@pytest.mark.parametrize(
    "fault", [None, "cleanup", "grade_infrastructure", "grade_exception"]
)
def test_full_round_sends_only_legacy_feedback_after_successful_cleanup(
    tmp_path, monkeypatch, fault
):
    from tests import test_v3_round as synthetic

    original_send = round_module._send_diagnostic_feedback
    sent = []

    async def capture(client, challenge_id, task_id, grants, evaluations, **kwargs):
        assert kwargs == {}
        recorder = RecordingClient()
        accepted = await original_send(
            recorder, challenge_id, task_id, grants, evaluations, **kwargs
        )
        assert accepted
        sent.extend(recorder.bodies)
        return await original_send(
            client, challenge_id, task_id, grants, evaluations, **kwargs
        )

    monkeypatch.setattr(round_module, "_send_diagnostic_feedback", capture)
    # The shared exercise checks completed/abandoned outcomes and actual score updates.
    synthetic.test_complete_synthetic_round_has_pass_fail_malformed_and_no_response(
        tmp_path,
        monkeypatch,
        False,
        False,
        fault,
    )
    if fault is not None:
        assert sent == []
        return
    assert len(sent) == 1
    items = json.loads(sent[0])["verdicts"]
    assert all("failure" not in item for item in items)


def test_round_directory_cleanup_failure_prevents_feedback_and_score_updates(
    tmp_path, monkeypatch
):
    from rlvr.scoring.eval_engine import EvalEngine
    from tests import test_v3_round as synthetic

    original_directory = round_module.tempfile.TemporaryDirectory
    original_evaluate = synthetic.evaluate_round
    original_send = round_module._send_diagnostic_feedback
    captured = {}
    feedback_calls = []

    class FailingExit(original_directory):
        def __exit__(self, *args):
            super().__exit__(*args)
            raise OSError("final temporary directory cleanup failed")

    async def capture_evaluate(*args, **kwargs):
        captured["result"] = await original_evaluate(*args, **kwargs)
        # Stop the shared fixture before its ordinary completed-round assertions.
        raise ObservedRound

    async def capture_send(*args, **kwargs):
        feedback_calls.append(kwargs)
        return await original_send(*args, **kwargs)

    class ObservedRound(Exception):
        pass

    monkeypatch.setattr(round_module.tempfile, "TemporaryDirectory", FailingExit)
    monkeypatch.setattr(round_module, "_send_diagnostic_feedback", capture_send)
    monkeypatch.setattr(synthetic, "evaluate_round", capture_evaluate)
    with pytest.raises(ObservedRound):
        synthetic.test_complete_synthetic_round_has_pass_fail_malformed_and_no_response(
            tmp_path,
            monkeypatch,
            False,
            False,
            None,
        )
    outcome = captured["result"]
    assert outcome.status == "abandoned"
    assert feedback_calls == []
    assert outcome.reason_code == RoundReason.CLEANUP_FAILED
    assert outcome.stage == Stage.CLEANUP
    assert outcome.evaluations == ()
    assert (
        compute_round_payments(outcome, speed_half_life_ms=180_000, speed_floor=0.95)
        == {}
    )
    engine = EvalEngine(6, 1, 200, 4)
    engine.update({1: 1.0}, hotkeys={1: "hk-1"}, dispatched={1})
    before = repr(engine.histories)
    assert not round_module.apply_round_scores(
        outcome,
        engine,
        active_hotkeys={uid: f"hk-{uid}" for uid in (1, 2, 3, 4)},
        speed_half_life_ms=180_000,
        speed_floor=0.95,
    )
    assert repr(engine.histories) == before


def test_full_request_budget_includes_escaped_identity_fields():
    failure = FailureExplanation(
        version=1,
        reason_code=MinerReason.CHECK_FAILED,
        failed_check="x" * 2046,
    )
    payload = ChallengeFeedbackRequest(
        protocol_version=3,
        challenge_id="\x00" * 128,
        task_id="a" * 64,
        verdicts=[
            verdict(
                uid=uid,
                hotkey="\x00" * 124 + f"{uid:04}",
                grading_duration_ms=2**53 - 1,
                failure=failure,
            )
            for uid in range(1024)
        ],
    )
    # These identifiers are legal in the existing schema and cost six bytes per
    # control character on the wire. Ordinary printable IDs are not the upper bound.
    encoded = serialize_feedback_request(payload)
    assert len(encoded) > 3_000_000
    assert len(ChallengeFeedbackRequest.model_validate_json(encoded).verdicts) == 1024
