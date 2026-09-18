"""Trajectory spot checks: a sampled, signed reference sent best effort."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from rlvr.neurons import trace_check
from rlvr.neurons.trace_check import send_trace_checks
from rlvr.v3.api import TraceCheckRequest, serialize_trace_check_request
from rlvr.v3.artifacts import MinerArtifactRef
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.reasons import MinerReason, Stage
from rlvr.v3.round import MinerEvaluation, RoundResult, compute_round_payments

URL = "https://traces.invalid/check"
SERVICE = "service-hotkey"
WALLET = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator-hotkey"))


def ref(uid=1, role="trajectory"):
    return MinerArtifactRef(
        artifact_role=role,
        artifact_format="trajectory_v1" if role == "trajectory" else "unified_diff_v1",
        upload_id=f"u{uid}-{role}",
        sha256="b" * 64,
        size_bytes=1234,
    )


def evaluation(uid, status="failed", *, trajectory=True):
    if status == "passed":
        result = EvaluationResult("passed", "", (), None, None, Stage.CHECK)
    elif status == "failed":
        result = EvaluationResult("failed", "wrong", (), None, MinerReason.CHECK_FAILED, Stage.CHECK)
    else:
        result = EvaluationResult("rejected", "bad patch", (), None, MinerReason.PATCH_STATIC_REJECTED, Stage.PATCH)
    return MinerEvaluation(uid, f"hk-{uid}", 10, result, 5, trajectory=ref(uid) if trajectory else None)


def completed(evaluations, assigned=None, status="completed"):
    return RoundResult(
        status, "", tuple(evaluations) if status == "completed" else (),
        challenge_id="chal-1", task_id="a" * 64,
        assigned_miners=assigned if assigned is not None else tuple((e.uid, e.hotkey) for e in evaluations),
        diagnostic_evaluations=tuple(evaluations), checks_total=1,
    )


class Service:
    def __init__(self, status=200, fail=None):
        self.status, self.fail, self.requests, self.raw = status, fail, [], []

    async def __call__(self, request):
        raw = await request.aread()
        self.raw.append(raw)
        self.requests.append((str(request.url), dict(request.headers), json.loads(raw)))
        if self.fail:
            raise self.fail
        return httpx.Response(self.status, json={"accepted": True})


def send(service, result, *, url=URL, hotkey=SERVICE, rate=1.0):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(service)) as http:
            return await send_trace_checks(
                result, wallet=WALLET, http=http, url=url, service_hotkey=hotkey, rate=rate,
            )
    return asyncio.run(run())


def test_ticket_names_the_trajectory_and_is_signed_for_the_service():
    service = Service()
    assert send(service, completed([evaluation(1, "failed")])) == 1
    url, headers, body = service.requests[0]
    assert url == URL
    assert headers["epistula-signed-for"] == SERVICE
    assert headers["epistula-signed-by"] == "validator-hotkey"
    assert body == {
        "protocol_version": 3,
        "message_type": "trace_check_v1",
        "challenge_id": "chal-1",
        "task_id": "a" * 64,
        "uid": 1,
        "hotkey": "hk-1",
        "trajectory": ref(1).model_dump(),
        "submission_status": "failed",
    }
    assert "wrong" not in json.dumps(body)


@pytest.mark.parametrize("status", ["passed", "failed", "rejected"])
def test_every_granted_outcome_is_eligible(status):
    service = Service()
    assert send(service, completed([evaluation(1, status)])) == 1
    assert service.requests[0][2]["submission_status"] == status


def test_commit_rejection_without_a_trusted_reference_is_never_sent():
    service = Service()
    granted_rejection = evaluation(1, "rejected")
    commit_rejection = evaluation(2, "rejected", trajectory=False)
    assert send(service, completed([granted_rejection, commit_rejection])) == 1
    assert [body["uid"] for _, _, body in service.requests] == [1]


def test_eligibility_is_the_trusted_reference_alone():
    service = Service()
    assert send(service, completed([evaluation(1)], assigned=())) == 1


def test_only_completed_rounds_are_sampled():
    service = Service()
    assert send(service, completed([evaluation(1)], status="abandoned")) == 0
    assert service.requests == []


def test_batch_deadline_cancels_active_requests(monkeypatch):
    # The per-request timeout is LONGER than the batch deadline, so only the
    # batch deadline can end this; removing it would hang for 5 seconds.
    monkeypatch.setattr(trace_check, "TRACE_CHECK_TIMEOUT_S", 5.0)
    monkeypatch.setattr(trace_check, "TRACE_CHECK_BATCH_DEADLINE_S", 0.2)
    in_flight, peak = [0], [0]

    async def stalled(request):
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        try:
            await asyncio.sleep(10)
        finally:
            in_flight[0] -= 1
        return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as http:
            loop = asyncio.get_running_loop()
            began = loop.time()
            count = await send_trace_checks(
                completed([evaluation(uid) for uid in range(1, 11)]),
                wallet=WALLET, http=http, url=URL, service_hotkey=SERVICE, rate=1.0,
            )
            # Inspect BEFORE the loop shuts down, which would cancel leftovers itself.
            return count, loop.time() - began, in_flight[0], peak[0]

    count, elapsed, leftover, peak_in_flight = asyncio.run(run())
    assert count == 0
    assert elapsed < 1.0
    assert leftover == 0  # active requests were cancelled by the batch deadline
    assert peak_in_flight <= trace_check.TRACE_CHECK_CONCURRENCY


@pytest.mark.parametrize("kwargs", [
    {"url": ""},
    {"url": "http://traces.invalid/check"},
    {"hotkey": ""},
    {"rate": 0.0},
])
def test_disabled_or_insecure_configuration_sends_nothing(kwargs):
    service = Service()
    assert send(service, completed([evaluation(1)]), **kwargs) == 0
    assert service.requests == []


def test_rate_is_applied_per_submission(monkeypatch):
    draws = iter([5, 999_999, 3, 500_000])  # below 1% only for the first and third
    monkeypatch.setattr(trace_check.secrets, "randbelow", lambda _n: next(draws))
    service = Service()
    evals = [evaluation(uid) for uid in (1, 2, 3, 4)]
    assert send(service, completed(evals), rate=0.01) == 2
    assert [body["uid"] for _, _, body in service.requests] == [1, 3]


@pytest.mark.parametrize("problem", [
    httpx.ConnectError("offline"),
    httpx.ReadTimeout("slow"),
])
def test_service_failures_are_harmless_and_change_nothing(problem):
    service = Service(fail=problem)
    result = completed([evaluation(1, "passed"), evaluation(2, "failed")])
    before = repr(result)
    assert send(service, result) == 0
    assert repr(result) == before
    assert compute_round_payments(result, speed_half_life_ms=180_000, speed_floor=0.95) == {1: 1.0, 2: 0.0}


def test_non_200_replies_are_not_counted_and_a_batch_warning_is_printed(capsys):
    assert send(Service(status=503), completed([evaluation(1), evaluation(2)])) == 0
    out = capsys.readouterr().out
    assert out.count("WARN") == 1 and "0 of 2 tickets acknowledged" in out


def test_nothing_sampled_prints_nothing(capsys):
    assert send(Service(), completed([evaluation(1)]), rate=0.0) == 0
    assert capsys.readouterr().out == ""


def test_single_stalled_request_is_cut_by_its_own_deadline(monkeypatch):
    monkeypatch.setattr(trace_check, "TRACE_CHECK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(trace_check, "TRACE_CHECK_BATCH_DEADLINE_S", 5.0)

    async def stalled(request):
        await asyncio.sleep(10)
        return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as http:
            loop = asyncio.get_running_loop()
            began = loop.time()
            count = await send_trace_checks(
                completed([evaluation(1)]), wallet=WALLET, http=http,
                url=URL, service_hotkey=SERVICE, rate=1.0,
            )
            return count, loop.time() - began

    count, elapsed = asyncio.run(run())
    assert count == 0 and elapsed < 1.0  # the per-request deadline, not the 5 s batch


def test_reply_body_is_never_read_and_redirects_are_not_followed():
    class Exploding(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("the reply body must never be read")
            yield b""  # pragma: no cover

    async def acked_with_trap(request):
        return httpx.Response(200, stream=Exploding())

    redirected = []

    async def redirecting(request):
        redirected.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://elsewhere.invalid/"})

    assert send(acked_with_trap, completed([evaluation(1)])) == 1
    assert send(redirecting, completed([evaluation(1)])) == 0
    assert redirected == [URL]  # one request, the redirect target never fetched


def test_request_model_requires_a_trajectory_reference():
    with pytest.raises(ValidationError):
        TraceCheckRequest(
            protocol_version=3, message_type="trace_check_v1", challenge_id="chal-1",
            task_id="a" * 64, uid=1, hotkey="hk-1", trajectory=ref(1, role="patch"),
            submission_status="failed",
        )


def test_serializer_revalidates_and_bounds_the_body():
    request = TraceCheckRequest(
        protocol_version=3, message_type="trace_check_v1", challenge_id="chal-1",
        task_id="a" * 64, uid=1, hotkey="hk-1", trajectory=ref(1), submission_status="failed",
    )
    body = serialize_trace_check_request(request)
    assert TraceCheckRequest.model_validate_json(body) == request
    with pytest.raises(TypeError):
        serialize_trace_check_request(request.model_dump())
    forged = request.model_copy(update={"trajectory": ref(1, role="patch")})
    with pytest.raises((ValidationError, ValueError)):
        serialize_trace_check_request(forged)
    widest = TraceCheckRequest(
        protocol_version=3, message_type="trace_check_v1", challenge_id="c" * 128,
        task_id="a" * 64, uid=1023, hotkey="h" * 128,
        trajectory=MinerArtifactRef(
            artifact_role="trajectory", artifact_format="trajectory_v1",
            upload_id="u" * 128, sha256="b" * 64, size_bytes=2**53 - 1,
        ),
        submission_status="rejected",
    )
    assert len(serialize_trace_check_request(widest)) <= 4096


def test_ticket_signature_verifies_against_the_bytes_sent_and_binds_the_recipient():
    from rlvr.protocol import _Keypair, verify_signature

    validator = _Keypair.create_from_uri("//Alice")
    service = Service()
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(service)) as http:
            return await send_trace_checks(
                completed([evaluation(1)]), wallet=validator, http=http,
                url=URL, service_hotkey=SERVICE, rate=1.0,
            )
    assert asyncio.run(run()) == 1
    _, headers, _ = service.requests[0]
    raw = service.raw[0]
    epistula = {k: v for k, v in headers.items() if k.lower().startswith("epistula-")}
    epistula = {"-".join(part.capitalize() for part in k.split("-")): v for k, v in epistula.items()}
    assert headers["epistula-signed-by"] == validator.ss58_address
    assert verify_signature(epistula, raw, expected_signed_for=SERVICE)
    assert not verify_signature(epistula, raw, expected_signed_for="someone-else")
    assert not verify_signature(epistula, raw + b" ", expected_signed_for=SERVICE)


@pytest.mark.parametrize("case", ["sends_after_save", "save_failure", "sender_raises"])
async def test_callback_sends_only_after_a_successful_score_save(tmp_path, monkeypatch, case):
    from rlvr.config import Settings
    from rlvr.neurons import decentralized

    if case == "save_failure":
        (tmp_path / "blocker").write_bytes(b"not a directory")
        score_file = tmp_path / "blocker" / "scores.json"
    else:
        score_file = tmp_path / "scores.json"
    settings = Settings(
        _env_file=None, problem_server_url="https://problems.invalid",
        validator_score_state_file=str(score_file),
        validator_trace_check_url=URL, validator_trace_check_hotkey=SERVICE,
    )
    outcome = completed([evaluation(1, "passed")], assigned=((1, "hk-1"),))
    calls, returned = [], []

    async def fake_send(result, **kwargs):
        # Record first, so a wrongly invoked sender is visible even if the
        # assertion below raises; then require scores already on disk.
        calls.append((result, kwargs))
        assert json.loads(score_file.read_text())["histories"]
        if case == "sender_raises":
            raise RuntimeError("service exploded")
        return 1

    class Validator:
        def __init__(self, *_args, **_kwargs):
            self.wallet = WALLET
            self.subtensor = object()
            self.metagraph = SimpleNamespace(hotkeys=["owner", "hk-1"], sync=lambda **_: None)

        def setup_bittensor(self):
            pass

        def set_round_callback(self, callback):
            self.callback = callback

        def set_weight_setter(self, _setter):
            pass

        async def run(self):
            returned.append(await self.callback(self))

    async def no_notices(*_args, **_kwargs):
        return 0

    async def evaluate(*_args, **_kwargs):
        return outcome

    monkeypatch.setattr(decentralized, "ValidatorNeuron", Validator)
    monkeypatch.setattr(decentralized, "_apply_weights_rate_limit", lambda *_: None)
    monkeypatch.setattr(decentralized, "v3_round_policy", lambda *_a, **_k: object())
    monkeypatch.setattr(decentralized, "_solver_clients", lambda *_a, **_k: [SimpleNamespace(uid=1, hotkey="hk-1")])
    monkeypatch.setattr(decentralized, "evaluate_round", evaluate)
    monkeypatch.setattr(decentralized, "send_failure_notices", no_notices)
    monkeypatch.setattr(decentralized, "send_trace_checks", fake_send)

    await decentralized._run_decentralized_validator_async(settings)

    assert len(returned) == 1 and returned[0][1] == 0.25  # the round was scored
    if case == "save_failure":
        assert calls == []  # nothing is sent when the score file was not written
    else:
        assert len(calls) == 1 and calls[0][0] is outcome
        assert calls[0][1] == {"wallet": WALLET, "http": calls[0][1]["http"], "url": URL,
                               "service_hotkey": SERVICE, "rate": 0.01}


def test_miner_evaluation_keeps_positional_construction():
    item = MinerEvaluation(1, "hk-1", 10, EvaluationResult("passed", "", (), None), 5)
    assert item.trajectory is None
