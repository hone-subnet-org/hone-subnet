from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from rlvr import protocol
from rlvr.config import Settings
from rlvr.neurons import feedback_sender as sender
from rlvr.neurons.live import LiveSolverClient, SendGate
from rlvr.v3.api import MinerFailureNotice
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.round import MinerEvaluation, RoundResult, compute_round_payments

DISPLAY = 'Required stdout: "red-fox\\n"'
PRIVATE = "candidate output and private grading reason"


@pytest.fixture(autouse=True)
def offline_signatures(monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)


def evaluation(uid=7, hotkey=None, *, status="failed", code=MinerReason.CHECK_FAILED,
               stage=Stage.CHECK, display=DISPLAY):
    result = EvaluationResult(status, "" if status == "passed" else PRIVATE, (), None, code, stage)
    # Exercise the defensive transport boundary with malformed optional metadata too.
    object.__setattr__(result, "failed_check", display)
    return MinerEvaluation(uid, hotkey or f"miner-{uid}", 10, result, 12)


def outcome(*evaluations, status="completed", assigned=None):
    return RoundResult(
        status, "", tuple(evaluations), challenge_id="challenge", task_id="a" * 64,
        assigned_miners=tuple((e.uid, e.hotkey) for e in evaluations) if assigned is None else assigned,
    )


def solver(http, uid=7, hotkey=None, gate=None):
    return LiveSolverClient(
        uid, hotkey or f"miner-{uid}", f"http://miner-{uid}", "validator",
        Settings(_env_file=None), http, gate=gate,
    )


async def deliver(result, handler, *, include_details=True, registrations=None):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        registrations = registrations or [(item.uid, item.hotkey) for item in result.evaluations]
        return await sender.send_failure_notices(
            result, [solver(http, uid, hotkey) for uid, hotkey in registrations],
            wallet="validator", http=http, include_details=include_details,
        )


async def test_signed_direct_notice_preserves_details_without_private_output_or_score_changes():
    result = outcome(evaluation(), evaluation(8, status="passed", display=None))
    before = repr(result)
    received = []

    async def handler(request):
        body = await request.aread()
        assert request.url.path == "/v3/failure"
        assert request.url.host == "miner-7"
        assert protocol.verify_signature(request.headers, body, expected_signed_for="miner-7")
        assert request.headers["Epistula-Signed-By"] == "validator"
        assert PRIVATE.encode() not in body
        received.append(MinerFailureNotice.model_validate_json(body))
        return httpx.Response(200, json={"accepted": True})

    await deliver(result, handler)
    assert len(received) == 1
    assert received[0].failure.failed_check == DISPLAY
    assert repr(result) == before
    assert compute_round_payments(result, speed_half_life_ms=180_000, speed_floor=0.95) == {7: 0, 8: 1}


@pytest.mark.parametrize("status", ["abandoned", "unavailable"])
async def test_noncompleted_round_cannot_send_even_with_diagnostic_evaluations(status):
    def unexpected(_):
        pytest.fail("noncompleted round disclosed a notice")

    await deliver(outcome(evaluation(), status=status), unexpected)


async def test_only_assigned_exact_registrations_receive_one_notice():
    result = outcome(
        evaluation(), evaluation(), evaluation(8), evaluation(9, "retired"),
        assigned=((7, "miner-7"), (9, "current")),
    )
    received = []

    async def handler(request):
        received.append(json.loads(await request.aread())["uid"])
        return httpx.Response(200)

    await deliver(result, handler, registrations=[(7, "miner-7"), (8, "miner-8"), (9, "current")])
    assert received == [7]


@pytest.mark.parametrize("status,code,stage,display,details,reason", [
    ("failed", MinerReason.CHECK_FAILED, Stage.CHECK, DISPLAY, False, "check_failed"),
    ("failed", MinerReason.CHECK_FAILED, Stage.SETUP, DISPLAY, True, "check_failed"),
    ("failed", MinerReason.TIMEOUT, Stage.CHECK, DISPLAY, True, "timeout"),
    ("rejected", MinerReason.PATCH_REJECTED, Stage.PATCH, DISPLAY, True, "patch_rejected"),
    ("failed", None, Stage.CHECK, DISPLAY, True, "evaluation_failed"),
    ("failed", PRIVATE, Stage.CHECK, DISPLAY, True, "evaluation_failed"),
    ("failed", RoundReason.VALIDATOR_ERROR, Stage.CHECK, DISPLAY, True, "evaluation_failed"),
    ("failed", MinerReason.CHECK_FAILED, Stage.CHECK, "x" * 2047, True, "check_failed"),
    ("failed", MinerReason.CHECK_FAILED, Stage.CHECK, {"actual": PRIVATE}, True, "check_failed"),
])
async def test_reason_only_fallback(status, code, stage, display, details, reason):
    received = []

    async def handler(request):
        received.append(json.loads(await request.aread())["failure"])
        return httpx.Response(200)

    await deliver(outcome(evaluation(status=status, code=code, stage=stage, display=display)),
                  handler, include_details=details)
    assert received == [{"version": 1, "reason_code": reason, "failed_check": None}]


@pytest.mark.parametrize("failure", ["connect", "sign", "404", "redirect"])
async def test_send_failures_do_not_retry_follow_redirects_or_change_scores(failure, monkeypatch):
    requests = []
    result = outcome(evaluation())
    before = repr(result)
    if failure == "sign":
        def broken(*args, **kwargs):
            raise RuntimeError("signing unavailable")
        monkeypatch.setattr(sender, "sign_message", broken)

    async def handler(request):
        requests.append(request)
        if failure == "connect":
            raise httpx.ConnectError("offline")
        return httpx.Response(404 if failure == "404" else 307, headers={"Location": "http://other/"})

    await deliver(result, handler)
    assert len(requests) == (0 if failure == "sign" else 1)
    assert repr(result) == before
    assert compute_round_payments(result, speed_half_life_ms=1, speed_floor=1) == {7: 0}


async def test_signing_waits_for_shared_send_gate_and_releases_it(monkeypatch):
    gate = SendGate(1)
    held = await gate.acquire()
    signed = []
    original = sender.sign_message

    def signing(*args, **kwargs):
        signed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(sender, "sign_message", signing)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as http:
        task = asyncio.create_task(sender.send_failure_notices(
            outcome(evaluation()), [solver(http, gate=gate)], wallet="validator", http=http,
            include_details=True,
        ))
        try:
            await asyncio.sleep(0.02)
            assert signed == []
        finally:
            held.release()
            await task
    assert signed == [True]
    assert gate.available == 1


async def test_per_recipient_deadline_bounds_a_stalled_exchange(monkeypatch):
    monkeypatch.setattr(sender, "NOTICE_TIMEOUT_S", 0.03)
    monkeypatch.setattr(sender, "NOTICE_BATCH_DEADLINE_S", 0.6)
    cancelled = []

    async def handler(_):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    start = time.monotonic()
    await deliver(outcome(evaluation()), handler)
    assert time.monotonic() - start < 0.3
    assert cancelled == [True]


async def test_batch_deadline_includes_gate_wait_and_leaves_no_work(monkeypatch):
    monkeypatch.setattr(sender, "NOTICE_BATCH_DEADLINE_S", 0.03)
    gate = SendGate(1)
    held = await gate.acquire()
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        try:
            await asyncio.wait_for(sender.send_failure_notices(
                outcome(evaluation()), [solver(http, gate=gate)], wallet="validator", http=http,
                include_details=True,
            ), timeout=0.3)
            assert requests == []
        finally:
            held.release()
        await asyncio.sleep(0)
    assert gate.available == 1
    assert requests == []


async def test_concurrent_requests_are_bounded():
    active = maximum = 0
    full = asyncio.Event()
    release = asyncio.Event()

    async def handler(_):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 8:
            full.set()
        try:
            await release.wait()
            return httpx.Response(200)
        finally:
            active -= 1

    task = asyncio.create_task(deliver(outcome(*(evaluation(uid) for uid in range(24))), handler))
    try:
        await asyncio.wait_for(full.wait(), timeout=1)
        assert maximum == 8
    finally:
        release.set()
        await task
    assert maximum == 8 and active == 0


async def test_batch_caps_unique_recipient_count():
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200)

    await deliver(outcome(*(evaluation(uid) for uid in range(1030))), handler)
    assert len(requests) <= 1024
    assert len(requests) > 0


async def test_cancelling_sender_cleans_up_requests_and_shared_gate():
    entered = asyncio.Event()
    exited = asyncio.Event()
    gate = SendGate(1)

    async def handler(_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        task = asyncio.create_task(sender.send_failure_notices(
            outcome(evaluation()), [solver(http, gate=gate)], wallet="validator", http=http,
            include_details=True,
        ))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
        assert gate.available == 1


async def test_ack_body_is_not_consumed_and_cannot_delay_scoring():
    class UnboundedAck(httpx.AsyncByteStream):
        read = False
        closed = False

        async def __aiter__(self):
            self.read = True
            await asyncio.Event().wait()
            yield b"never consumed"

        async def aclose(self):
            self.closed = True

    stream = UnboundedAck()
    await asyncio.wait_for(deliver(
        outcome(evaluation()), lambda _: httpx.Response(200, stream=stream),
    ), timeout=1)
    assert not stream.read
    assert stream.closed


async def test_sender_to_demo_miner_with_real_signatures(monkeypatch):
    keypair = pytest.importorskip("bittensor_wallet").Keypair
    from rlvr.neurons import demo_miner

    validator_key = keypair.create_from_uri("//Alice")
    miner_key = keypair.create_from_uri("//Bob")
    validator_wallet = SimpleNamespace(hotkey=validator_key)
    miner_wallet = SimpleNamespace(hotkey=miner_key)
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", True)
    monkeypatch.setattr(protocol, "_Keypair", keypair)
    printed = []
    monkeypatch.setattr(demo_miner, "print_feedback", lambda lines: printed.extend(lines))

    class UnusedProvider:
        async def complete(self, *_args, **_kwargs):
            pytest.fail("a failure notice must never call the model provider")

    miner = demo_miner.DemoMiner(
        demo_miner.DemoMinerSettings(_env_file=None), UnusedProvider(), wallet=miner_wallet,
        metagraph=SimpleNamespace(hotkeys=[validator_key.ss58_address], validator_permit=[True]),
    )
    miner.served_tasks.add(
        validator_key.ss58_address, "challenge", "a" * 64, 7,
        miner_key.ss58_address, time.monotonic(),
    )
    app = demo_miner.build_demo_miner_app(miner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as http:
        client = LiveSolverClient(
            7, miner_key.ss58_address, "http://miner", validator_wallet,
            Settings(_env_file=None), http,
        )
        result = outcome(evaluation(hotkey=miner_key.ss58_address))
        accepted = await sender.send_failure_notices(
            result, [client], wallet=validator_wallet, http=http, include_details=True,
        )
    assert accepted == 1
    assert any("check_failed" in line for line in printed)
    assert printed[-1] == "[demo-miner] feedback:   " + DISPLAY
    assert all(PRIVATE not in line for line in printed)
