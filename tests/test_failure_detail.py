"""A rejected response is explained: the exact cause reaches the record, the journal and the notice."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import time

import httpx
import pytest

from rlvr import protocol
from rlvr.config import Settings
from rlvr.neurons import demo_miner
from rlvr.neurons import feedback_sender as sender
from rlvr.neurons.live import LiveSolverClient
from rlvr.protocol import crypto_available, sign_message
from rlvr.v3 import diagnostics
from rlvr.v3.api import (
    FailureExplanation,
    LeaseResponse,
    MinerFailureNotice,
    MinerSubmission,
    MinerTaskRequest,
    MinerTaskResponse,
    derive_miner_request_id,
)
from rlvr.v3.client import V3ProblemServerClient
from rlvr.v3.diagnostics import EvaluationLog, round_records
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.identity import compute_task_id
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.round import DispatchFailure, MinerEvaluation, RoundResult, dispatch_summary, evaluate_round
from tests.test_demo_miner_feedback import deliver_notice, make_receiver, remember_notice
from tests.test_v3_api import lease, slot_set
from tests.test_v3_archive import compress, make_tar
from tests.test_v3_grading import eval_repo, eval_term, result as container, runner  # noqa: F401 - fixture
from tests.test_v3_round import Solver, artifact_ref, policy, streamed

TASK = "a" * 64


# --------------------------------------------------------------------------- #
# The solver names what was wrong with the response
# --------------------------------------------------------------------------- #
class WalletLike:
    def __init__(self, hotkey):
        self.hotkey = hotkey


def task_for(miner_hotkey):
    leased = lease()
    return MinerTaskRequest(
        protocol_version=3,
        challenge_id=leased["challenge_id"],
        task_id=leased["task_id"],
        identity=leased["identity"],
        workspace=leased["workspace"],
        workspace_url=leased["workspace_url"],
        expires_at=2**53 - 1,
        slots=slot_set(uid=7, hotkey=miner_hotkey),
    )


def good_response(task, **changes):
    """A valid response body, with `changes` applied to its submission as raw JSON."""
    document = MinerTaskResponse(
        protocol_version=3,
        challenge_id=task.challenge_id,
        task_id=task.task_id,
        response_type="repository_patch_v1",
        submission={
            "artifact_role": "patch",
            "artifact_format": "unified_diff_v1",
            "upload_id": task.slots.submission.upload_id,
            "sha256": "c" * 64,
            "size_bytes": 0,
        },
        trajectory={
            "artifact_role": "trajectory",
            "artifact_format": "trajectory_v1",
            "upload_id": task.slots.trajectory.upload_id,
            "sha256": "d" * 64,
            "size_bytes": 1,
        },
    ).model_dump(mode="json")
    document["submission"].update(changes)
    return json.dumps(document).encode("utf-8")


def signed(signer, body, request, **header_changes):
    headers = sign_message(WalletLike(signer), body, signed_for=request.headers["Epistula-Signed-By"])
    headers.update(header_changes)
    return httpx.Response(200, content=body, headers=headers)


async def dispatch(handler, validator, miner):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LiveSolverClient(
        uid=7, hotkey=miner.ss58_address, url="http://miner",
        wallet=WalletLike(validator), settings=Settings(_env_file=None, solve_deadline_s=3600.0), http=http,
    )
    try:
        return await client.solve_v3(task_for(miner.ss58_address))
    finally:
        await http.aclose()


def cases():
    """(name, handler builder, pattern the error must match)."""
    def different_hotkey(task, validator, miner, other):
        async def handler(request):
            return signed(other, good_response(task), request)
        return handler

    def stale_timestamp(task, validator, miner, other):
        async def handler(request):
            stale = str(round(time.time() * 1000) - 20_000)
            return signed(miner, good_response(task), request, **{"Epistula-Timestamp": stale})
        return handler

    def bad_signature(task, validator, miner, other):
        async def handler(request):
            response = signed(miner, good_response(task), request)
            return httpx.Response(200, content=good_response(task) + b" ", headers=response.headers)
        return handler

    def unparsable(task, validator, miner, other):
        async def handler(request):
            return signed(miner, b"{}", request)
        return handler

    def wrong_slot(task, validator, miner, other):
        async def handler(request):
            return signed(miner, good_response(task, upload_id="u" * 32), request)
        return handler

    def oversized(task, validator, miner, other):
        async def handler(request):
            too_big = task.slots.submission.max_bytes + 1
            return signed(miner, good_response(task, size_bytes=too_big), request)
        return handler

    def wrong_role(task, validator, miner, other):
        # Role and format are tied to the response type by the model, so a
        # mismatch shows up as a parse error; only upload ID and size reach the slot check.
        async def handler(request):
            body = good_response(task, artifact_role="script", artifact_format="bash_script_v1")
            return signed(miner, body, request)
        return handler

    def wrong_format(task, validator, miner, other):
        # The model refuses a patch in script format before the slot is compared.
        async def handler(request):
            return signed(miner, good_response(task, artifact_format="bash_script_v1"), request)
        return handler

    def validator_bug(task, validator, miner, other):
        async def handler(request):
            raise RuntimeError("/home/validator/.secrets leaked")
        return handler

    def http_error(task, validator, miner, other):
        async def handler(request):
            return httpx.Response(503, content=b"maintenance at proxy 10.0.0.7 (secrets)")
        return handler

    def dropped(task, validator, miner, other):
        async def handler(request):
            raise httpx.RemoteProtocolError("Server disconnected")
        return handler

    def read_error_about_the_validator(task, validator, miner, other):
        async def handler(request):
            raise httpx.ReadError("[Errno 104] via proxy /home/validator/.secrets")
        return handler

    def timed_out(task, validator, miner, other):
        async def handler(request):
            raise httpx.ReadTimeout("")
        return handler

    def unreachable(task, validator, miner, other):
        async def handler(request):
            raise httpx.ConnectTimeout("")
        return handler

    def pool_exhausted(task, validator, miner, other):
        async def handler(request):
            raise httpx.PoolTimeout("")
        return handler

    def huge_timestamp(task, validator, miner, other):
        async def handler(request):
            return signed(miner, good_response(task), request, **{"Epistula-Timestamp": "9" * 400})
        return handler

    def absurd_timestamp(task, validator, miner, other):
        # Still a float, 1e305 s off: not a clock that is off, so no skew is named.
        async def handler(request):
            return signed(miner, good_response(task), request, **{"Epistula-Timestamp": "9" * 308})
        return handler

    def format_character_key(task, validator, miner, other):
        # The key is NFC on the wire only because U+200D blocks composition;
        # with the format character dropped, "e" + U+0301 would not be.
        key = "e\u200d\u0301"

        async def handler(request):
            return signed(miner, good_response(task, **{key: 1}), request)
        return handler

    return [
        ("different-hotkey", different_hotkey, r"signed by a different hotkey"),
        ("stale-timestamp", stale_timestamp, r"signature timestamp outside the \+/-8 s window: off by -(19|20|21)\.\d s"),
        ("bad-signature", bad_signature, r"signature does not verify"),
        ("unparsable", unparsable, r"response does not parse: \S+: .+"),
        ("wrong-slot", wrong_slot, r"submission does not match its slot: upload ID"),
        ("oversized", oversized, r"submission does not match its slot: \d+ bytes, limit \d+"),
        ("wrong-role", wrong_role, r"response does not parse: body: .+"),
        ("wrong-format", wrong_format, r"response does not parse: submission: .+"),
        ("validator-bug", validator_bug, r"validator error: RuntimeError"),
        ("http-error", http_error, r"HTTP 503: maintenance at proxy 10\.0\.0\.7 \(secrets\)"),
        ("dropped", dropped, r"RemoteProtocolError"),
        ("read-error", read_error_about_the_validator, r"ReadError"),
        ("timed-out", timed_out, r"no response within \d+ s"),  # the time actually waited
        ("unreachable", unreachable, r"ConnectTimeout"),
        ("pool-exhausted", pool_exhausted, r"PoolTimeout"),
        ("huge-timestamp", huge_timestamp, r"signature does not verify"),
        ("absurd-timestamp", absurd_timestamp, r"signature does not verify"),
        ("format-character-key", format_character_key, r"response does not parse: submission\.\u00e9: Extra inputs are not permitted"),
    ]


async def through_record_and_notice(committed, monkeypatch):
    """What the round does with a failed submission: the record line and the notice sent."""
    failure = DispatchFailure(
        committed.uid, committed.hotkey, MinerReason.RESPONSE_UNAVAILABLE, committed.error, committed.latency_ms
    )
    result = completed([server_verdict(committed.uid, hotkey=committed.hotkey)], [failure])
    record = miner_records(result)[committed.uid]
    received = []

    async def handler(request):
        received.append(json.loads(await request.aread())["failure"])
        return httpx.Response(200)

    with monkeypatch.context() as offline:
        offline.setattr(protocol, "_HAVE_CRYPTO", False)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = LiveSolverClient(
                committed.uid, committed.hotkey, "http://miner", "validator", Settings(_env_file=None, solve_deadline_s=3600.0), http
            )
            assert await sender.send_failure_notices(
                result, [client], wallet="validator", http=http, include_details=True
            ) == 1
    (notice,) = received
    return record, notice


@pytest.mark.skipif(not crypto_available(), reason="no sr25519 crypto stack installed")
@pytest.mark.parametrize("name, build, pattern", cases(), ids=[case[0] for case in cases()])
async def test_the_exact_cause_reaches_the_record_and_the_notice(name, build, pattern, monkeypatch, capsys):
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")
    other = Keypair.create_from_uri("//Eve")
    task = task_for(miner.ss58_address)
    committed, parsed = await dispatch(build(task, validator, miner, other), validator, miner)
    assert parsed is None
    assert re.fullmatch(pattern, committed.error), committed.error
    assert type(committed.latency_ms) is int and committed.latency_ms >= 0
    # Whatever the validator's own exceptions said stays out of what is committed.
    # An HTTP error body is the miner's own response and was always committed;
    # it is the notice that must not carry it.
    assert "secrets" not in committed.error or name == "http-error"
    journal = capsys.readouterr().out
    if name == "validator-bug":
        assert "RuntimeError: /home/validator/.secrets leaked" in journal  # the operator still sees it
    else:
        assert journal == ""

    record, notice = await through_record_and_notice(committed, monkeypatch)
    assert record["dispatch_reason_code"] == "response_unavailable"
    assert record["dispatch_detail"] == committed.error
    assert record["response_latency_ms"] == committed.latency_ms
    assert notice["dispatch_reason_code"] == "response_unavailable"
    if name == "http-error":
        assert notice["dispatch_detail"] == "HTTP 503"  # the unsigned body is not forwarded to the miner
    else:
        assert notice["dispatch_detail"] == committed.error[:200]
    assert notice["reason_code"] == "artifact_invalid"
    assert "secrets" not in json.dumps(notice)


@pytest.mark.skipif(not crypto_available(), reason="no sr25519 crypto stack installed")
@pytest.mark.parametrize("offset_ms, accepted", [(-7_500, True), (7_500, True), (-8_500, False), (8_500, False)])
async def test_the_timestamp_message_never_changes_what_verification_accepts(offset_ms, accepted, monkeypatch):
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")
    task = task_for(miner.ss58_address)

    async def handler(request):
        body = good_response(task)
        headers = sign_message(WalletLike(miner), body, signed_for=request.headers["Epistula-Signed-By"])
        return httpx.Response(200, content=body, headers=headers)

    # The miner signs with a clock off by offset_ms; the validator verifies with the real one.
    real_clock = time.time
    shift = [offset_ms / 1000]
    monkeypatch.setattr(protocol.time, "time", lambda: real_clock() + shift[0])

    async def shifted(request):
        response = await handler(request)
        shift[0] = 0.0
        return response

    committed, parsed = await dispatch(shifted, validator, miner)
    if accepted:
        assert parsed is not None and committed.error == ""
    else:
        assert parsed is None
        assert re.fullmatch(r"signature timestamp outside the \+/-8 s window: off by [+-]\d+\.\d s", committed.error), committed.error


@pytest.mark.skipif(not crypto_available(), reason="no sr25519 crypto stack installed")
async def test_a_signing_failure_names_the_class_and_keeps_the_wallet_path_in_the_journal(monkeypatch, capsys):
    from bittensor_wallet import Keypair

    miner = Keypair.create_from_uri("//Dave")

    class UnreadableWallet:
        @property
        def hotkey(self):
            raise RuntimeError("Keyfile at: /home/validator/.bittensor/wallets/vali/hotkeys/default is not readable")

    async def never_called(request):
        raise AssertionError("no request may leave without a signature")

    http = httpx.AsyncClient(transport=httpx.MockTransport(never_called))
    client = LiveSolverClient(
        uid=7, hotkey=miner.ss58_address, url="http://miner",
        wallet=UnreadableWallet(), settings=Settings(_env_file=None, solve_deadline_s=3600.0), http=http,
    )
    try:
        committed, parsed = await client.solve_v3(task_for(miner.ss58_address))
    finally:
        await http.aclose()
    assert parsed is None and committed.error == "validator error: RuntimeError"
    assert "Keyfile at: /home/validator/.bittensor" in capsys.readouterr().out
    record, notice = await through_record_and_notice(committed, monkeypatch)
    assert record["dispatch_detail"] == "validator error: RuntimeError"
    assert notice["dispatch_detail"] == "validator error: RuntimeError"
    assert ".bittensor" not in json.dumps(notice)


@pytest.mark.skipif(not crypto_available(), reason="no sr25519 crypto stack installed")
async def test_the_journal_line_for_a_validator_exception_is_one_printable_line(capsys):
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")

    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no message for you")

    async def control_sequence(request):
        raise RuntimeError("failure\n\x1b[2Jforged journal entry " + "x" * 300)

    async def cannot_print(request):
        raise Unprintable()

    committed, parsed = await dispatch(control_sequence, validator, miner)
    assert parsed is None and committed.error == "validator error: RuntimeError"
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("[validator] WARN: dispatch to uid 7 failed inside the exchange: RuntimeError: failure [2Jforged")
    assert line.isprintable() and len(line) < 300

    committed, parsed = await dispatch(cannot_print, validator, miner)
    assert parsed is None and committed.error == "validator error: Unprintable"
    (line,) = capsys.readouterr().out.splitlines()
    assert line.endswith("failed inside the exchange: Unprintable")


@pytest.mark.skipif(not crypto_available(), reason="no sr25519 crypto stack installed")
async def test_the_timeout_message_states_the_time_actually_waited():
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")

    async def silent_for_a_while(request):
        await asyncio.sleep(1.2)
        raise httpx.ReadTimeout("")

    committed, parsed = await dispatch(silent_for_a_while, validator, miner)
    assert parsed is None and committed.error == "no response within 1 s"
    assert committed.latency_ms >= 1_200


@pytest.mark.skipif(not crypto_available(), reason="no sr25519 crypto stack installed")
async def test_a_failed_dispatch_is_timed_until_the_failure():
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")

    async def slow_refusal(request):
        time.sleep(0.05)
        return httpx.Response(500, content=b"")

    committed, parsed = await dispatch(slow_refusal, validator, miner)
    assert parsed is None and committed.error == "HTTP 500: "
    assert committed.latency_ms >= 50


# --------------------------------------------------------------------------- #
# The round keeps what the solver said, and times a solver that raises
# --------------------------------------------------------------------------- #
async def test_round_keeps_the_solver_error_and_times_an_exception(tmp_path):
    workspace_tar = make_tar([("repo", "dir", b"", 0o755), ("repo/a.txt", "file", b"a", 0o644)])
    workspace_blob = compress(workspace_tar)
    workspace_ref = artifact_ref("workspace", workspace_tar, workspace_blob)
    verifier_tar = make_tar([("manifest.json", "file", b"{}", 0o644)])
    verifier_ref = artifact_ref("verifier", verifier_tar, compress(verifier_tar))
    identity = lease()["identity"].model_copy(
        update={"workspace_sha256": workspace_ref.sha256, "verifier_sha256": verifier_ref.sha256}
    )
    task_id = compute_task_id(identity)
    slots = []
    for uid in (1, 2, 3, 4):
        current = slot_set(uid=uid, hotkey=f"hk-{uid}", prefix=f"u{uid}")
        slots.append(current.model_validate({
            "submission": current.submission.model_dump() | {"task_id": task_id},
            "trajectory": current.trajectory.model_dump() | {"task_id": task_id},
        }))
    leased = LeaseResponse(**lease(
        identity=identity, task_id=task_id, slot_pool=slots, commit_min_signed_responses=1,
        workspace=workspace_ref, verifier=verifier_ref, expires_at=2**53 - 1,
    ))

    committed = {}

    async def handler(request):
        if request.url.path == "/v3/challenges/lease":
            return httpx.Response(200, content=leased.model_dump_json())
        if request.url.path == "/workspace":
            return streamed(workspace_blob)
        if request.url.path == "/v3/challenges/commit":
            for item in json.loads(await request.aread())["submissions"]:
                committed[item["uid"]] = item["error"]
            return httpx.Response(403)  # the round ends here; the commit body is what matters
        return httpx.Response(404)

    class Rejecting:
        uid, hotkey = 1, "hk-1"

        async def solve_v3(self, task):
            submission = MinerSubmission(
                uid=1, hotkey="hk-1", request_id=derive_miner_request_id(task.challenge_id, 1, "hk-1"),
                response_body="", response_headers={}, error="signature does not verify", latency_ms=77,
            )
            return submission, None

    class Exploding:
        uid, hotkey = 2, "hk-2"

        async def solve_v3(self, task):
            await asyncio.sleep(0.02)
            # Longer than the 4 KiB the committed error field accepts, and
            # multibyte, so a careless cut would not even be valid UTF-8.
            raise ValueError("boom " + "±" * 5_000)

    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no message for you")

    class Mute:
        uid, hotkey = 3, "hk-3"

        async def solve_v3(self, task):
            raise Unprintable()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = V3ProblemServerClient("https://problems.invalid", "validator", http, retries=1)
        result = await evaluate_round(
            client, http, [Rejecting(), Exploding(), Mute(), Solver(4, "hk-4", b"pass")],
            dataclasses.replace(policy(tmp_path), commit_quorum=1),
            cache_dir=tmp_path / "cache", work_dir=tmp_path / "work",
            candidates=[(1, "hk-1"), (2, "hk-2"), (3, "hk-3"), (4, "hk-4")],
        )
    # No exception took the round down: the one signed response met quorum and
    # the round went on to commit, where the fixture server refused it.
    assert (result.status, result.reason_code) == ("abandoned", RoundReason.COMMIT_FAILED)
    assert result.signed_responses == 1
    # The problem server gets the solver's own words, and for a validator
    # exception the class alone; the message stays in the local record.
    assert committed == {
        1: "signature does not verify", 2: "validator error: ValueError", 3: "validator error: Unprintable", 4: "",
    }
    rejected, exploded, mute = result.dispatch_failures
    assert rejected == DispatchFailure(1, "hk-1", MinerReason.RESPONSE_UNAVAILABLE, "signature does not verify", 77)
    assert exploded.code is MinerReason.DISPATCH_FAILED
    assert exploded.detail.startswith("ValueError: boom ±") and len(exploded.detail.encode()) <= 512
    assert exploded.latency_ms >= 20
    assert (mute.code, mute.detail) == (MinerReason.DISPATCH_FAILED, "Unprintable")
    records = miner_records(result)
    assert (records[1]["dispatch_detail"], records[1]["response_latency_ms"]) == ("signature does not verify", 77)
    assert (records[2]["dispatch_detail"], records[2]["response_latency_ms"]) == (exploded.detail, exploded.latency_ms)
    assert records[3]["dispatch_detail"] == "Unprintable"
    assert dispatch_summary(result) == "dispatch: 1 ok, 1 rejected (1 signature does not verify), 2 dispatch failed"


async def test_a_round_that_never_dispatched_has_no_summary(tmp_path):
    async def handler(request):
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = V3ProblemServerClient("https://problems.invalid", "validator", http, retries=1)
        result = await evaluate_round(
            client, http, [Solver(1, "hk-1", b"pass")], policy(tmp_path),
            cache_dir=tmp_path / "cache", work_dir=tmp_path / "work", candidates=[(1, "hk-1")],
        )
    assert result.status != "completed"
    assert result.signed_responses is None and dispatch_summary(result) is None


# --------------------------------------------------------------------------- #
# The record keeps the cause and the time to it
# --------------------------------------------------------------------------- #
def rejection(uid, detail, code=MinerReason.RESPONSE_UNAVAILABLE, latency_ms=42):
    return DispatchFailure(uid, f"hk-{uid}", code, detail, latency_ms)


def server_verdict(uid, latency_ms=0, hotkey=None):
    verdict = EvaluationResult(
        "rejected", "artifact_invalid", (), None, MinerReason.ARTIFACT_INVALID, Stage.COMMIT,
    )
    return MinerEvaluation(uid, hotkey or f"hk-{uid}", latency_ms, verdict)


def completed(evaluations, failures, signed_responses=None):
    return RoundResult(
        "completed", "", tuple(evaluations), challenge_id="chal-1", task_id=TASK, checks_total=2,
        assigned_miners=tuple((item.uid, item.hotkey) for item in evaluations),
        diagnostic_evaluations=tuple(evaluations), dispatch_failures=tuple(failures),
        signed_responses=len(evaluations) - len(failures) if signed_responses is None else signed_responses,
    )


def miner_records(result):
    return {item["uid"]: item for item in round_records(result, "validator") if item["record_type"] == "miner_evaluation"}


def test_record_keeps_the_dispatch_detail_beside_the_server_verdict():
    result = completed(
        [server_verdict(1), server_verdict(2)],
        [rejection(1, "signature does not verify"), rejection(2, "no response within 1170 s", latency_ms=1_170_000)],
    )
    records = miner_records(result)
    assert (records[1]["reason_code"], records[1]["dispatch_reason_code"]) == ("artifact_invalid", "response_unavailable")
    assert records[1]["dispatch_detail"] == "signature does not verify"
    assert records[1]["response_latency_ms"] == 42
    assert records[2]["dispatch_detail"] == "no response within 1170 s"
    assert records[2]["response_latency_ms"] == 1_170_000


def test_record_times_a_failed_dispatch_even_without_an_evaluation():
    result = RoundResult(
        "abandoned", "signed response quorum was not met", (), reason_code=RoundReason.QUORUM_NOT_MET,
        stage=Stage.DISPATCH, challenge_id="chal-1", task_id=TASK,
        assigned_miners=((1, "hk-1"), (2, "hk-2"), (3, "hk-3")),
        dispatch_failures=(
            rejection(1, "RemoteProtocolError: Server disconnected", latency_ms=310),
            rejection(2, "ValueError: boom", MinerReason.DISPATCH_FAILED, latency_ms=5),
            rejection(3, "miner is not serving", MinerReason.NOT_SERVING, latency_ms=None),
        ),
    )
    records = miner_records(result)
    assert all(records[uid]["status"] == "not_evaluated" for uid in (1, 2, 3))
    assert (records[1]["dispatch_detail"], records[1]["response_latency_ms"]) == ("RemoteProtocolError: Server disconnected", 310)
    assert (records[2]["dispatch_reason_code"], records[2]["dispatch_detail"], records[2]["response_latency_ms"]) == ("dispatch_failed", "ValueError: boom", 5)
    assert (records[3]["dispatch_detail"], records[3]["response_latency_ms"]) == ("miner is not serving", None)


def test_record_without_a_dispatch_failure_has_no_detail():
    verdict = EvaluationResult("passed", "", (), None, None, Stage.CHECK)
    result = completed([MinerEvaluation(1, "hk-1", 10, verdict)], [])
    record = miner_records(result)[1]
    assert record["dispatch_detail"] is None and record["dispatch_reason_code"] is None


def test_dispatch_detail_is_bounded_and_printable():
    result = completed([server_verdict(1)], [rejection(1, "HTTP 503: " + "\x1b[31m" + "x\n" * 2_000)])
    detail = miner_records(result)[1]["dispatch_detail"]
    assert detail.startswith("HTTP 503: [31mx x") and len(detail.encode()) <= 512
    assert detail.isprintable()


def test_dispatch_detail_is_dropped_after_the_check_display_and_before_the_reason(tmp_path, monkeypatch):
    big_display = "Required stdout: " + "x" * 5_000
    verdict = EvaluationResult(
        "failed", "r" * 100, (), None, MinerReason.CHECK_FAILED, Stage.CHECK, failed_check=big_display,
    )
    result = completed(
        [MinerEvaluation(1, "hk-1", 10, verdict)], [rejection(1, "d" * 400)],
    )
    record = miner_records(result)[1]
    without_display = dict(record, failed_check=None)
    without_detail = dict(without_display, dispatch_detail=None)
    # Room for the reason and the detail is gone, but the reason alone fits.
    monkeypatch.setattr(diagnostics, "MAX_RECORD_BYTES", len(diagnostics._encode(without_display)) - 1)
    assert len(diagnostics._encode(without_detail)) <= diagnostics.MAX_RECORD_BYTES
    log = EvaluationLog(str(tmp_path / "log.jsonl"))
    assert log.record_round(result, "validator") is True
    written = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    miner = next(item for item in written if item["record_type"] == "miner_evaluation")
    assert (miner["failed_check"], miner["dispatch_detail"], miner["reason"]) == (None, None, "r" * 100)


# --------------------------------------------------------------------------- #
# The journal sums dispatch up by cause
# --------------------------------------------------------------------------- #
def test_dispatch_summary_counts_by_cause():
    result = RoundResult(
        "completed", "", (), challenge_id="chal-1", task_id=TASK, signed_responses=3,
        assigned_miners=tuple((uid, f"hk-{uid}") for uid in range(1, 10)),
        dispatch_failures=(
            rejection(1, "signature timestamp outside the +/-8 s window: off by +9.4 s"),
            rejection(2, "signature timestamp outside the +/-8 s window: off by -12.0 s"),
            rejection(3, "response does not parse: submission.sha256: String should have at least 64 characters"),
            rejection(4, "no response within 1170 s"),
            rejection(5, "miner is not serving", MinerReason.NOT_SERVING, None),
            rejection(6, "ValueError: boom", MinerReason.DISPATCH_FAILED),
        ),
    )
    assert dispatch_summary(result) == (
        "dispatch: 3 ok, 4 rejected (2 signature timestamp outside the +/-8 s window, "
        "1 response does not parse, 1 no response within 1170 s), 1 not serving, 1 dispatch failed"
    )


def test_dispatch_summary_is_printable_whatever_a_solver_said():
    result = completed([server_verdict(1)], [rejection(1, "signature\x00 does\x1b[2J not\u2028 verify: x")])
    summary = dispatch_summary(result)
    assert summary == "dispatch: 0 ok, 1 rejected (1 signature does[2J not verify)" and summary.isprintable()


def test_dispatch_summary_of_a_clean_round():
    result = RoundResult("completed", "", (), challenge_id="chal-1", task_id=TASK,
                         assigned_miners=((1, "hk-1"), (2, "hk-2")), signed_responses=2)
    assert dispatch_summary(result) == "dispatch: 2 ok"


def test_no_dispatch_summary_before_dispatch_ran():
    result = RoundResult(
        "abandoned", "insufficient workspace storage", (), reason_code=RoundReason.INSUFFICIENT_STORAGE,
        stage=Stage.WORKSPACE_DOWNLOAD, challenge_id="chal-1", task_id=TASK,
        assigned_miners=((1, "hk-1"), (2, "hk-2")),
    )
    assert result.signed_responses is None and dispatch_summary(result) is None


# --------------------------------------------------------------------------- #
# Grading timeouts say which check, its limit and how long it ran
# --------------------------------------------------------------------------- #
def slow_timeout(request):
    """A container that runs for a while and is then stopped for its time limit."""
    time.sleep(0.15)
    return container(timed_out=True, exit_code=124)


def timed_out_reason(verdict, runner, what):
    """The limit and elapsed time the reason states, checked against the request made and the time taken."""
    assert verdict.status == "failed" and verdict.reason_code is MinerReason.TIMEOUT
    found = re.fullmatch(what + r" timed out: limit (\d+) s, elapsed (\d+\.\d) s", verdict.reason)
    assert found, verdict.reason
    assert int(found.group(1)) == runner.requests[-1].timeout_s
    assert float(found.group(2)) >= 0.1


def test_check_timeout_reason_names_the_check_its_limit_and_elapsed(tmp_path, runner):
    runner.responses = [container(), slow_timeout]
    timed_out_reason(eval_repo(tmp_path, runner), runner, "check c01")


def test_script_timeout_reason_names_the_script(tmp_path, runner):
    runner.responses = [slow_timeout]
    timed_out_reason(eval_term(tmp_path, runner), runner, "script")


def test_setup_timeout_reason_names_setup(tmp_path, runner):
    runner.responses = [slow_timeout]
    timed_out_reason(eval_repo(tmp_path, runner), runner, "setup")


# --------------------------------------------------------------------------- #
# The notice tells the miner what the validator saw
# --------------------------------------------------------------------------- #
@pytest.fixture
def offline_signatures(monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)


def solver(http, uid):
    return LiveSolverClient(uid, f"hk-{uid}", f"http://miner-{uid}", "validator", Settings(_env_file=None, solve_deadline_s=3600.0), http)


async def deliver(result, handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        return await sender.send_failure_notices(
            result, [solver(http, item.uid) for item in result.evaluations],
            wallet="validator", http=http, include_details=True,
        )


@pytest.mark.usefixtures("offline_signatures")
async def test_notice_carries_the_local_code_and_detail():
    received = []

    async def handler(request):
        received.append(json.loads(await request.aread())["failure"])
        return httpx.Response(200)

    result = completed([server_verdict(1)], [rejection(1, "signature does not verify")])
    assert await deliver(result, handler) == 1
    assert received == [{
        "version": 1, "reason_code": "artifact_invalid", "failed_check": None,
        "dispatch_reason_code": "response_unavailable", "dispatch_detail": "signature does not verify",
    }]


@pytest.mark.usefixtures("offline_signatures")
async def test_notice_detail_is_one_short_printable_line():
    received = []

    async def handler(request):
        received.append(json.loads(await request.aread())["failure"])
        return httpx.Response(200)

    result = completed([server_verdict(1)], [rejection(1, "response does not parse: \x1b[31m" + "x\n" * 300)])
    await deliver(result, handler)
    detail = received[0]["dispatch_detail"]
    assert detail.startswith("response does not parse: [31mx x") and len(detail) == 200 and detail.isprintable()


@pytest.mark.usefixtures("offline_signatures")
async def test_a_validator_exception_stays_in_the_record_and_out_of_the_notice():
    received = []

    async def handler(request):
        received.append(await request.aread())
        return httpx.Response(200)

    result = completed(
        [server_verdict(1)], [rejection(1, "ValueError: /home/validator/.secrets", MinerReason.DISPATCH_FAILED, 5)],
    )
    assert miner_records(result)[1]["dispatch_detail"] == "ValueError: /home/validator/.secrets"
    await deliver(result, handler)
    assert json.loads(received[0])["failure"] == {
        "version": 1, "reason_code": "artifact_invalid", "failed_check": None,
        "dispatch_reason_code": "dispatch_failed",
    }
    assert b"secrets" not in received[0]


def test_notice_refuses_a_detail_on_anything_but_a_rejected_response():
    with pytest.raises(ValueError):
        FailureExplanation(version=1, reason_code=MinerReason.ARTIFACT_INVALID,
                           dispatch_reason_code=MinerReason.DISPATCH_FAILED, dispatch_detail="x")
    with pytest.raises(ValueError):
        FailureExplanation(version=1, reason_code=MinerReason.ARTIFACT_INVALID, dispatch_detail="orphan")
    with pytest.raises(ValueError):
        FailureExplanation(version=1, reason_code=MinerReason.ARTIFACT_INVALID,
                           dispatch_reason_code=MinerReason.CHECK_FAILED)
    assert FailureExplanation(version=1, reason_code=MinerReason.ARTIFACT_INVALID,
                              dispatch_reason_code=MinerReason.DISPATCH_FAILED).dispatch_detail is None


def test_a_miner_chosen_key_in_a_parse_error_is_kept_short_and_printable():
    from pydantic import ValidationError
    from rlvr.neurons.live import _first_validation_error

    task = task_for("hk-7")
    body = good_response(task, **{"evil\n\x1b[31m" + "k" * 300: 1})  # an extra key inside the submission
    with pytest.raises(ValidationError) as caught:
        MinerTaskResponse.model_validate_json(body)
    text = _first_validation_error(caught.value)
    assert text.startswith("submission.evil[31mkkk") and text.isprintable()
    assert len(text.split(":")[0]) <= 80
    assert text.endswith("Extra inputs are not permitted")


def test_notice_detail_bounds_count_characters_and_the_largest_notice_fits():
    from rlvr.v3.api import DISPATCH_DETAIL_MAX_CHARS, FAILURE_NOTICE_MAX_BYTES, serialize_failure_notice
    from rlvr.v3.feedback import FAILED_CHECK_MAX_BYTES

    multibyte = "±" * DISPATCH_DETAIL_MAX_CHARS
    largest_display = "x" * (FAILED_CHECK_MAX_BYTES - 2)  # the quotes count once escaped
    explanation = FailureExplanation(
        version=1, reason_code=MinerReason.CHECK_FAILED, failed_check=largest_display,
        dispatch_reason_code=MinerReason.RESPONSE_UNAVAILABLE, dispatch_detail=multibyte,
    )
    rejected = {
        "version": 1, "reason_code": MinerReason.TIMEOUT, "dispatch_reason_code": MinerReason.RESPONSE_UNAVAILABLE,
    }
    assert FailureExplanation(**rejected, dispatch_detail=multibyte).dispatch_detail == multibyte
    with pytest.raises(ValueError):
        FailureExplanation(**rejected, dispatch_detail=multibyte + "±")
    with pytest.raises(ValueError):
        FailureExplanation(**rejected, dispatch_detail="tab\there")
    notice = MinerFailureNotice(
        protocol_version=3, message_type="failure_notice_v1", challenge_id="\x00" * 128, task_id=TASK,
        uid=7, hotkey="\x00" * 128, failure=explanation,
    )
    body = serialize_failure_notice(notice)
    assert len(body) < FAILURE_NOTICE_MAX_BYTES
    assert MinerFailureNotice.model_validate_json(body) == notice


def test_a_full_multibyte_record_is_written_within_4_kib_and_reads_back(tmp_path):
    verdict = EvaluationResult(
        "failed", "±" * 100, (), None, MinerReason.CHECK_FAILED, Stage.CHECK,
        failed_check="Required stdout: " + "x" * 3_500,
    )
    result = completed([MinerEvaluation(1, "hk-1", 10, verdict)], [rejection(1, "HTTP 503: " + "±" * 1_000)])
    log = EvaluationLog(str(tmp_path / "log.jsonl"))
    assert log.record_round(result, "validator") is True
    lines = (tmp_path / "log.jsonl").read_bytes().splitlines(keepends=True)
    assert all(len(line) <= diagnostics.MAX_RECORD_BYTES for line in lines)  # the newline counts
    miner = next(json.loads(line) for line in lines if json.loads(line)["record_type"] == "miner_evaluation")
    assert miner["failed_check"] is None  # dropped first to make the line fit
    assert miner["dispatch_detail"].startswith("HTTP 503: ±") and len(miner["dispatch_detail"].encode()) <= 512
    assert "\ufffd" not in miner["dispatch_detail"]
    assert miner["reason"] == "±" * 100


@pytest.mark.usefixtures("offline_signatures")
async def test_a_detail_the_wire_refuses_is_left_out_and_the_code_still_goes():
    received = []

    async def handler(request):
        received.append(json.loads(await request.aread())["failure"])
        return httpx.Response(200)

    # Printable and short, but not NFC, which the wire model refuses.
    detail = "response does not parse: submission.e\u0301: Extra inputs are not permitted"
    result = completed([server_verdict(1)], [rejection(1, detail)])
    assert await deliver(result, handler) == 1
    assert received == [{
        "version": 1, "reason_code": "artifact_invalid", "failed_check": None,
        "dispatch_reason_code": "response_unavailable",
    }]


@pytest.mark.usefixtures("offline_signatures")
async def test_notice_without_a_dispatch_failure_has_no_local_keys():
    received = []

    async def handler(request):
        received.append(json.loads(await request.aread())["failure"])
        return httpx.Response(200)

    verdict = EvaluationResult("failed", "tests failed", (), None, MinerReason.CHECK_FAILED, Stage.CHECK)
    await deliver(completed([MinerEvaluation(1, "hk-1", 10, verdict)], []), handler)
    assert received == [{"version": 1, "reason_code": "check_failed", "failed_check": None}]


@pytest.mark.usefixtures("offline_signatures")
async def test_a_miner_that_refuses_the_local_keys_still_gets_the_reason_once():
    bodies = []

    async def handler(request):
        bodies.append(json.loads(await request.aread())["failure"])
        return httpx.Response(400 if len(bodies) == 1 else 200)

    result = completed([server_verdict(1)], [rejection(1, "signature does not verify")])
    assert await deliver(result, handler) == 1
    assert [sorted(body) for body in bodies] == [
        ["dispatch_detail", "dispatch_reason_code", "failed_check", "reason_code", "version"],
        ["failed_check", "reason_code", "version"],
    ]


@pytest.mark.usefixtures("offline_signatures")
async def test_demo_miner_prints_what_the_validator_saw(monkeypatch):
    output = []
    monkeypatch.setattr(demo_miner, "print_feedback", output.append)
    miner = make_receiver()
    notice = MinerFailureNotice(
        protocol_version=3, message_type="failure_notice_v1", challenge_id="challenge", task_id=TASK,
        uid=7, hotkey="miner",
        failure=FailureExplanation(
            version=1, reason_code=MinerReason.ARTIFACT_INVALID,
            dispatch_reason_code=MinerReason.RESPONSE_UNAVAILABLE,
            dispatch_detail="signature timestamp outside the +/-8 s window: off by +9.4 s",
        ),
    )
    body = notice.model_dump_json().encode()
    remember_notice(miner, body)
    assert await deliver_notice(miner, body) == (200, {"accepted": True})
    assert output == [[
        "[demo-miner] feedback: artifact_invalid challenge challenge task " + "a" * 16,
        "[demo-miner] feedback:   validator saw: response_unavailable, "
        "signature timestamp outside the +/-8 s window: off by +9.4 s",
    ]]
