from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from rlvr import protocol
from rlvr.neurons import demo_miner
from rlvr.neurons.demo_miner import (
    FEEDBACK_PRINT_HEADER_CHARS,
    FEEDBACK_PRINT_MAX_CHARS,
    FEEDBACK_PRINT_MAX_LINES,
    SERVED_TASK_LIMIT,
    SERVED_TASK_TTL_S,
    DemoMiner,
    DemoMinerSettings,
    ServedTasks,
    build_demo_miner_app,
    feedback_lines,
    printable,
)
from rlvr.v3.api import (
    FAILURE_NOTICE_MAX_BYTES,
    FailureExplanation,
    MinerFailureNotice,
    MinerTaskResponse,
    serialize_failure_notice,
)
from rlvr.v3.feedback import FAILED_CHECK_MAX_BYTES
from rlvr.v3.reasons import MinerReason

IDENTITY = ("validator", "challenge", "task", 7, "miner")
PREFIX = "[demo-miner] feedback: "
DISPLAY_PREFIX = PREFIX + "  "


def test_served_task_accepts_only_the_exact_served_identity():
    cache = ServedTasks()
    cache.add(*IDENTITY, now=0)
    for index, replacement in enumerate(
        ("other-validator", "other-challenge", "other-task", 8, "other-miner")
    ):
        wrong = list(IDENTITY)
        wrong[index] = replacement
        assert cache.classify(*wrong, now=1) == "unknown"
    assert cache.classify(*IDENTITY, now=1) == "new"
    assert cache.classify(*IDENTITY, now=2) == "duplicate"


def test_served_task_history_is_lost_on_restart():
    original = ServedTasks()
    original.add(*IDENTITY, now=0)
    assert original.classify(*IDENTITY, now=1) == "new"
    assert ServedTasks().classify(*IDENTITY, now=1) == "unknown"


@pytest.mark.parametrize("first_notice", [False, True])
def test_exact_expiry_is_not_extended_by_recognized_access(first_notice):
    cache = ServedTasks(ttl_s=10)
    cache.add(*IDENTITY, now=50)
    if first_notice:
        assert cache.classify(*IDENTITY, now=51) == "new"
    expected = "duplicate" if first_notice else "new"
    assert cache.classify(*IDENTITY, now=59.999) == expected
    assert cache.classify(*IDENTITY, now=60) == "unknown"
    assert not cache.served
    assert not cache.answered


def test_served_task_defaults_bound_retention():
    cache = ServedTasks()
    assert cache.limit == SERVED_TASK_LIMIT == 256
    assert cache.ttl_s == SERVED_TASK_TTL_S == 7_200
    cache.add(*IDENTITY, now=0)
    assert cache.classify(*IDENTITY, now=7_200) == "unknown"


@pytest.mark.parametrize("access_is_duplicate", [False, True])
def test_recognized_access_changes_lru_order(access_is_duplicate):
    cache = ServedTasks(limit=2)
    first = (*IDENTITY[:2], "first", *IDENTITY[3:])
    second = (*IDENTITY[:2], "second", *IDENTITY[3:])
    third = (*IDENTITY[:2], "third", *IDENTITY[3:])
    cache.add(*first, now=0)
    if access_is_duplicate:
        assert cache.classify(*first, now=0.5) == "new"
    cache.add(*second, now=1)
    assert cache.classify(*first, now=2) == (
        "duplicate" if access_is_duplicate else "new"
    )
    cache.add(*third, now=3)
    assert cache.classify(*second, now=4) == "unknown"
    assert cache.classify(*first, now=4) == "duplicate"
    assert cache.classify(*third, now=4) == "new"


def test_wrong_registration_cannot_protect_an_entry_from_eviction():
    cache = ServedTasks(limit=2)
    cache.add(*IDENTITY, now=0)
    cache.add("validator", "challenge", "second", 7, "miner", now=1)
    assert cache.classify(*IDENTITY[:3], 8, "miner", now=2) == "unknown"
    cache.add("validator", "challenge", "third", 7, "miner", now=3)
    assert cache.classify(*IDENTITY, now=4) == "unknown"


def test_served_and_dedup_collections_remain_bounded_and_expire_together():
    cache = ServedTasks(limit=3, ttl_s=100)
    for index in range(20):
        identity = ("validator", "challenge", str(index), 7, "miner")
        cache.add(*identity, now=index)
        assert cache.classify(*identity, now=index) == "new"
        assert len(cache.served) <= 3
        assert len(cache.answered) <= 3
        assert cache.answered <= cache.served.keys()
    assert cache.classify(*IDENTITY, now=119) == "unknown"
    assert not cache.served
    assert not cache.answered


def test_add_purges_expired_answered_entries():
    cache = ServedTasks(ttl_s=10)
    cache.add(*IDENTITY, now=0)
    assert cache.classify(*IDENTITY, now=1) == "new"
    cache.add("validator", "challenge", "fresh", 7, "miner", now=10)
    assert len(cache.served) == 1
    assert not cache.answered
    assert cache.classify(*IDENTITY, now=10) == "unknown"


def test_repeated_served_response_does_not_reset_notice_dedup():
    cache = ServedTasks()
    cache.add(*IDENTITY, now=0)
    assert cache.classify(*IDENTITY, now=1) == "new"
    cache.add(*IDENTITY, now=2)
    assert cache.classify(*IDENTITY, now=3) == "duplicate"
    assert len(cache.served) == len(cache.answered) == 1


def test_printable_preserves_plain_ascii_including_quotes_and_backslashes():
    value = "".join(chr(code) for code in range(32, 127))
    assert printable(value) == value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("\x00", r"\x00"),
        ("\x08", r"\x08"),
        ("\x1b[2J", r"\x1b[2J"),
        ("\x7f", r"\x7f"),
        ("\x85", r"\x85"),
        ("\u2028", r"\u2028"),
        ("\u202e", r"\u202e"),
        ("\ud800", r"\ud800"),
        ("\U0001f600", r"\U0001f600"),
        ("\U0010ffff", r"\U0010ffff"),
    ],
)
def test_printable_escapes_terminal_controls_and_unicode(value, expected):
    assert printable(value) == expected
    assert all(" " <= char <= "~" for char in printable(value))


@pytest.mark.parametrize("display", [None, ""])
def test_reason_only_notice_has_only_a_header(display):
    assert feedback_lines("check_failed", display) == [PREFIX + "check_failed"]


def test_header_cannot_inject_terminal_lines_or_escape_sequences():
    header = "task\nforged\x1b[2J\r\u202e\U0001f600"
    assert feedback_lines(header, None) == [PREFIX + printable(header)]
    long_header = feedback_lines("\x1b" * 10_000, None)
    assert len(long_header) == 1
    assert len(long_header[0]) == len(PREFIX) + FEEDBACK_PRINT_HEADER_CHARS
    assert all(" " <= char <= "~" for char in long_header[0])


@pytest.mark.parametrize("control", ["\r", "\v", "\f", "\x85", "\u2028", "\u2029"])
def test_only_lf_splits_display_lines_other_controls_remain_visible(control):
    assert feedback_lines("check_failed", f"a{control}b\nc") == [
        PREFIX + "check_failed",
        DISPLAY_PREFIX + "a" + printable(control) + "b",
        DISPLAY_PREFIX + "c",
    ]


def test_largest_legal_ascii_failed_check_keeps_the_entire_expected_value():
    template = (
        'Command (argv): ["/usr/bin/python3","main.py"]\n'
        'Working directory: "/work"\n'
        'Stdin: ""\n'
        "Required exit code: 0\n"
        'Required stdout: "{}"\n'
        "Required stderr: not checked"
    )
    overhead = len(json.dumps(template.format(""), ensure_ascii=True).encode("ascii"))
    expected = "x" * (FAILED_CHECK_MAX_BYTES - overhead)
    display = template.format(expected)
    explanation = FailureExplanation(
        version=1, reason_code=MinerReason.CHECK_FAILED, failed_check=display
    )
    assert len(expected) > 200
    assert (
        len(json.dumps(display, ensure_ascii=True).encode("ascii"))
        == FAILED_CHECK_MAX_BYTES
    )
    lines = feedback_lines("check_failed", explanation.failed_check)
    assert lines == [PREFIX + "check_failed"] + [
        DISPLAY_PREFIX + line for line in display.split("\n")
    ]
    assert lines[-2] == DISPLAY_PREFIX + f'Required stdout: "{expected}"'


def test_display_line_limit_includes_header_and_preserves_empty_lines():
    display = "\n".join(["check"] * (FEEDBACK_PRINT_MAX_LINES - 2) + [""])
    lines = feedback_lines("check_failed", display)
    assert len(lines) == FEEDBACK_PRINT_MAX_LINES
    assert lines[-1] == DISPLAY_PREFIX
    omitted = feedback_lines("check_failed", display + "\none more")
    assert omitted == [
        PREFIX + "check_failed",
        DISPLAY_PREFIX + "display omitted: larger than this miner prints",
    ]


@pytest.mark.parametrize("value", ["x", "\x1b", "\U0001f600"])
def test_display_character_limit_uses_escaped_length_and_never_slices_a_test(value):
    escaped_size = len(printable(value))
    fitting = value * (FEEDBACK_PRINT_MAX_CHARS // escaped_size)
    assert feedback_lines("check_failed", fitting) == [
        PREFIX + "check_failed",
        DISPLAY_PREFIX + printable(fitting),
    ]
    assert feedback_lines("check_failed", fitting + value) == [
        PREFIX + "check_failed",
        DISPLAY_PREFIX + "display omitted: larger than this miner prints",
    ]


def notice_body(**updates):
    fields = {
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
    }
    fields.update(updates)
    return serialize_failure_notice(MinerFailureNotice(**fields))


def remember_notice(miner, body, signer="validator"):
    notice = MinerFailureNotice.model_validate_json(body)
    miner.served_tasks.add(
        signer,
        notice.challenge_id,
        notice.task_id,
        notice.uid,
        notice.hotkey,
        time.monotonic(),
    )


def make_receiver(**settings_updates):
    class UnusedProvider:
        calls = 0

        async def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("notices must not invoke the model")

    return DemoMiner(
        DemoMinerSettings(_env_file=None, bedrock_api_key="key", **settings_updates),
        UnusedProvider(),
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner")),
        metagraph=SimpleNamespace(
            hotkeys=["validator", "other-validator"],
            validator_permit=[True, True],
            S=[20.0, 20.0],
        ),
    )


@pytest.fixture
def receiver(monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)
    miner = make_receiver()
    output = []
    monkeypatch.setattr(demo_miner, "print_feedback", output.append)
    return miner, output


async def deliver_notice(miner, body=None, *, signer="validator", signed_for="miner"):
    body = body if body is not None else notice_body()
    headers = protocol.sign_message(signer, body, signed_for=signed_for)
    return await miner.handle_failure_notice(headers, body)


@pytest.mark.parametrize("missing", ["wallet", "hotkey", "metagraph"])
async def test_notice_receiver_fails_closed_without_identity_or_metagraph(
    receiver, missing
):
    miner, output = receiver
    body = notice_body()
    remember_notice(miner, body)
    if missing == "hotkey":
        miner.wallet.hotkey.ss58_address = ""
    else:
        setattr(miner, missing, None)
    status, _ = await deliver_notice(miner, body)
    assert status == 403
    assert output == []


@pytest.mark.parametrize(
    ("settings_updates", "permits", "stakes", "expected_status"),
    [
        ({}, [False, True], [20, 20], 403),
        ({}, None, [20, 20], 403),
        ({"miner_require_validator_permit": False}, [False, True], [20, 20], 200),
        ({"miner_min_stake": 10}, [True, True], [9, 20], 403),
        ({"miner_min_stake": 10}, [True, True], None, 403),
        ({"miner_min_stake": 10}, [True, True], [10, 20], 200),
    ],
)
async def test_notice_receiver_enforces_configured_validator_policy(
    monkeypatch, settings_updates, permits, stakes, expected_status
):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)
    output = []
    monkeypatch.setattr(demo_miner, "print_feedback", output.append)
    miner = make_receiver(**settings_updates)
    miner.metagraph.validator_permit = permits
    miner.metagraph.S = stakes
    body = notice_body()
    remember_notice(miner, body)
    status, _ = await deliver_notice(miner, body)
    assert status == expected_status
    assert bool(output) is (expected_status == 200)


@pytest.mark.parametrize(
    "invalid",
    [
        "unsigned",
        "tampered",
        "wrong-recipient",
        "missing-recipient",
        "stale",
        "forged-signer",
    ],
)
async def test_notice_receiver_checks_signature_body_freshness_and_recipient(
    receiver, monkeypatch, invalid
):
    miner, output = receiver
    body = notice_body()
    remember_notice(miner, body)
    if invalid == "stale":
        real_now = time.time()
        with monkeypatch.context() as clock:
            clock.setattr(protocol.time, "time", lambda: real_now - 10)
            headers = protocol.sign_message("validator", body, signed_for="miner")
    else:
        recipient = "other" if invalid == "wrong-recipient" else "miner"
        if invalid == "missing-recipient":
            recipient = ""
        headers = protocol.sign_message("validator", body, signed_for=recipient)
    if invalid == "unsigned":
        headers = {}
    elif invalid == "tampered":
        body = body.replace(b"red-fox", b"bad-fox")
    elif invalid == "forged-signer":
        headers["Epistula-Signed-By"] = "other-validator"
    status, _ = await miner.handle_failure_notice(headers, body)
    assert status == 401
    assert output == []


@pytest.mark.parametrize(
    ("updates", "signer", "expected_status"),
    [
        ({"challenge_id": "other"}, "validator", 404),
        ({"task_id": "b" * 64}, "validator", 404),
        ({"uid": 8}, "validator", 404),
        ({"hotkey": "other-miner"}, "validator", 403),
        ({}, "other-validator", 404),
        ({}, "unregistered-validator", 403),
    ],
)
async def test_notice_receiver_requires_exact_served_task_and_registration(
    receiver, updates, signer, expected_status
):
    miner, output = receiver
    remember_notice(miner, notice_body())
    status, _ = await deliver_notice(miner, notice_body(**updates), signer=signer)
    assert status == expected_status
    assert output == []
    assert (await deliver_notice(miner))[0] == 200


async def test_fresh_duplicate_notice_is_acknowledged_but_printed_once(receiver):
    miner, output = receiver
    body = notice_body()
    remember_notice(miner, body)
    assert await deliver_notice(miner, body) == (200, {"accepted": True})
    assert await deliver_notice(miner, body) == (200, {"accepted": True})
    assert len(output) == 1
    assert 'Required stdout: "red-fox\\n"' in "\n".join(output[0])


@pytest.mark.parametrize("first_endpoint", ["solve", "notice"])
async def test_notice_and_solve_share_the_nonce_cache(receiver, first_endpoint):
    miner, output = receiver
    body = notice_body()
    remember_notice(miner, body)
    headers = protocol.sign_message("validator", body, signed_for="miner")
    if first_endpoint == "solve":
        assert (await miner.handle_request(headers, body))[0] == 400
        assert (await miner.handle_failure_notice(headers, body))[0] == 409
        assert not output
    else:
        assert (await miner.handle_failure_notice(headers, body))[0] == 200
        assert (await miner.handle_request(headers, body))[0] == 409
        assert (await miner.handle_failure_notice(headers, body))[0] == 409
        assert len(output) == 1


async def test_notice_does_not_take_solve_slots_or_call_provider(receiver):
    miner, output = receiver
    miner.solve_slots = asyncio.Semaphore(0)
    remember_notice(miner, notice_body())
    assert await asyncio.wait_for(deliver_notice(miner), timeout=1) == (
        200,
        {"accepted": True},
    )
    assert len(output) == 1
    assert miner.client.calls == 0
    assert miner.solve_slots.locked()


@pytest.mark.parametrize("error", [BrokenPipeError, OSError, ValueError, RuntimeError])
async def test_terminal_output_errors_do_not_escape_notice_handling(monkeypatch, error):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)
    miner = make_receiver()
    body = notice_body()
    remember_notice(miner, body)

    def broken_output(*_args, **_kwargs):
        raise error("closed output")

    with monkeypatch.context() as output_patch:
        output_patch.setattr("builtins.print", broken_output)
        result = await deliver_notice(miner, body)
    assert result == (200, {"accepted": True})
    assert miner.client.calls == 0


@pytest.mark.parametrize("chunked", [False, True])
async def test_failure_route_rejects_body_over_8192_before_handler(
    receiver, monkeypatch, chunked
):
    miner, output = receiver
    calls = []

    async def handler(*args):
        calls.append(args)
        return 200, {"accepted": True}

    monkeypatch.setattr(miner, "handle_failure_notice", handler)
    body = b" " * (FAILURE_NOTICE_MAX_BYTES + 1)

    async def chunks():
        yield body[:4096]
        yield body[4096:]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_demo_miner_app(miner)),
        base_url="http://miner",
    ) as http:
        response = await http.post("/v3/failure", content=chunks() if chunked else body)
    assert response.status_code == 413
    assert calls == []
    assert output == []


async def test_failure_route_accepts_exact_8192_with_valid_signature(receiver):
    miner, output = receiver
    body = notice_body()
    remember_notice(miner, body)
    padded_body = body + b" " * (FAILURE_NOTICE_MAX_BYTES - len(body))
    headers = protocol.sign_message("validator", padded_body, signed_for="miner")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_demo_miner_app(miner)),
        base_url="http://miner",
    ) as http:
        response = await http.post("/v3/failure", content=padded_body, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert len(output) == 1


@pytest.mark.parametrize("chunked", [False, True])
async def test_failure_route_honors_a_smaller_configured_body_limit(
    receiver, monkeypatch, chunked
):
    miner, output = receiver
    miner.settings.miner_max_request_bytes = 128
    calls = []

    async def handler(*args):
        calls.append(args)
        return 200, {"accepted": True}

    monkeypatch.setattr(miner, "handle_failure_notice", handler)

    async def chunks():
        yield b"x" * 100
        yield b"y" * 29

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_demo_miner_app(miner)),
        base_url="http://miner",
    ) as http:
        response = await http.post(
            "/v3/failure", content=chunks() if chunked else b"x" * 129
        )
    assert response.status_code == 413
    assert calls == output == []


@pytest.mark.parametrize(
    "updates",
    [
        {"message_type": "solve_v3"},
        {"message_type": None},
        {"protocol_version": "3"},
        {"uid": True},
        {"actual_stdout": "untrusted output"},
        {"failure": {"version": 1, "reason_code": "invented"}},
        {
            "failure": {
                "version": 1,
                "reason_code": "evaluation_failed",
                "failed_check": "not allowed",
            }
        },
        {
            "failure": {
                "version": 1,
                "reason_code": "check_failed",
                "failed_check": "x" * 2048,
            }
        },
    ],
)
async def test_notice_rejects_invalid_wire_without_consuming_task_eligibility(
    receiver, updates
):
    miner, output = receiver
    body = notice_body()
    remember_notice(miner, body)
    malformed = json.loads(body)
    malformed.update(updates)
    assert (await deliver_notice(miner, json.dumps(malformed).encode()))[0] == 400
    assert output == []
    assert (await deliver_notice(miner, body))[0] == 200
    assert len(output) == 1


async def test_handler_itself_rejects_oversized_notice(receiver):
    miner, output = receiver
    body = notice_body()
    remember_notice(miner, body)
    body += b" " * (FAILURE_NOTICE_MAX_BYTES + 1 - len(body))
    assert (await deliver_notice(miner, body))[0] == 413
    assert output == []


@pytest.mark.skipif(
    not protocol.crypto_available(), reason="no sr25519 crypto stack installed"
)
async def test_failure_route_real_signature_roundtrip_and_tamper(monkeypatch):
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner_key = Keypair.create_from_uri("//Bob")
    miner = make_receiver()
    miner.wallet = SimpleNamespace(hotkey=miner_key)
    miner.metagraph.hotkeys = [validator.ss58_address]
    output = []
    monkeypatch.setattr(demo_miner, "print_feedback", output.append)
    body = notice_body(hotkey=miner_key.ss58_address)
    remember_notice(miner, body, signer=validator.ss58_address)
    headers = protocol.sign_message(
        SimpleNamespace(hotkey=validator), body, signed_for=miner_key.ss58_address
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_demo_miner_app(miner)),
        base_url="http://miner",
    ) as http:
        bad = await http.post(
            "/v3/failure", content=body.replace(b"red-fox", b"bad-fox"), headers=headers
        )
        good = await http.post("/v3/failure", content=body, headers=headers)
    assert bad.status_code == 401
    assert good.status_code == 200
    assert good.json() == {"accepted": True}
    assert len(output) == 1


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_blocked_stdout_has_bounded_admission_even_after_request_cancellation(
    receiver, monkeypatch, cancel_first
):
    miner, _output = receiver
    release = threading.Event()
    all_started = threading.Event()
    lock = threading.Lock()
    started = 0
    active = 0
    maximum_active = 0

    def blocked_output(_lines):
        nonlocal started, active, maximum_active
        with lock:
            started += 1
            active += 1
            maximum_active = max(maximum_active, active)
            if started >= 4:
                all_started.set()
        try:
            release.wait(timeout=5)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(demo_miner, "print_feedback", blocked_output)
    bodies = [notice_body(challenge_id=f"challenge-{index}") for index in range(12)]
    for body in bodies:
        remember_notice(miner, body)
    tasks = [asyncio.create_task(deliver_notice(miner, body)) for body in bodies[:4]]
    excess = []
    try:

        async def wait_until_started():
            while not all_started.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_until_started(), timeout=1)
        if cancel_first:
            tasks[0].cancel()
            await asyncio.sleep(0)
        excess = [
            asyncio.create_task(deliver_notice(miner, body)) for body in bodies[4:]
        ]
        finished, waiting = await asyncio.wait(excess, timeout=0.25)
        assert not waiting, (
            "busy notices must be rejected immediately, not queue behind blocked stdout"
        )
        assert all(task.result()[0] in {429, 503} for task in finished)
        assert maximum_active == 4
        assert miner.client.calls == 0
    finally:
        release.set()
        for pending in excess:
            if not pending.done():
                pending.cancel()
        await asyncio.gather(*tasks, *excess, return_exceptions=True)

        async def wait_until_idle():
            while active:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_until_idle(), timeout=1)
    assert (await deliver_notice(miner, bodies[4]))[0] == 200
    assert started == 5, (
        "a refused notice must remain eligible after output capacity returns"
    )


@pytest.mark.parametrize("solve_succeeds", [True, False])
async def test_served_context_is_created_by_successful_solve_response(
    receiver, monkeypatch, solve_succeeds
):
    from tests.test_demo_miner import task

    miner, output = receiver
    request = task(hotkey="miner")
    body = notice_body(challenge_id=request.challenge_id, task_id=request.task_id)

    async def solve(received, _timeout):
        assert received == request
        if not solve_succeeds:
            raise TimeoutError("model deadline exceeded")
        return MinerTaskResponse(
            protocol_version=3,
            challenge_id=received.challenge_id,
            task_id=received.task_id,
            response_type="repository_patch_v1",
            submission={
                "artifact_role": "patch", "artifact_format": "unified_diff_v1",
                "upload_id": received.slots.submission.upload_id,
                "sha256": "c" * 64, "size_bytes": 0,
            },
            trajectory={
                "artifact_role": "trajectory", "artifact_format": "trajectory_v1",
                "upload_id": received.slots.trajectory.upload_id,
                "sha256": "d" * 64, "size_bytes": 1,
            },
        )

    monkeypatch.setattr(miner, "solve", solve)
    assert (await deliver_notice(miner, body))[0] == 404
    signed_task = request.model_dump_json().encode()
    headers = protocol.sign_message("validator", signed_task, signed_for="miner")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_demo_miner_app(miner)),
        base_url="http://miner",
    ) as http:
        response = await http.post("/solve", content=signed_task, headers=headers)
    assert response.status_code == (200 if solve_succeeds else 504)
    assert (await deliver_notice(miner, body))[0] == (200 if solve_succeeds else 404)
    assert len(output) == int(solve_succeeds)
