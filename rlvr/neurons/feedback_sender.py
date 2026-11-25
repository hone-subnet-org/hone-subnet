"""Tell each graded miner how it did, straight after a graded round.

A miner that passed hears `passed`; one that did not hears the reason. Best
effort by design: one signed request per graded miner, no retries, bounded
concurrency and a deadline for the whole batch. Nothing here can change a grade,
a score or a weight, and every failure is swallowed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from ..protocol import sign_message
from ..v3.api import (
    DISPATCH_DETAIL_MAX_CHARS,
    FailureExplanation,
    MinerFailureNotice,
    serialize_failure_notice,
)
from ..v3.reasons import MinerReason, Stage
from .live import LiveSolverClient

NOTICE_PATH = "/v3/failure"
NOTICE_CONCURRENCY = 32
NOTICE_TIMEOUT_S = 2.0
NOTICE_BATCH_DEADLINE_S = 15.0
NOTICE_MAX_RECIPIENTS = 1_024


def short_detail(value: str | None) -> str | None:
    """What a miner may be told: one printable line of at most DISPATCH_DETAIL_MAX_CHARS, or None."""
    if not value:
        return None
    if value.startswith("HTTP "):
        # The body of an HTTP error is unsigned text from whatever answered,
        # which may not have been the miner; it gets the status alone.
        value = value.split(":", 1)[0]
    clean = "".join(ch for ch in " ".join(value.split()) if ch.isprintable())
    return clean[:DISPATCH_DETAIL_MAX_CHARS] or None


def build_notice(
    challenge_id: str,
    task_id: str,
    uid: int,
    hotkey: str,
    evaluation_status: str,
    reason_code: object,
    stage: object,
    failed_check: str | None,
    *,
    include_details: bool,
    dispatch_reason_code: object = None,
    dispatch_detail: str | None = None,
) -> MinerFailureNotice | None:
    """Build one notice, or None when this outcome does not earn one."""

    if evaluation_status == "passed":
        code = "passed"
    elif evaluation_status in ("failed", "rejected"):
        code = reason_code if type(reason_code) is MinerReason else "evaluation_failed"
    else:
        return None
    display = (
        failed_check
        if include_details
        and code == MinerReason.CHECK_FAILED
        and stage == Stage.CHECK
        else None
    )
    # What the validator saw locally when it got no usable response. Only the
    # solver's own description of this miner's response is forwarded, which
    # may be "validator error: <class>" so the miner knows it was not at
    # fault; a solver that raised stays in the local record, code alone out.
    local_code = (
        dispatch_reason_code
        if code != "passed" and type(dispatch_reason_code) is MinerReason
        else None
    )
    local_detail = (
        short_detail(dispatch_detail)
        if local_code is MinerReason.RESPONSE_UNAVAILABLE
        else None
    )
    if local_detail is not None:
        try:
            FailureExplanation(
                version=1, reason_code=code, dispatch_reason_code=local_code, dispatch_detail=local_detail
            )
        except Exception:  # noqa: BLE001 - a detail the wire refuses is left out; the code still goes
            local_detail = None
    for attempt in ((display, None) if display is not None else (None,)):
        try:
            return MinerFailureNotice(
                protocol_version=3,
                message_type="failure_notice_v1",
                challenge_id=challenge_id,
                task_id=task_id,
                uid=uid,
                hotkey=hotkey,
                failure=FailureExplanation(
                    version=1,
                    reason_code=code,
                    failed_check=attempt,
                    dispatch_reason_code=local_code,
                    dispatch_detail=local_detail,
                ),
            )
        except Exception:  # noqa: BLE001, S112 - fall back to the reason, then give up
            continue
    return None


def reason_only(notice: MinerFailureNotice) -> MinerFailureNotice:
    return notice.model_copy(
        update={"failure": FailureExplanation(version=1, reason_code=notice.failure.reason_code)}
    )


def has_extras(notice: MinerFailureNotice) -> bool:
    """True when the notice carries anything beyond the reason code."""
    failure = notice.failure
    return (
        failure.failed_check is not None
        or failure.dispatch_reason_code is not None
        or failure.dispatch_detail is not None
    )


async def send_failure_notices(
    result,
    solvers: Sequence[LiveSolverClient],
    *,
    wallet,
    http: httpx.AsyncClient,
    include_details: bool,
) -> int:
    """Send one notice per graded miner. Returns how many were accepted.

    Only a completed round, only the miners it graded, and only to the exact
    registration that was dispatched.
    """

    if getattr(result, "status", None) != "completed":
        return 0
    challenge_id = getattr(result, "challenge_id", None)
    task_id = getattr(result, "task_id", None)
    if not challenge_id or not task_id:
        return 0

    clients = {(client.uid, client.hotkey): client for client in solvers}
    assigned = set(getattr(result, "assigned_miners", ()) or ())
    failures = {
        (item.uid, item.hotkey): item
        for item in getattr(result, "dispatch_failures", ()) or ()
    }
    jobs = []
    seen: set[tuple[int, str]] = set()
    for item in result.evaluations:
        registration = (item.uid, item.hotkey)
        if registration in seen or registration not in assigned:
            continue
        client = clients.get(registration)
        if client is None:
            continue
        failure = failures.get(registration)
        notice = build_notice(
            challenge_id,
            task_id,
            item.uid,
            item.hotkey,
            item.result.status,
            item.result.reason_code,
            getattr(item.result, "stage", None),
            getattr(item.result, "failed_check", None),
            include_details=include_details,
            dispatch_reason_code=None if failure is None else failure.code,
            dispatch_detail=None if failure is None else failure.detail,
        )
        if notice is None:
            continue
        seen.add(registration)
        jobs.append((client, notice))
    if not jobs:
        return 0
    # Failures first: when the batch runs out of room or time, it is passes that
    # go unsent.
    jobs.sort(key=lambda job: job[1].failure.reason_code == "passed")
    del jobs[NOTICE_MAX_RECIPIENTS:]

    limit = asyncio.Semaphore(NOTICE_CONCURRENCY)
    delivered = 0

    async def post(client: LiveSolverClient, notice: MinerFailureNotice) -> int:
        # Sign after the gate, so the signature is fresh when it goes out.
        body = serialize_failure_notice(notice)
        headers = sign_message(wallet, body, signed_for=client.hotkey)
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
        async with http.stream(
            "POST",
            f"{client.url}{NOTICE_PATH}",
            content=body,
            headers=headers,
            timeout=NOTICE_TIMEOUT_S,
            follow_redirects=False,
        ) as response:
            # The body is never read: the ack is an observation, not evidence.
            return response.status_code

    async def deliver(client: LiveSolverClient, notice: MinerFailureNotice) -> None:
        nonlocal delivered
        permit = await client.gate.acquire()
        try:
            status = await post(client, notice)
            if status in (400, 413) and has_extras(notice):
                # A miner on an older release refuses a display larger than its
                # own cap, or fields it does not know. It still gets the reason, once.
                status = await post(client, reason_only(notice))
            if status == 200:
                delivered += 1
        finally:
            permit.release()

    async def send_one(client: LiveSolverClient, notice: MinerFailureNotice) -> None:
        async with limit:
            try:
                # httpx timeouts are per I/O operation, so bound the whole exchange.
                await asyncio.wait_for(deliver(client, notice), NOTICE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - one unreachable miner costs nothing
                return

    async def send_all() -> None:
        await asyncio.gather(
            *(send_one(client, notice) for client, notice in jobs),
            return_exceptions=True,
        )

    try:
        await asyncio.wait_for(send_all(), timeout=NOTICE_BATCH_DEADLINE_S)
    except Exception:  # noqa: BLE001, S110 - feedback never disturbs the round
        pass
    return delivered
