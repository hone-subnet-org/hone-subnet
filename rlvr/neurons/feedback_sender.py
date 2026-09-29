"""Tell each failed miner why it failed, straight after a graded round.

Best effort by design: one signed request per failed miner, no retries, bounded
concurrency and a deadline for the whole batch. Nothing here can change a grade,
a score or a weight, and every failure is swallowed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from ..protocol import sign_message
from ..v3.api import FailureExplanation, MinerFailureNotice, serialize_failure_notice
from ..v3.reasons import MinerReason, Stage
from .live import LiveSolverClient

NOTICE_PATH = "/v3/failure"
NOTICE_CONCURRENCY = 8
NOTICE_TIMEOUT_S = 2.0
NOTICE_BATCH_DEADLINE_S = 5.0
NOTICE_MAX_RECIPIENTS = 1_024


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
) -> MinerFailureNotice | None:
    """Build one notice, or None when this outcome does not earn one."""

    if evaluation_status not in ("failed", "rejected"):
        return None
    code = reason_code if type(reason_code) is MinerReason else "evaluation_failed"
    display = (
        failed_check
        if include_details
        and code == MinerReason.CHECK_FAILED
        and stage == Stage.CHECK
        else None
    )
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
                    version=1, reason_code=code, failed_check=attempt
                ),
            )
        except Exception:  # noqa: BLE001, S112 - fall back to the reason, then give up
            continue
    return None


async def send_failure_notices(
    result,
    solvers: Sequence[LiveSolverClient],
    *,
    wallet,
    http: httpx.AsyncClient,
    include_details: bool,
) -> int:
    """Send one notice per failed miner. Returns how many were accepted.

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
    jobs = []
    seen: set[tuple[int, str]] = set()
    for item in result.evaluations:
        registration = (item.uid, item.hotkey)
        if registration in seen or registration not in assigned:
            continue
        client = clients.get(registration)
        if client is None:
            continue
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
        )
        if notice is None:
            continue
        seen.add(registration)
        jobs.append((client, notice))
        if len(jobs) >= NOTICE_MAX_RECIPIENTS:
            break
    if not jobs:
        return 0

    limit = asyncio.Semaphore(NOTICE_CONCURRENCY)
    delivered = 0

    async def deliver(client: LiveSolverClient, notice: MinerFailureNotice) -> None:
        nonlocal delivered
        permit = await client.gate.acquire()
        try:
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
                if response.status_code == 200:
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
