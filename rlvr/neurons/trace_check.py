"""Send a sample of trajectory references to the evaluation service.

Best effort, like miner notices: a small fraction of graded submissions each
round, one signed ticket each, no retries, bounded time, every error swallowed.
Nothing here reads a result back or can change a grade, a score or a weight.
"""

from __future__ import annotations

import asyncio
import secrets

import httpx

from ..protocol import sign_message
from ..v3.api import TraceCheckRequest, serialize_trace_check_request

TRACE_CHECK_CONCURRENCY = 4
TRACE_CHECK_TIMEOUT_S = 2.0
TRACE_CHECK_BATCH_DEADLINE_S = 5.0
_DRAW_SCALE = 1_000_000


def selected(rate: float) -> bool:
    """One unpredictable draw per submission with probability ``rate``."""

    return secrets.randbelow(_DRAW_SCALE) < int(rate * _DRAW_SCALE)


async def send_trace_checks(
    result,
    *,
    wallet,
    http: httpx.AsyncClient,
    url: str,
    service_hotkey: str,
    rate: float,
) -> int:
    """Send a ticket for each sampled submission. Returns how many were accepted."""

    if not url.startswith("https://") or not service_hotkey or rate <= 0:
        return 0
    if getattr(result, "status", None) != "completed":
        return 0
    challenge_id = getattr(result, "challenge_id", None)
    task_id = getattr(result, "task_id", None)
    if not challenge_id or not task_id:
        return 0

    tickets = []
    for item in result.evaluations:
        # A reference exists only for submissions the server accepted at commit;
        # a patch or script rejected during grading still carries one.
        if item.trajectory is None:
            continue
        if not selected(rate):
            continue
        try:
            tickets.append(
                TraceCheckRequest(
                    protocol_version=3,
                    message_type="trace_check_v1",
                    challenge_id=challenge_id,
                    task_id=task_id,
                    uid=item.uid,
                    hotkey=item.hotkey,
                    trajectory=item.trajectory,
                    submission_status=item.result.status,
                )
            )
        except Exception:  # noqa: BLE001, S112 - an unusable reference is skipped
            continue
    if not tickets:
        return 0

    limit = asyncio.Semaphore(TRACE_CHECK_CONCURRENCY)
    accepted = 0

    async def deliver(ticket: TraceCheckRequest) -> None:
        nonlocal accepted
        body = serialize_trace_check_request(ticket)
        headers = sign_message(wallet, body, signed_for=service_hotkey)
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
        async with http.stream(
            "POST",
            url,
            content=body,
            headers=headers,
            timeout=TRACE_CHECK_TIMEOUT_S,
            follow_redirects=False,
        ) as response:
            # The body is never read: the reply is an acknowledgement, not a result.
            if response.status_code == 200:
                accepted += 1

    async def send_one(ticket: TraceCheckRequest) -> None:
        async with limit:
            try:
                await asyncio.wait_for(deliver(ticket), TRACE_CHECK_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - one failed ticket costs nothing
                return

    async def send_all() -> None:
        await asyncio.gather(*(send_one(ticket) for ticket in tickets), return_exceptions=True)

    try:
        await asyncio.wait_for(send_all(), timeout=TRACE_CHECK_BATCH_DEADLINE_S)
    except Exception:  # noqa: BLE001, S110 - the round is already finished
        pass
    if accepted < len(tickets):
        # One line per batch, so a broken service is visible without spam.
        print(f"[validator] WARN: trace checks: {accepted} of {len(tickets)} tickets acknowledged")
    return accepted
