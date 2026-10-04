from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote
from uuid import uuid4

from ..problemserver.client import (
    LeaseCategory,
    ProblemServerClient,
    bounded_detail,
    parse_retry_after,
    response_detail,
    status_category,
)
from .api import (
    LEDGER_PAGE_LIMIT,
    ChallengeCommitRequest,
    ChallengeFeedbackRequest,
    ChallengeFeedbackResponse,
    CommitRevealResponse,
    LeaseRequest,
    LeaseResponse,
    LedgerPage,
    MinerCandidate,
    serialize_commit_request,
    serialize_feedback_request,
)


@dataclass(frozen=True)
class LedgerFetch:
    status: int | None
    page: LedgerPage | None


@dataclass(frozen=True)
class V3LeaseOutcome:
    category: LeaseCategory
    challenge: LeaseResponse | None = None
    status: int | None = None
    detail: str = ""
    retry_after_s: int | None = None
    transport_error: str = ""


class V3ProblemServerClient:
    def __init__(self, *args, **kwargs):
        self._transport = ProblemServerClient(*args, **kwargs)

    async def lease(
        self, candidates: Sequence[tuple[int, str]], *, miners_per_task: int | None = None
    ) -> V3LeaseOutcome:
        body = LeaseRequest(
            request_id=uuid4().hex,
            candidates=[MinerCandidate(uid=uid, hotkey=hotkey) for uid, hotkey in candidates],
            miners_per_task=miners_per_task,
        ).model_dump_json(exclude_none=True).encode("utf-8")
        post = await self._transport.post_result("/v3/challenges/lease", body)
        response = post.response
        if response is None:
            error = bounded_detail(post.transport_error)
            return V3LeaseOutcome(LeaseCategory.TRANSPORT, detail=error, transport_error=error)
        status = response.status_code
        retry_after = post.retry_after_s
        if retry_after is None:
            retry_after = parse_retry_after(response)
        if post.transport_error:
            return V3LeaseOutcome(
                LeaseCategory.TRANSPORT,
                status=status,
                detail=post.protocol_error or response_detail(response),
                retry_after_s=retry_after,
                transport_error=bounded_detail(post.transport_error),
            )
        if post.protocol_error:
            category = LeaseCategory.MALFORMED if status == 200 else status_category(status)
            return V3LeaseOutcome(category, status=status, detail=post.protocol_error, retry_after_s=retry_after)
        if status != 200:
            return V3LeaseOutcome(
                status_category(status),
                status=status,
                detail=response_detail(response),
                retry_after_s=retry_after,
            )
        try:
            challenge = LeaseResponse.model_validate_json(response.content)
        except Exception as error:  # noqa: BLE001
            return V3LeaseOutcome(
                LeaseCategory.MALFORMED,
                status=status,
                detail=bounded_detail(str(error)),
                retry_after_s=retry_after,
            )
        return V3LeaseOutcome(
            LeaseCategory.LEASED,
            challenge=challenge,
            status=status,
            retry_after_s=retry_after,
        )

    async def commit(
        self, request: ChallengeCommitRequest
    ) -> Optional[CommitRevealResponse]:
        body = serialize_commit_request(request)
        response = await self._transport.post("/v3/challenges/commit", body)
        if response is None or response.status_code != 200:
            return None
        try:
            return CommitRevealResponse.model_validate_json(response.content)
        except Exception as error:  # noqa: BLE001
            print(f"[validator] WARN: invalid V3 commit response: {error}")
            return None

    async def fetch_rounds(self, *, since: str | None, limit: int = LEDGER_PAGE_LIMIT) -> LedgerFetch:
        """One page of the shared ledger. status None when nothing answered;
        page None unless the answer was a valid 200."""
        query = f"?limit={int(limit)}" + (f"&since={quote(since, safe='')}" if since else "")
        response = await self._transport.get(f"/v3/rounds{query}")
        if response is None:
            return LedgerFetch(None, None)
        if response.status_code != 200:
            return LedgerFetch(response.status_code, None)
        try:
            return LedgerFetch(200, LedgerPage.model_validate_json(response.content))
        except Exception as error:  # noqa: BLE001 - a malformed page is dropped whole
            print(f"[validator] WARN: invalid ledger page: {bounded_detail(str(error))}")
            return LedgerFetch(200, None)

    async def signed_round_status(self, body: bytes) -> int | None:
        """POST one signed round exactly as given; the HTTP status, or None
        when nothing answered. Kept rounds are resent byte for byte."""
        response = await self._transport.post("/v3/challenges/round", bytes(body))
        return None if response is None else response.status_code

    async def feedback(self, request: ChallengeFeedbackRequest) -> bool:
        body = serialize_feedback_request(request)
        response = await self._transport.post("/v3/challenges/feedback", body)
        if response is None or response.status_code != 200:
            return False
        try:
            parsed = ChallengeFeedbackResponse.model_validate_json(response.content)
        except Exception as error:  # noqa: BLE001
            print(f"[validator] WARN: invalid V3 feedback response: {error}")
            return False
        return (
            parsed.challenge_id == request.challenge_id
            and parsed.task_id == request.task_id
        )
