from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
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
    ChallengeCommitRequest,
    ChallengeFeedbackRequest,
    ChallengeFeedbackResponse,
    CommitRevealResponse,
    LeaseRequest,
    LeaseResponse,
    serialize_commit_request,
    serialize_feedback_request,
)


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

    async def lease(self) -> V3LeaseOutcome:
        body = LeaseRequest(request_id=uuid4().hex).model_dump_json().encode("utf-8")
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
