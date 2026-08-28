from __future__ import annotations

import asyncio

import httpx

from rlvr.problemserver.client import LeaseCategory
from rlvr.v3.api import (
    ChallengeCommitRequest,
    ChallengeFeedbackRequest,
    ChallengeFeedbackResponse,
    CommitRevealResponse,
    LeaseResponse,
)
from rlvr.v3.client import V3ProblemServerClient
from tests.test_v3_api import lease, reveal, submission


def run(handler, action):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = V3ProblemServerClient(
                "https://problems.invalid", "validator", http, retries=1
            )
            return await action(client)

    return asyncio.run(go())


def test_v3_client_uses_v3_lease_route_and_parses_strict_response():
    seen = {}

    async def handler(request):
        seen["path"] = request.url.path
        seen["body"] = await request.aread()
        return httpx.Response(200, content=LeaseResponse(**lease()).model_dump_json())

    outcome = run(handler, lambda client: client.lease())
    assert outcome.category is LeaseCategory.LEASED
    assert outcome.challenge is not None
    assert seen["path"] == "/v3/challenges/lease"
    assert b'"request_id"' in seen["body"]


def test_v3_client_reports_pacing_without_collapsing_it():
    handler = lambda request: httpx.Response(429, headers={"Retry-After": "9"})
    outcome = run(handler, lambda client: client.lease())
    assert outcome.category is LeaseCategory.PACED and outcome.retry_after_s == 9


def test_v3_commit_reuses_one_exact_serialized_body_across_retry():
    bodies = []

    async def handler(request):
        bodies.append(await request.aread())
        if len(bodies) == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=CommitRevealResponse(**reveal()).model_dump_json())

    request = ChallengeCommitRequest(protocol_version=3, challenge_id="chal-1", submissions=[submission()])

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = V3ProblemServerClient(
                "https://problems.invalid", "validator", http, retries=2
            )
            return await client.commit(request)

    result = asyncio.run(go())
    assert result is not None
    assert len(bodies) == 2 and bodies[0] == bodies[1]


def test_v3_commit_rejects_non_200_or_malformed_response():
    request = ChallengeCommitRequest(protocol_version=3, challenge_id="chal-1", submissions=[])
    assert run(lambda r: httpx.Response(409), lambda client: client.commit(request)) is None
    assert run(lambda r: httpx.Response(200, content=b"{}"), lambda client: client.commit(request)) is None


def test_v3_feedback_retries_identical_bytes_and_checks_response_binding():
    request = ChallengeFeedbackRequest(
        protocol_version=3,
        challenge_id="chal-1",
        task_id="a" * 64,
        verdicts=[
            {"uid": 7, "hotkey": "hk-7", "passed": True, "grading_duration_ms": 42}
        ],
    )
    bodies = []

    async def handler(http_request):
        bodies.append(await http_request.aread())
        if len(bodies) == 1:
            return httpx.Response(503)
        response = ChallengeFeedbackResponse(
            protocol_version=3,
            challenge_id=request.challenge_id,
            task_id=request.task_id,
        )
        return httpx.Response(200, content=response.model_dump_json())

    async def retry_feedback():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = V3ProblemServerClient(
                "https://problems.invalid", "validator", http, retries=2
            )
            return await client.feedback(request)

    assert asyncio.run(retry_feedback()) is True
    assert len(bodies) == 2 and bodies[0] == bodies[1]

    wrong = ChallengeFeedbackResponse(protocol_version=3, challenge_id="other", task_id=request.task_id)
    assert run(
        lambda _request: httpx.Response(200, content=wrong.model_dump_json()),
        lambda client: client.feedback(request),
    ) is False
