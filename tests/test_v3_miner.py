from __future__ import annotations

import asyncio

import httpx

from rlvr.v3.api import MinerTaskRequest
from rlvr.v3.miner import upload_miner_result
from tests.test_v3_api import lease, slot_set


def test_miner_uploads_submission_and_trajectory_to_assigned_slots():
    leased = lease()
    task = MinerTaskRequest(
        protocol_version=3,
        challenge_id=leased["challenge_id"],
        task_id=leased["task_id"],
        identity=leased["identity"],
        workspace=leased["workspace"],
        workspace_url=leased["workspace_url"],
        expires_at=leased["expires_at"],
        slots=slot_set(),
    )
    seen = []

    async def handler(request):
        seen.append((request.url.path, await request.aread(), request.headers))
        return httpx.Response(200)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await upload_miner_result(
                http,
                task,
                b"",
                b'{"events":[]}',
                allowed_origins=frozenset({"https://uploads.invalid:443"}),
            )

    response = asyncio.run(go())
    assert [item[1] for item in seen] == [b"", b'{"events":[]}']
    assert all(item[2]["if-none-match"] == "*" for item in seen)
    assert response.submission.size_bytes == 0
    assert response.trajectory.size_bytes > 0
