from __future__ import annotations

import httpx

from .api import MinerTaskRequest, MinerTaskResponse
from .upload import upload_artifact


async def upload_miner_result(
    http: httpx.AsyncClient,
    task: MinerTaskRequest,
    submission: bytes,
    trajectory: bytes,
    *,
    allowed_origins: frozenset[str],
) -> MinerTaskResponse:
    if type(task) is not MinerTaskRequest:
        raise TypeError("task must be a validated V3 miner request")
    if type(submission) is not bytes or type(trajectory) is not bytes:
        raise TypeError("miner artifacts must be bytes")
    submission_ref = await upload_artifact(
        http,
        task.slots.submission,
        submission,
        allowed_origins=allowed_origins,
    )
    trajectory_ref = await upload_artifact(
        http,
        task.slots.trajectory,
        trajectory,
        allowed_origins=allowed_origins,
    )
    return MinerTaskResponse(
        protocol_version=3,
        challenge_id=task.challenge_id,
        task_id=task.task_id,
        response_type=task.identity.task_type,
        submission=submission_ref,
        trajectory=trajectory_ref,
    )
