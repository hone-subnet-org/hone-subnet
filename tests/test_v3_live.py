from __future__ import annotations

import asyncio
import time

import pytest

from rlvr.config import get_settings
from rlvr.neurons.live import LiveSolverClient
from rlvr.protocol import crypto_available, sign_message
from rlvr.v3.api import EPISTULA_HEADERS, MinerTaskRequest, MinerTaskResponse
from tests.test_v3_api import lease, slot_set

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")
pytestmark = pytest.mark.skipif(
    not crypto_available(), reason="no sr25519 crypto stack installed"
)


class WalletLike:
    def __init__(self, hotkey):
        self.hotkey = hotkey


async def test_v3_live_transport_preserves_canonical_signed_envelope():
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")
    leased = lease()
    slots = slot_set(uid=7, hotkey=miner.ss58_address)
    task = MinerTaskRequest(
        protocol_version=3,
        challenge_id=leased["challenge_id"],
        task_id=leased["task_id"],
        identity=leased["identity"],
        workspace=leased["workspace"],
        workspace_url=leased["workspace_url"],
        expires_at=2**53 - 1,
        slots=slots,
    )
    app = fastapi.FastAPI()

    @app.post("/solve")
    async def solve(request: fastapi.Request):
        received = MinerTaskRequest.model_validate_json(await request.body())
        payload = MinerTaskResponse(
            protocol_version=3,
            challenge_id=received.challenge_id,
            task_id=received.task_id,
            response_type="repository_patch_v1",
            submission={
                "artifact_role": "patch",
                "artifact_format": "unified_diff_v1",
                "upload_id": received.slots.submission.upload_id,
                "sha256": "c" * 64,
                "size_bytes": 0,
            },
            trajectory={
                "artifact_role": "trajectory",
                "artifact_format": "trajectory_v1",
                "upload_id": received.slots.trajectory.upload_id,
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
        )
        body = payload.model_dump_json().encode("utf-8")
        response = fastapi.Response(body, media_type="application/json")
        response.headers.update(
            sign_message(
                WalletLike(miner),
                body,
                signed_for=request.headers["Epistula-Signed-By"],
            )
        )
        return response

    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    )
    client = LiveSolverClient(
        uid=7,
        hotkey=miner.ss58_address,
        url="http://miner",
        wallet=WalletLike(validator),
        settings=get_settings(),
        http=http,
    )
    try:
        committed, parsed = await client.solve_v3(task)
    finally:
        await http.aclose()
    assert parsed is not None
    assert committed.error == ""
    assert set(committed.response_headers) == set(EPISTULA_HEADERS)
    assert committed.latency_ms >= 0 and type(committed.latency_ms) is int


async def test_v3_live_transport_clamps_oversized_miner_response_to_failure():
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")
    leased = lease()
    slots = slot_set(uid=7, hotkey=miner.ss58_address)
    task = MinerTaskRequest(
        protocol_version=3,
        challenge_id=leased["challenge_id"],
        task_id=leased["task_id"],
        identity=leased["identity"],
        workspace=leased["workspace"],
        workspace_url=leased["workspace_url"],
        expires_at=2**53 - 1,
        slots=slots,
    )
    app = fastapi.FastAPI()

    @app.post("/solve")
    async def solve():
        return fastapi.Response(b"x" * 20_000)

    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://miner"
    )
    client = LiveSolverClient(
        uid=7,
        hotkey=miner.ss58_address,
        url="http://miner",
        wallet=WalletLike(validator),
        settings=get_settings(),
        http=http,
    )
    try:
        committed, parsed = await client.solve_v3(task)
    finally:
        await http.aclose()
    assert parsed is None
    assert committed.response_body == "" and committed.response_headers == {}
    assert committed.error


async def test_v3_live_transport_has_a_total_response_deadline(monkeypatch):
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    miner = Keypair.create_from_uri("//Dave")
    leased = lease()
    task = MinerTaskRequest(
        protocol_version=3,
        challenge_id=leased["challenge_id"],
        task_id=leased["task_id"],
        identity=leased["identity"],
        workspace=leased["workspace"],
        workspace_url=leased["workspace_url"],
        expires_at=2**53 - 1,
        slots=slot_set(uid=7, hotkey=miner.ss58_address),
    )

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.1)
                yield b"x"

    async def handler(_request):
        return httpx.Response(200, stream=SlowBody())

    settings = get_settings().model_copy(update={"solve_deadline_s": 0.05})
    monkeypatch.setattr("rlvr.neurons.live._SOLVE_DEADLINE_GRACE_S", 0.01)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LiveSolverClient(
        uid=7,
        hotkey=miner.ss58_address,
        url="http://miner",
        wallet=WalletLike(validator),
        settings=settings,
        http=http,
    )
    started = time.monotonic()
    try:
        committed, parsed = await client.solve_v3(task)
    finally:
        await http.aclose()
    assert time.monotonic() - started < 1
    assert parsed is None
    assert "deadline" in committed.error
