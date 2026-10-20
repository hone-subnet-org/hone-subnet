"""The validator's HTTP connections send TCP keepalive probes while a miner works.

A /solve request is silent for the whole solve window. Networks that drop idle
TCP connections after ~10 minutes made one validator lose every slow answer,
so its pool enables keepalive and probes once a minute. The probes are there
to keep middleboxes from expiring the connection; unanswered probes cannot run
out before the solve deadline does.
"""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from rlvr.config import SOLVE_DEADLINE_CEILING_S, Settings
from rlvr.neurons import decentralized
from rlvr.neurons.live import _SOLVE_DEADLINE_GRACE_S

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "TCP_KEEPIDLE"), reason="keepalive tuning uses the Linux socket options"
)


def test_unanswered_probes_outlast_the_solve_deadline():
    longest_wait_s = SOLVE_DEADLINE_CEILING_S + _SOLVE_DEADLINE_GRACE_S
    budget_s = decentralized.KEEPALIVE_IDLE_S + decentralized.KEEPALIVE_PROBES * decentralized.KEEPALIVE_INTERVAL_S
    assert budget_s > longest_wait_s


def test_transport_keeps_the_pool_size():
    settings = Settings(validator_dispatch_concurrency=16)
    transport = decentralized._validator_http_transport(settings)
    limits = decentralized._validator_http_limits(settings)
    pool = transport._pool
    assert pool._max_connections == limits.max_connections
    assert pool._max_keepalive_connections == limits.max_keepalive_connections


async def test_a_real_connection_has_keepalive_enabled():
    handlers: list[asyncio.Task] = []

    async def serve(reader, writer):
        handlers.append(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()  # until the client hangs up
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        transport = decentralized._validator_http_transport(Settings())
        async with httpx.AsyncClient(transport=transport) as http:
            async with http.stream("GET", f"http://127.0.0.1:{port}/") as response:
                sock = response.extensions["network_stream"].get_extra_info("socket")
                assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 60
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 60
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 60
                assert await response.aread() == b"ok"
    finally:
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers)
