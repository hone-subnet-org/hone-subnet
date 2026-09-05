from __future__ import annotations

import hashlib

import httpx
import pytest

from rlvr.v3.artifacts import UploadSlot
from rlvr.v3.upload import ArtifactUploadError, upload_artifact


ORIGIN = "https://uploads.invalid:443"
URL = "https://uploads.invalid/upload?signature=secret"


def slot(**over) -> UploadSlot:
    fields = {
        "challenge_id": "challenge-1",
        "task_id": "a" * 64,
        "uid": 7,
        "hotkey": "hotkey-7",
        "artifact_role": "patch",
        "artifact_format": "unified_diff_v1",
        "upload_id": "upload-1",
        "upload_url": URL,
        "expires_at": 1_800_000_000,
        "max_bytes": 1024,
    }
    fields.update(over)
    return UploadSlot(**fields)


@pytest.mark.asyncio
async def test_upload_sends_exact_bytes_and_returns_bound_reference():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        seen.update(method=request.method, headers=request.headers, body=body)
        return httpx.Response(200)

    contents = b"diff --git a/a b/a\n"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reference = await upload_artifact(
            http, slot(), contents, allowed_origins=frozenset({ORIGIN})
        )
    assert seen["method"] == "PUT"
    assert seen["body"] == contents
    assert seen["headers"]["content-type"] == "application/octet-stream"
    assert seen["headers"]["content-length"] == str(len(contents))
    assert seen["headers"]["if-none-match"] == "*"
    assert reference.model_dump() == {
        "artifact_role": "patch",
        "artifact_format": "unified_diff_v1",
        "upload_id": "upload-1",
        "sha256": hashlib.sha256(contents).hexdigest(),
        "size_bytes": len(contents),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 201])
async def test_success_statuses_are_accepted(status):
    transport = httpx.MockTransport(lambda request: httpx.Response(status))
    async with httpx.AsyncClient(transport=transport) as http:
        assert (
            await upload_artifact(
                http, slot(), b"", allowed_origins=frozenset({ORIGIN})
            )
        ).size_bytes == 0


@pytest.mark.asyncio
async def test_oversize_and_wrong_origin_fail_before_network():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ArtifactUploadError):
            await upload_artifact(
                http,
                slot(max_bytes=1),
                b"xx",
                allowed_origins=frozenset({ORIGIN}),
            )
        with pytest.raises(ArtifactUploadError):
            await upload_artifact(
                http, slot(), b"", allowed_origins=frozenset({"https://elsewhere:443"})
            )
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [204, 302, 307, 400, 403, 409, 412])
async def test_redirects_and_failure_statuses_are_rejected(status):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, headers={"Location": "https://elsewhere/"})
    )
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(ArtifactUploadError):
            await upload_artifact(
                http, slot(), b"x", allowed_origins=frozenset({ORIGIN})
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [500, httpx.ConnectError("lost")])
@pytest.mark.parametrize("stored", [409, 412])
async def test_ambiguous_attempt_then_already_stored_succeeds(first, stored):
    calls = 0
    bodies = []
    delays = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        bodies.append(await request.aread())
        if calls == 1:
            if isinstance(first, Exception):
                raise httpx.ConnectError("lost", request=request)
            return httpx.Response(first)
        return httpx.Response(stored)

    async def sleep(delay):
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await upload_artifact(
            http, slot(), b"same", allowed_origins=frozenset({ORIGIN}), sleep=sleep
        )
    assert result.size_bytes == 4
    assert calls == 2 and bodies == [b"same", b"same"] and delays == [1.0]


@pytest.mark.asyncio
async def test_three_ambiguous_attempts_use_fixed_backoff_and_fail():
    calls = 0
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async def sleep(delay):
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ArtifactUploadError):
            await upload_artifact(
                http, slot(), b"x", allowed_origins=frozenset({ORIGIN}), sleep=sleep
            )
    assert calls == 3 and delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_errors_never_include_the_presigned_url():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed at {request.url}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ArtifactUploadError) as caught:
            await upload_artifact(
                http, slot(), b"x", allowed_origins=frozenset({ORIGIN})
            )
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_upload_requires_exact_validated_types():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as http:
        with pytest.raises(TypeError):
            await upload_artifact(http, object(), b"", allowed_origins=frozenset())
        with pytest.raises(TypeError):
            await upload_artifact(http, slot(), bytearray(), allowed_origins=frozenset({ORIGIN}))
