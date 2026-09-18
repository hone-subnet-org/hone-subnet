from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable

import httpx

from .artifacts import MinerArtifactRef, UploadSlot
from .download import normalize_https_origin


class ArtifactUploadError(RuntimeError):
    pass


async def upload_artifact(
    http: httpx.AsyncClient,
    slot: UploadSlot,
    contents: bytes,
    *,
    allowed_origins: frozenset[str],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> MinerArtifactRef:
    if type(slot) is not UploadSlot or type(contents) is not bytes:
        raise TypeError("upload requires a validated slot and bytes")
    if len(contents) > slot.max_bytes:
        raise ArtifactUploadError("artifact exceeds its upload slot")
    try:
        if normalize_https_origin(slot.upload_url) not in allowed_origins:
            raise ArtifactUploadError("artifact upload origin is not allowed")
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(contents)),
            "If-None-Match": "*",
        }
        ambiguous = False
        for attempt in range(3):
            if attempt:
                await sleep(float(attempt))
            try:
                async with http.stream(
                    "PUT",
                    slot.upload_url,
                    headers=headers,
                    content=contents,
                    follow_redirects=False,
                ) as response:
                    status = response.status_code
                    if status in (200, 201):
                        break
                    if status in (409, 412) and ambiguous:
                        break
                    if 500 <= status <= 599:
                        ambiguous = True
                        if attempt < 2:
                            continue
                    raise ArtifactUploadError("artifact upload was rejected")
            except httpx.HTTPError:
                ambiguous = True
                if attempt < 2:
                    continue
                raise ArtifactUploadError("artifact upload failed") from None
    except ArtifactUploadError:
        raise
    except (httpx.HTTPError, ValueError):
        raise ArtifactUploadError("artifact upload failed") from None
    return MinerArtifactRef(
        artifact_role=slot.artifact_role,
        artifact_format=slot.artifact_format,
        upload_id=slot.upload_id,
        sha256=hashlib.sha256(contents).hexdigest(),
        size_bytes=len(contents),
    )
