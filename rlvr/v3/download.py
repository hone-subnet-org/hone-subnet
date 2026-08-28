from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

import httpx

from .archive import ArchiveLimits
from .artifacts import ArtifactRef


class ArtifactDownloadError(RuntimeError):
    pass


def normalize_https_origin(url: str) -> str:
    try:
        parsed = httpx.URL(url)
        if (
            parsed.scheme != "https"
            or not parsed.raw_host
            or parsed.userinfo
            or parsed.fragment
        ):
            raise ValueError
        host = parsed.raw_host.decode("ascii").lower()
        port = parsed.port or 443
    except (UnicodeError, ValueError, httpx.InvalidURL):
        raise ArtifactDownloadError("artifact URL is invalid") from None
    return f"https://{host}:{port}"


async def download_artifact(
    http: httpx.AsyncClient,
    url: str,
    ref: ArtifactRef,
    destination: str | os.PathLike[str],
    limits: ArchiveLimits,
    *,
    allowed_origins: frozenset[str],
) -> Path:
    if ref.artifact_format != "tar_zst_v1":
        raise ArtifactDownloadError("artifact format is not supported")
    return await download_object(
        http,
        url,
        sha256=ref.sha256,
        size_bytes=ref.compressed_size_bytes,
        max_bytes=limits.max_compressed_bytes,
        destination=destination,
        allowed_origins=allowed_origins,
    )


async def download_object(
    http: httpx.AsyncClient,
    url: str,
    *,
    sha256: str,
    size_bytes: int,
    max_bytes: int,
    destination: str | os.PathLike[str],
    allowed_origins: frozenset[str],
) -> Path:
    target = Path(destination)
    temporary: Path | None = None
    try:
        if type(size_bytes) is not int or size_bytes < 0:
            raise ArtifactDownloadError("artifact size is invalid")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ArtifactDownloadError("artifact download limit is invalid")
        if size_bytes > max_bytes:
            raise ArtifactDownloadError("artifact exceeds the download limit")
        if normalize_https_origin(url) not in allowed_origins:
            raise ArtifactDownloadError("artifact origin is not allowed")
        if target.exists():
            raise ArtifactDownloadError("artifact destination already exists")

        async with http.stream("GET", url, follow_redirects=False) as response:
            if response.status_code != 200:
                raise ArtifactDownloadError("artifact request failed")

            encoding = response.headers.get("Content-Encoding")
            if encoding is not None and encoding.strip().lower() != "identity":
                raise ArtifactDownloadError("artifact response encoding is not supported")

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                if re.fullmatch(r"[0-9]+", content_length) is None:
                    raise ArtifactDownloadError("artifact response length is invalid")
                if int(content_length) != size_bytes:
                    raise ArtifactDownloadError("artifact response length does not match")

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            digest = hashlib.sha256()
            received = 0
            with os.fdopen(descriptor, "wb") as output:
                async for chunk in response.aiter_raw(1 << 20):
                    received += len(chunk)
                    if received > size_bytes:
                        raise ArtifactDownloadError("artifact response is too large")
                    digest.update(chunk)
                    output.write(chunk)
                if received != size_bytes:
                    raise ArtifactDownloadError("artifact response is truncated")
                if digest.hexdigest() != sha256:
                    raise ArtifactDownloadError("artifact digest does not match")
                output.flush()
                os.fsync(output.fileno())

        os.replace(temporary, target)
        temporary = None
        return target
    except ArtifactDownloadError:
        raise
    except (OSError, httpx.HTTPError, ValueError):
        raise ArtifactDownloadError("artifact download failed") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
