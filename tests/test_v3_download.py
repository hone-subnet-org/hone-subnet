"""V3 artifact download to a fresh file.

Contract for module ``rlvr.v3.download``:

    ArtifactDownloadError(RuntimeError)
    normalize_https_origin(url) -> str
        "https://<idna-ascii-host>:<port>"; scheme must be https; rejects
        userinfo, fragment, empty host; port defaults to 443.
    async download_artifact(http, url, ref, destination, limits, *,
                            allowed_origins) -> Path
        Before any request: ref.artifact_format is tar_zst_v1,
        ref.compressed_size_bytes <= limits.max_compressed_bytes, the
        normalized origin is allowlisted, destination does not exist.
        GET without following redirects; status must be exactly 200; any
        Content-Encoding other than identity is rejected; Content-Length,
        when present, must be ASCII digits equal to
        ref.compressed_size_bytes. Raw bytes stream in bounded chunks to a
        uniquely created temporary file beside destination, hashed while
        written, aborted as soon as the count exceeds the reference size;
        final count and sha256 must match exactly; then fsync and atomic
        rename. Every failure removes the temporary file and leaves
        destination absent. Error messages never contain the URL.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from pathlib import Path

import httpx
import pytest

BLOB = random.Random(0).randbytes(3 * 1024 * 1024 + 17)
DIGEST = hashlib.sha256(BLOB).hexdigest()
URL = "https://uploads.example/objects/one?X-Signature=abc&X-Expires=9"
ORIGIN = "https://uploads.example:443"


def _mod():
    from rlvr.v3 import download

    return download


def _ref(**over):
    from rlvr.v3.artifacts import ArtifactRef

    base = dict(
        artifact_role="workspace",
        artifact_format="tar_zst_v1",
        sha256=DIGEST,
        compressed_size_bytes=len(BLOB),
        expanded_size_bytes=len(BLOB) * 2,
    )
    base.update(over)
    return ArtifactRef(**base)


def _limits(**over):
    from rlvr.v3.archive import ArchiveLimits

    base = dict(
        max_compressed_bytes=8 << 20,
        max_expanded_bytes=64 << 20,
        max_file_bytes=8 << 20,
        max_entries=64,
        max_path_bytes=256,
        max_zstd_window_bytes=8 << 20,
    )
    base.update(over)
    return ArchiveLimits(**base)


class Server:
    """MockTransport handler: serves BLOB (or an override) and records what was sent."""

    def __init__(self, body=None, status=200, headers=None, chunk=1 << 20, fail_after=None):
        self.body = BLOB if body is None else body
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(self.body))}
        self.chunk = chunk
        self.fail_after = fail_after
        self.calls = 0
        self.sent = 0

    async def __call__(self, request):
        self.calls += 1
        self.request = request
        return httpx.Response(self.status, headers=self.headers, content=self._stream())

    async def _stream(self):
        for offset in range(0, len(self.body), self.chunk):
            if self.fail_after is not None and self.sent >= self.fail_after:
                raise httpx.ReadError("connection lost")
            piece = self.body[offset : offset + self.chunk]
            self.sent += len(piece)
            yield piece


def run(server, tmp_path, *, url=URL, ref=None, limits=None, origins=frozenset({ORIGIN}), dest=None):
    m = _mod()
    dest = dest or tmp_path / "artifact.tar.zst"

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=True) as http:
            return await m.download_artifact(
                http, url, ref or _ref(), dest, limits or _limits(), allowed_origins=origins
            )

    return asyncio.run(go())


def assert_failed(server, tmp_path, **kw):
    m = _mod()
    dest = kw.get("dest") or tmp_path / "artifact.tar.zst"
    with pytest.raises(m.ArtifactDownloadError) as info:
        run(server, tmp_path, **kw)
    assert "uploads.example" not in str(info.value) and "X-Signature" not in str(info.value)
    assert not dest.exists()
    assert stray(tmp_path) == []
    return info.value


def stray(parent: Path) -> list[str]:
    return sorted(p.name for p in parent.iterdir() if p.name not in ("artifact.tar.zst", "keep"))


# --------------------------------------------------------------------------- #
# normalize_https_origin
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://uploads.example/a?b=c", "https://uploads.example:443"),
        ("HTTPS://Uploads.EXAMPLE:443/a", "https://uploads.example:443"),
        ("https://bücher.example/x", "https://xn--bcher-kva.example:443"),
        ("https://uploads.example:8443/", "https://uploads.example:8443"),
    ],
    ids=["default-port", "case-and-explicit-443", "idna", "non-default-port"],
)
def test_normalize_https_origin(url, expected):
    assert _mod().normalize_https_origin(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://uploads.example/a", id="http"),
        pytest.param("https://user:pw@uploads.example/a", id="userinfo"),
        pytest.param("https://uploads.example/a#frag", id="fragment"),
        pytest.param("https:///a", id="empty-host"),
        pytest.param("uploads.example/a", id="no-scheme"),
        pytest.param("ftp://uploads.example/a", id="ftp"),
    ],
)
def test_normalize_https_origin_rejects(url):
    m = _mod()
    with pytest.raises(m.ArtifactDownloadError):
        m.normalize_https_origin(url)


# --------------------------------------------------------------------------- #
# Refusals before any request
# --------------------------------------------------------------------------- #
def test_origin_not_allowlisted_makes_no_request(tmp_path):
    server = Server()
    assert_failed(server, tmp_path, origins=frozenset({"https://other.example:443"}))
    assert server.calls == 0


def test_http_url_makes_no_request(tmp_path):
    server = Server()
    assert_failed(server, tmp_path, url="http://uploads.example/objects/one", origins=frozenset({"https://uploads.example:443", "http://uploads.example:80"}))
    assert server.calls == 0


def test_reference_over_policy_makes_no_request(tmp_path):
    server = Server()
    assert_failed(server, tmp_path, limits=_limits(max_compressed_bytes=len(BLOB) - 1))
    assert server.calls == 0


def test_wrong_format_makes_no_request(tmp_path):
    server = Server()
    ref = _ref().model_copy(update={"artifact_format": "zip_v1"})
    assert_failed(server, tmp_path, ref=ref)
    assert server.calls == 0


def test_existing_destination_is_refused_untouched(tmp_path):
    server = Server()
    dest = tmp_path / "artifact.tar.zst"
    dest.write_bytes(b"keep me")
    m = _mod()
    with pytest.raises(m.ArtifactDownloadError):
        run(server, tmp_path, dest=dest)
    assert dest.read_bytes() == b"keep me"
    assert server.calls == 0
    assert stray(tmp_path) == []


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #
def test_redirect_is_rejected_and_never_followed(tmp_path):
    server = Server(body=b"", status=302, headers={"Location": "https://uploads.example/elsewhere"})
    assert_failed(server, tmp_path)
    assert server.calls == 1


@pytest.mark.parametrize("status", [404, 403, 500, 206])
def test_non_200_status_is_rejected(tmp_path, status):
    server = Server(status=status)
    assert_failed(server, tmp_path)


def test_content_length_mismatch_is_rejected_before_reading_the_body(tmp_path):
    server = Server(headers={"Content-Length": str(len(BLOB) + 1)})
    assert_failed(server, tmp_path)
    assert server.sent == 0


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("abc", id="non-numeric"),
        pytest.param("-1", id="negative"),
        pytest.param(f"{len(BLOB)}, {len(BLOB)}", id="comma-joined"),
        pytest.param(f" {len(BLOB)}x", id="trailing-garbage"),
        pytest.param("", id="empty"),
    ],
)
def test_malformed_content_length_is_rejected(tmp_path, value):
    server = Server(headers={"Content-Length": value})
    assert_failed(server, tmp_path)


def test_missing_content_length_is_allowed(tmp_path):
    server = Server(headers={})
    dest = run(server, tmp_path)
    assert dest.read_bytes() == BLOB


@pytest.mark.parametrize("encoding", ["gzip", "br", "zstd", "identity, gzip"])
def test_non_identity_content_encoding_is_rejected(tmp_path, encoding):
    server = Server(headers={"Content-Length": str(len(BLOB)), "Content-Encoding": encoding})
    assert_failed(server, tmp_path)


def test_identity_content_encoding_is_accepted(tmp_path):
    server = Server(headers={"Content-Length": str(len(BLOB)), "Content-Encoding": "identity"})
    assert run(server, tmp_path).read_bytes() == BLOB


# --------------------------------------------------------------------------- #
# Stream bounds and integrity
# --------------------------------------------------------------------------- #
def test_body_longer_than_reference_is_cut_off_early(tmp_path):
    server = Server(body=BLOB + bytes(8 << 20), headers={}, chunk=64 << 10)
    assert_failed(server, tmp_path)
    assert server.sent <= len(BLOB) + (2 << 20)


def test_body_shorter_than_reference_is_rejected(tmp_path):
    server = Server(body=BLOB[:-1], headers={})
    assert_failed(server, tmp_path)


def test_digest_mismatch_is_rejected(tmp_path):
    corrupted = BLOB[:-1] + bytes([BLOB[-1] ^ 0x01])
    server = Server(body=corrupted)
    assert_failed(server, tmp_path)


def test_transport_error_mid_stream_cleans_up(tmp_path):
    server = Server(fail_after=1 << 20)
    assert_failed(server, tmp_path)


# --------------------------------------------------------------------------- #
# Success
# --------------------------------------------------------------------------- #
def test_success_writes_exact_bytes_atomically(tmp_path):
    server = Server()
    dest = run(server, tmp_path)
    assert dest == tmp_path / "artifact.tar.zst"
    assert dest.read_bytes() == BLOB
    assert stray(tmp_path) == []
    assert server.calls == 1
    assert str(server.request.url) == URL
    assert server.request.method == "GET"


def test_success_with_client_level_redirects_enabled_still_sends_one_request(tmp_path):
    server = Server(chunk=100_000)
    dest = run(server, tmp_path)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == DIGEST
    assert server.calls == 1


def test_verified_download_feeds_archive_verification(tmp_path):
    from rlvr.v3.archive import verify_archive_file

    dest = run(Server(), tmp_path)
    assert verify_archive_file(dest, _ref(), _limits()) is None


# --------------------------------------------------------------------------- #
# The URL never reaches logs, even through the exception chain
# --------------------------------------------------------------------------- #
def _formatted(exc: BaseException) -> str:
    import traceback

    return "".join(traceback.format_exception(exc))


def test_invalid_url_error_chain_carries_no_url():
    m = _mod()
    secret_url = "https://user:sekrit-user@secret-host.example/obj?X-Signature=sekrit-sig#frag"
    with pytest.raises(m.ArtifactDownloadError) as info:
        m.normalize_https_origin(secret_url)
    assert info.value.__cause__ is None
    text = _formatted(info.value)
    assert "secret-host" not in text and "sekrit" not in text


def test_transport_error_chain_carries_no_url(tmp_path):
    m = _mod()
    secret_url = "https://secret-host.example/objects/one?X-Signature=sekrit-sig"
    origins = frozenset({m.normalize_https_origin(secret_url)})
    server = Server(fail_after=0)
    with pytest.raises(m.ArtifactDownloadError) as info:
        run(server, tmp_path, url=secret_url, origins=origins)
    assert info.value.__cause__ is None
    text = _formatted(info.value)
    assert "secret-host" not in text and "sekrit" not in text
    assert stray(tmp_path) == []


# --------------------------------------------------------------------------- #
# download_object: the shared primitive download_artifact wraps
# --------------------------------------------------------------------------- #
def _object(tmp_path, server, *, sha256=DIGEST, size_bytes=None, max_bytes=8 << 20, dest=None, url=URL):
    m = _mod()
    dest = dest or tmp_path / "object.bin"

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=True) as http:
            return await m.download_object(
                http, url, sha256=sha256, size_bytes=len(BLOB) if size_bytes is None else size_bytes,
                max_bytes=max_bytes, destination=dest, allowed_origins=frozenset({ORIGIN}),
            )

    return asyncio.run(go())


def test_download_object_writes_exact_bytes(tmp_path):
    server = Server()
    dest = _object(tmp_path, server)
    assert dest == tmp_path / "object.bin"
    assert dest.read_bytes() == BLOB
    assert server.calls == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["object.bin"]


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"size_bytes": len(BLOB) + 1}, id="size-mismatch"),
        pytest.param({"sha256": "b" * 64}, id="digest-mismatch"),
        pytest.param({"max_bytes": len(BLOB) - 1}, id="over-cap"),
    ],
)
def test_download_object_rejects_and_cleans_up(tmp_path, kw):
    m = _mod()
    server = Server()
    with pytest.raises(m.ArtifactDownloadError) as info:
        _object(tmp_path, server, **kw)
    assert info.value.__cause__ is None
    assert not (tmp_path / "object.bin").exists()
    assert [p.name for p in tmp_path.iterdir()] == []
    if "max_bytes" in kw:
        assert server.calls == 0


def test_download_artifact_still_requires_tar_zst_format(tmp_path):
    # The wrapper keeps the format check that the primitive does not have.
    m = _mod()
    server = Server()
    ref = _ref().model_copy(update={"artifact_format": "zip_v1"})
    with pytest.raises(m.ArtifactDownloadError):
        run(server, tmp_path, ref=ref)
    assert server.calls == 0
    dest = _object(tmp_path, Server(), sha256=ref.sha256, size_bytes=ref.compressed_size_bytes)
    assert dest.read_bytes() == BLOB
