"""V3 miner submission fetch.

Contract for module ``rlvr.v3.submission``:

    SubmissionLimits(max_patch_bytes, max_script_bytes)   frozen, both required
    FetchedSubmission(path, artifact_role, artifact_format, size_bytes, sha256)
    async fetch_submission(http, grant, destination, limits, *,
                           allowed_origins) -> FetchedSubmission
        cap chosen by grant.format (patch -> max_patch_bytes,
        script -> max_script_bytes); grant.size_bytes over the cap fails
        before any request; the object is downloaded from grant.read_url via
        rlvr.v3.download.download_object and verified against grant.sha256
        and grant.size_bytes; the stored file is made read-only (0444).
        Every failure raises ArtifactDownloadError with no URL in the message
        or chain, and destination stays absent. Script content validation is
        rlvr.v3.script's job, not this module's.
"""

from __future__ import annotations

import asyncio
import stat
import traceback

import httpx
import pytest

from tests.test_v3_download import BLOB, DIGEST, ORIGIN, URL, Server

HEX = "a" * 64


def _mod():
    from rlvr.v3 import submission

    return submission


def _limits(**over):
    base = dict(max_patch_bytes=4 << 20, max_script_bytes=8 << 20)
    base.update(over)
    return _mod().SubmissionLimits(**base)


def _grant(**over):
    from rlvr.v3.artifacts import ArtifactGrant

    base = dict(
        uid=7,
        hotkey="hk-7",
        upload_id="up-patch",
        sha256=DIGEST,
        size_bytes=len(BLOB),
        format="unified_diff_v1",
        read_url=URL,
    )
    base.update(over)
    return ArtifactGrant(**base)


def _script_grant(**over):
    return _grant(upload_id="up-script", format="bash_script_v1", **over)


def fetch(tmp_path, server, grant=None, limits=None, *, origins=frozenset({ORIGIN}), dest=None):
    m = _mod()

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as http:
            return await m.fetch_submission(
                http, grant or _grant(), dest or tmp_path / "submission.bin", limits or _limits(),
                allowed_origins=origins,
            )

    return asyncio.run(go())


def assert_failed(tmp_path, server, **kw):
    from rlvr.v3.download import ArtifactDownloadError

    with pytest.raises(ArtifactDownloadError) as info:
        fetch(tmp_path, server, **kw)
    text = "".join(traceback.format_exception(info.value))
    assert "uploads.example" not in text and "X-Signature" not in text
    assert info.value.__cause__ is None
    assert not (tmp_path / "submission.bin").exists()
    assert [p.name for p in tmp_path.iterdir()] == []
    return server


def test_limits_require_both_fields():
    m = _mod()
    with pytest.raises(TypeError):
        m.SubmissionLimits(max_patch_bytes=1)


def test_patch_grant_is_fetched_verified_and_read_only(tmp_path):
    m = _mod()
    server = Server()
    fetched = fetch(tmp_path, server)
    assert isinstance(fetched, m.FetchedSubmission)
    assert fetched.path == tmp_path / "submission.bin"
    assert fetched.path.read_bytes() == BLOB
    assert stat.S_IMODE(fetched.path.stat().st_mode) == 0o444
    assert (fetched.artifact_role, fetched.artifact_format) == ("patch", "unified_diff_v1")
    assert (fetched.size_bytes, fetched.sha256) == (len(BLOB), DIGEST)
    assert server.calls == 1 and str(server.request.url) == URL
    assert [p.name for p in tmp_path.iterdir()] == ["submission.bin"]


def test_script_grant_uses_the_script_cap(tmp_path):
    # Patch cap smaller than the object; script cap large enough.
    fetched = fetch(tmp_path, Server(), _script_grant(), _limits(max_patch_bytes=1024))
    assert fetched.artifact_role == "script"
    assert fetched.path.read_bytes() == BLOB


def test_patch_grant_over_patch_cap_makes_no_request(tmp_path):
    server = assert_failed(tmp_path, Server(), limits=_limits(max_patch_bytes=len(BLOB) - 1, max_script_bytes=64 << 20))
    assert server.calls == 0


def test_script_grant_over_script_cap_makes_no_request(tmp_path):
    server = assert_failed(tmp_path, Server(), grant=_script_grant(), limits=_limits(max_script_bytes=len(BLOB) - 1))
    assert server.calls == 0


def test_origin_not_allowlisted_makes_no_request(tmp_path):
    server = assert_failed(tmp_path, Server(), origins=frozenset({"https://other.example:443"}))
    assert server.calls == 0


def test_digest_mismatch_is_rejected(tmp_path):
    assert_failed(tmp_path, Server(), grant=_grant(sha256="b" * 64))


def test_size_mismatch_is_rejected(tmp_path):
    assert_failed(tmp_path, Server(), grant=_grant(size_bytes=len(BLOB) - 1))


def test_existing_destination_is_refused_untouched(tmp_path):
    from rlvr.v3.download import ArtifactDownloadError

    dest = tmp_path / "submission.bin"
    dest.write_bytes(b"keep")
    server = Server()
    with pytest.raises(ArtifactDownloadError):
        fetch(tmp_path, server, dest=dest)
    assert dest.read_bytes() == b"keep" and server.calls == 0
    assert [p.name for p in tmp_path.iterdir()] == ["submission.bin"]


def test_fetch_goes_through_download_object(tmp_path, monkeypatch):
    from rlvr.v3 import download

    seen = {}

    async def fake_download_object(http, url, *, sha256, size_bytes, max_bytes, destination, allowed_origins):
        seen.update(url=url, sha256=sha256, size_bytes=size_bytes, max_bytes=max_bytes)
        destination.write_bytes(BLOB)
        return destination

    monkeypatch.setattr(download, "download_object", fake_download_object)
    fetched = fetch(tmp_path, Server(), _script_grant(), _limits(max_script_bytes=len(BLOB) + 5))
    assert seen == dict(url=URL, sha256=DIGEST, size_bytes=len(BLOB), max_bytes=len(BLOB) + 5)
    assert stat.S_IMODE(fetched.path.stat().st_mode) == 0o444
