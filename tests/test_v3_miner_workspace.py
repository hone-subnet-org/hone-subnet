from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import replace

import httpx
import pytest

from rlvr.policy import RELEASE_POLICY
from rlvr.v3.archive import ArchiveError
from rlvr.v3.download import ArtifactDownloadError
from rlvr.v3.miner_workspace import READ_BYTES, WorkspaceReader, open_miner_workspace
from tests.test_demo_miner import task
from tests.test_v3_archive import compress, make_tar


def read(reader, path, offset=0):
    return json.loads(
        reader.execute({"tool": "read_file", "path": path, "offset": offset})
    )


def test_reader_paginates_files_and_directory_entries(tmp_path):
    reader = WorkspaceReader(tmp_path)
    contents = b"a" * READ_BYTES + b"tail"
    (tmp_path / "large").write_bytes(contents)
    first = read(reader, "large")
    assert first["content"].encode() == contents[:READ_BYTES]
    second = read(reader, "large", first["next_offset"])
    assert second["content"] == "tail" and second["next_offset"] is None
    for index in range(210):
        (tmp_path / f"file-{index:03}").touch()
    page = json.loads(reader.execute({"tool": "list_files", "path": ".", "offset": 0}))
    assert len(page["entries"]) == 200
    last = json.loads(
        reader.execute(
            {"tool": "list_files", "path": ".", "offset": page["next_offset"]}
        )
    )
    assert len(last["entries"]) == 11 and last["next_offset"] is None


def test_reader_preserves_utf8_across_page_boundaries(tmp_path):
    content = "a" * (READ_BYTES - 1) + "é🙂tail"
    (tmp_path / "utf8").write_text(content)
    reader = WorkspaceReader(tmp_path)
    first = read(reader, "utf8")
    second = read(reader, "utf8", first["next_offset"])
    assert first["content"] + second["content"] == content
    assert second["next_offset"] is None


@pytest.mark.parametrize(
    "path", ["../secret", "/etc/passwd", "nested/../../secret", "missing"]
)
def test_reader_rejects_unsafe_and_missing_paths(tmp_path, path):
    assert "error" in read(WorkspaceReader(tmp_path), path)


def test_reader_rejects_symlink_ancestors(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("private")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    assert "error" in read(WorkspaceReader(workspace), "link/secret")


@pytest.mark.parametrize("offset", [-1, True, "0"])
def test_reader_rejects_invalid_offsets(tmp_path, offset):
    assert "error" in read(WorkspaceReader(tmp_path), "anything", offset)


@pytest.mark.parametrize("fault", [None, "digest", "expanded-size"])
async def test_workspace_is_verified_before_access_and_cleaned_up(fault):
    tar = make_tar([("a.txt", "file", b"original source\n", 0o644)])
    blob = compress(tar)
    request = task()
    ref = request.workspace.model_copy(
        update={
            "sha256": hashlib.sha256(blob).hexdigest()
            if fault != "digest"
            else "0" * 64,
            "compressed_size_bytes": len(blob),
            "expanded_size_bytes": len(tar)
            if fault != "expanded-size"
            else len(tar) - 1,
        }
    )
    # Only the archive reference is consumed by this helper; wire binding is
    # validated by MinerTaskRequest before the production helper is called.
    request = request.model_copy(update={"workspace": ref})
    policy = replace(
        RELEASE_POLICY, v3_artifact_origins=("https://uploads.invalid:443",)
    )

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield blob

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
    ) as http:
        if fault:
            with pytest.raises((ArtifactDownloadError, ArchiveError)):
                async with open_miner_workspace(http, request, policy):
                    pytest.fail("invalid workspace was exposed")
        else:
            async with open_miner_workspace(http, request, policy) as reader:
                root = reader.root
                assert read(reader, "a.txt")["content"] == "original source\n"
            assert not root.exists()


async def test_cancellation_waits_for_extraction_before_removing_files(monkeypatch):
    started, finish = threading.Event(), threading.Event()
    paths = []

    async def download(_http, _url, _ref, destination, *_args, **_kwargs):
        destination.touch()
        return destination

    def extract(_archive, _ref, workspace, *_args, **_kwargs):
        paths.append(workspace)
        started.set()
        assert finish.wait(5)
        assert workspace.parent.exists()
        workspace.mkdir()

    monkeypatch.setattr("rlvr.v3.miner_workspace.download_artifact", download)
    monkeypatch.setattr("rlvr.v3.miner_workspace.extract_archive", extract)

    async def run():
        async with open_miner_workspace(None, task(), RELEASE_POLICY):
            pytest.fail("cancelled extraction exposed a workspace")

    running = asyncio.create_task(run())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        running.cancel()
        await asyncio.sleep(0)
        assert not running.done()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not paths[0].parent.exists()
