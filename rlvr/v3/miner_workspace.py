"""Verified, temporary workspace access for the reference miner."""

from __future__ import annotations

import asyncio
import codecs
import json
import shutil
import stat
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from ..policy import ValidatorPolicy
from .api import MinerTaskRequest
from .archive import ArchiveLimits, extract_archive
from .download import download_artifact
from .identity import validate_relative_path

READ_BYTES = 32 * 1024
LIST_ENTRIES = 200


class WorkspaceReader:
    """Expose bounded reads without executing code from the workspace."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, relative: str) -> Path:
        validate_relative_path(relative)
        current = self.root
        for segment in () if relative == "." else relative.split("/"):
            current = current / segment
            info = current.lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("workspace path is not a regular file or directory")
        return current

    def execute(self, arguments: dict) -> bytes:
        try:
            if set(arguments) != {"tool", "path", "offset"}:
                raise ValueError("tool requires tool, path, and offset")
            relative, offset = arguments["path"], arguments["offset"]
            if type(relative) is not str or type(offset) is not int or offset < 0:
                raise ValueError(
                    "path must be a string and offset a nonnegative integer"
                )
            path = self._path(relative)
            if arguments["tool"] == "list_files":
                entries = sorted(path.iterdir(), key=lambda item: item.name)
                page = entries[offset : offset + LIST_ENTRIES]
                value = {
                    "entries": [
                        {"name": item.name, "directory": item.is_dir()} for item in page
                    ],
                    "next_offset": offset + len(page)
                    if offset + len(page) < len(entries)
                    else None,
                }
            elif arguments["tool"] == "read_file":
                if not path.is_file():
                    raise ValueError("read_file requires a regular file")
                with path.open("rb") as handle:
                    handle.seek(offset)
                    raw = handle.read(READ_BYTES)
                    has_more = bool(handle.read(1))
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                content = decoder.decode(raw, final=not has_more)
                buffered, _ = decoder.getstate()
                value = {
                    "content": content,
                    "encoding": "UTF-8; invalid byte sequences replaced",
                    "next_offset": offset + len(raw) - len(buffered)
                    if has_more
                    else None,
                }
            else:
                raise ValueError("unknown workspace tool")
        except (OSError, ValueError, OverflowError) as error:
            # Never include host paths from filesystem exceptions in model context.
            value = {
                "error": str(error)
                if isinstance(error, ValueError)
                else "workspace path could not be read"
            }
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )


@asynccontextmanager
async def open_miner_workspace(
    http: httpx.AsyncClient,
    request: MinerTaskRequest,
    policy: ValidatorPolicy,
):
    limits = ArchiveLimits(
        max_compressed_bytes=policy.v3_workspace_compressed_bytes,
        max_expanded_bytes=policy.v3_workspace_bytes,
        max_file_bytes=policy.v3_max_file_bytes,
        max_entries=200_000,
        max_path_bytes=4_096,
        max_zstd_window_bytes=128 * 1024**2,
    )
    if request.workspace.expanded_size_bytes > limits.max_expanded_bytes:
        raise ValueError("workspace exceeds the expanded byte limit")
    with tempfile.TemporaryDirectory(prefix="hone-v3-miner-") as temporary:
        root = Path(temporary)
        required = (
            request.workspace.compressed_size_bytes
            + 2 * request.workspace.expanded_size_bytes
        )
        if shutil.disk_usage(root).free < required:
            raise ValueError("insufficient workspace storage")
        archive = await download_artifact(
            http,
            request.workspace_url,
            request.workspace,
            root / "workspace.tar.zst",
            limits,
            allowed_origins=frozenset(policy.v3_artifact_origins),
        )
        workspace = root / "workspace"
        extracting = asyncio.create_task(
            asyncio.to_thread(
                extract_archive,
                archive,
                request.workspace,
                workspace,
                limits,
                scratch_dir=root,
            )
        )
        try:
            # Keep temporary files alive until the bounded extraction finishes,
            # including when the solve deadline cancels the caller.
            await asyncio.shield(extracting)
        except asyncio.CancelledError:
            await asyncio.gather(extracting, return_exceptions=True)
            raise
        yield WorkspaceReader(workspace)
