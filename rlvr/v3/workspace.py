from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from .archive import ArchiveError, ArchiveLimits, extract_archive
from .artifacts import ArtifactRef


class WorkspaceError(RuntimeError):
    pass


def cached_workspace_dir(
    cache_dir: str | os.PathLike[str], ref: ArtifactRef
) -> Path:
    return Path(cache_dir) / ref.sha256


def _trusted_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)


def ensure_cached_workspace(
    cache_dir: str | os.PathLike[str],
    archive_path: str | os.PathLike[str],
    ref: ArtifactRef,
    limits: ArchiveLimits,
    *,
    scratch_dir: str | os.PathLike[str],
) -> Path:
    cache = Path(cache_dir)
    scratch = Path(scratch_dir)
    if not _trusted_directory(cache) or not _trusted_directory(scratch):
        raise WorkspaceError("workspace directories are unavailable")

    target = cached_workspace_dir(cache, ref)
    if target.exists():
        if not _trusted_directory(target):
            raise WorkspaceError("cached workspace is invalid")
        return target

    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=".workspace-", dir=cache))
        staging.rmdir()
        extract_archive(
            archive_path,
            ref,
            staging,
            limits,
            scratch_dir=scratch,
        )
        os.replace(staging, target)
        staging = None
        return target
    except (ArchiveError, OSError):
        if _trusted_directory(target):
            return target
        raise WorkspaceError("workspace could not be cached") from None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _copy_tree(source: Path, destination: Path) -> None:
    for entry in os.scandir(source):
        source_path = Path(entry.path)
        destination_path = destination / entry.name
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            destination_path.mkdir(mode=0o755)
            _copy_tree(source_path, destination_path)
            os.chmod(destination_path, 0o755)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise WorkspaceError("cached workspace contains an unsafe entry")

        read_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        with os.fdopen(os.open(source_path, read_flags), "rb") as source_file:
            with os.fdopen(os.open(destination_path, write_flags, 0o600), "wb") as output:
                shutil.copyfileobj(source_file, output, length=1 << 20)
                mode = 0o755 if info.st_mode & stat.S_IXUSR else 0o644
                os.fchmod(output.fileno(), mode)


def materialize_workspace(
    cached: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> Path:
    source = Path(cached)
    target = Path(destination)
    if not _trusted_directory(source):
        raise WorkspaceError("cached workspace is unavailable")
    if not _trusted_directory(target.parent):
        raise WorkspaceError("workspace parent is unavailable")
    if target.exists() or target.is_symlink():
        raise WorkspaceError("workspace destination already exists")

    created = False
    complete = False
    try:
        target.mkdir(mode=0o755)
        created = True
        _copy_tree(source, target)
        os.chmod(target, 0o755)
        complete = True
        return target
    except WorkspaceError:
        raise
    except OSError:
        raise WorkspaceError("workspace could not be materialized") from None
    finally:
        if created and not complete:
            shutil.rmtree(target, ignore_errors=True)
