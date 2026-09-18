from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import zstandard

from .artifacts import ArtifactRef
from .canonical import validate_protocol_string


class ArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class ArchiveLimits:
    max_compressed_bytes: int
    max_expanded_bytes: int
    max_file_bytes: int
    max_entries: int
    max_path_bytes: int
    max_zstd_window_bytes: int

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if type(value) is not int or value <= 0:
                raise ValueError("archive limits must be positive integers")


@dataclass(frozen=True)
class ExtractionReport:
    entries: tuple[str, ...]
    expanded_bytes: int
    file_bytes: int


def verify_archive_file(
    path: str | os.PathLike[str],
    ref: ArtifactRef,
    limits: ArchiveLimits,
) -> None:
    source = Path(path)
    if ref.artifact_format != "tar_zst_v1":
        raise ArchiveError("unsupported archive format")
    if ref.expanded_size_bytes > limits.max_expanded_bytes:
        raise ArchiveError("advertised expanded size exceeds the policy limit")
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ArchiveError("archive is unavailable") from exc
    if size != ref.compressed_size_bytes:
        raise ArchiveError("compressed archive size does not match its reference")
    if size > limits.max_compressed_bytes:
        raise ArchiveError("compressed archive exceeds the policy limit")

    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArchiveError("archive could not be read") from exc
    if digest.hexdigest() != ref.sha256:
        raise ArchiveError("archive digest does not match its reference")


def _decompress_archive(
    source: Path,
    target: Path,
    limits: ArchiveLimits,
    advertised_size: int,
) -> int:
    try:
        with source.open("rb") as compressed:
            header = compressed.read(18)
            parameters = zstandard.get_frame_parameters(header)
            if parameters.window_size > limits.max_zstd_window_bytes:
                raise ArchiveError("archive compression window exceeds the policy limit")
            compressed.seek(0)
            decompressor = zstandard.ZstdDecompressor(
                max_window_size=limits.max_zstd_window_bytes
            ).decompressobj()
            written = 0
            with target.open("wb") as expanded:
                for compressed_chunk in iter(lambda: compressed.read(1 << 10), b""):
                    chunk = decompressor.decompress(compressed_chunk)
                    if chunk:
                        written += len(chunk)
                        if written > advertised_size:
                            raise ArchiveError("expanded archive exceeds its advertised size")
                        expanded.write(chunk)
                    if decompressor.eof:
                        break
                if decompressor.eof and (
                    decompressor.unused_data or compressed.read(1)
                ):
                    raise ArchiveError("archive has trailing compressed data")
                final = decompressor.flush()
                if final:
                    written += len(final)
                    if written > advertised_size:
                        raise ArchiveError("expanded archive exceeds its advertised size")
                    expanded.write(final)
            if not decompressor.eof:
                raise ArchiveError("archive compression stream is truncated")
            return written
    except ArchiveError:
        raise
    except (OSError, zstandard.ZstdError) as exc:
        raise ArchiveError("archive decompression failed") from exc


def _validate_name(name: str, limits: ArchiveLimits) -> tuple[str, ...]:
    try:
        validate_protocol_string(name)
    except ValueError as exc:
        raise ArchiveError("archive path is not valid Unicode NFC") from exc
    if len(name.encode("utf-8")) > limits.max_path_bytes:
        raise ArchiveError("archive path exceeds the policy limit")
    if "\x00" in name or "\\" in name or name.startswith("/"):
        raise ArchiveError("archive path is unsafe")
    segments = tuple(name.split("/"))
    if any(segment in ("", ".", "..") for segment in segments):
        raise ArchiveError("archive path is not normalized")
    return segments


def _inspect_members(
    archive: tarfile.TarFile,
    limits: ArchiveLimits,
) -> tuple[list[tarfile.TarInfo], int]:
    members: list[tarfile.TarInfo] = []
    paths: dict[tuple[str, ...], bool] = {}
    file_bytes = 0

    for member in archive:
        if len(members) >= limits.max_entries:
            raise ArchiveError("archive contains too many entries")
        if member.issparse() or not (member.isfile() or member.isdir()):
            raise ArchiveError("archive contains an unsupported entry type")
        segments = _validate_name(member.name, limits)
        if segments in paths:
            raise ArchiveError("archive contains a duplicate path")
        is_file = member.isfile()
        if is_file:
            if member.size < 0 or member.size > limits.max_file_bytes:
                raise ArchiveError("archive file exceeds the policy limit")
            file_bytes += member.size
            if file_bytes > limits.max_expanded_bytes:
                raise ArchiveError("archive files exceed the policy limit")
        paths[segments] = is_file
        members.append(member)

    for path, is_file in paths.items():
        if not is_file:
            continue
        if any(other[: len(path)] == path and len(other) > len(path) for other in paths):
            raise ArchiveError("archive file is used as a parent directory")
    return members, file_bytes


def _extract_members(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    destination: Path,
) -> None:
    for member in members:
        output = destination.joinpath(*member.name.split("/"))
        if member.isdir():
            output.mkdir(mode=0o755, parents=True, exist_ok=True)
            os.chmod(output, 0o755)
            continue

        output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ArchiveError("archive file contents are unavailable")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        copied = 0
        with source, os.fdopen(os.open(output, flags, 0o600), "wb") as target:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                copied += len(chunk)
                if copied > member.size:
                    raise ArchiveError("archive file exceeds its declared size")
                target.write(chunk)
            if copied != member.size:
                raise ArchiveError("archive file is truncated")
            mode = 0o755 if member.mode & stat.S_IXUSR else 0o644
            os.fchmod(target.fileno(), mode)


def extract_archive(
    path: str | os.PathLike[str],
    ref: ArtifactRef,
    destination: str | os.PathLike[str],
    limits: ArchiveLimits,
    *,
    scratch_dir: str | os.PathLike[str],
) -> ExtractionReport:
    source = Path(path)
    dest = Path(destination)
    scratch = Path(scratch_dir)
    if dest.exists():
        raise ArchiveError("archive destination already exists")
    if not scratch.is_dir():
        raise ArchiveError("archive scratch directory is unavailable")

    verify_archive_file(source, ref, limits)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="hone-v3-", suffix=".tar", dir=scratch
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    created_destination = False
    try:
        expanded_bytes = _decompress_archive(
            source, temporary, limits, ref.expanded_size_bytes
        )
        if expanded_bytes != ref.expanded_size_bytes:
            raise ArchiveError("expanded archive size does not match its reference")
        try:
            with tarfile.open(temporary, mode="r:", errorlevel=2) as archive:
                members, file_bytes = _inspect_members(archive, limits)
                dest.mkdir(mode=0o700)
                created_destination = True
                _extract_members(archive, members, dest)
        except (OSError, tarfile.TarError) as exc:
            raise ArchiveError("archive tar payload is invalid") from exc
        return ExtractionReport(
            entries=tuple(member.name for member in members),
            expanded_bytes=expanded_bytes,
            file_bytes=file_bytes,
        )
    except ArchiveError:
        if created_destination and dest.exists():
            shutil.rmtree(dest)
        raise
    except OSError as exc:
        if created_destination and dest.exists():
            shutil.rmtree(dest)
        raise ArchiveError("archive extraction failed") from exc
    finally:
        temporary.unlink(missing_ok=True)
