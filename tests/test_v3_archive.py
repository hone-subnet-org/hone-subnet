"""V3 ``tar_zst_v1`` archive verification and extraction.

Contract for module ``rlvr.v3.archive``:

    ArchiveError(ValueError)
    ArchiveLimits(max_compressed_bytes, max_expanded_bytes, max_file_bytes,
                  max_entries, max_path_bytes, max_zstd_window_bytes)
                                frozen dataclass, every field required
    ExtractionReport(entries, expanded_bytes, file_bytes)
    verify_archive_file(path, ref, limits) -> None
                                artifact_format tar_zst_v1; file size equals
                                ref.compressed_size_bytes and is within the
                                limit; streamed sha256 equals ref.sha256
    extract_archive(path, ref, dest, limits, *, scratch_dir) -> ExtractionReport
                                verify, stream-decompress into a temporary
                                file under scratch_dir capped at
                                max_expanded_bytes with the zstd window bound,
                                inspect every tarfile member before writing,
                                then copy validated file streams; dest must
                                not exist beforehand; any failure removes dest
                                and the temporary file

Member rules: regular files and directories only; names strict UTF-8, NFC,
relative, no ".", "..", or empty segments, no backslash or NUL, at most
max_path_bytes UTF-8 bytes; no duplicates; a file is never a parent; per-file
and entry caps. Metadata is discarded: files 0755 when the owner-exec bit is
set, else 0644; directories 0755.

The advertised expanded_size_bytes is exact and bounds decompression.
Requires the ``zstandard`` package.
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tarfile
from pathlib import Path

import pytest

NFC_E = "é"
NFD_E = "é"


def _mod():
    from rlvr.v3 import archive

    return archive


def _limits(**over):
    m = _mod()
    base = dict(
        max_compressed_bytes=1 << 20,
        max_expanded_bytes=4 << 20,
        max_file_bytes=1 << 20,
        max_entries=64,
        max_path_bytes=256,
        max_zstd_window_bytes=8 << 20,
    )
    base.update(over)
    return m.ArchiveLimits(**base)


def make_tar(entries, fmt=tarfile.PAX_FORMAT) -> bytes:
    """entries: (name, kind, data_or_linkname, mode). kind in
    file|dir|symlink|hardlink|fifo|chr|sparse."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for name, kind, payload, mode in entries:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.uid = info.gid = 12345
            info.uname = info.gname = "nobody"
            if kind == "file":
                info.type = tarfile.REGTYPE
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
                continue
            info.type = {
                "dir": tarfile.DIRTYPE,
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
                "chr": tarfile.CHRTYPE,
                "sparse": tarfile.GNUTYPE_SPARSE,
            }[kind]
            if kind in ("symlink", "hardlink"):
                info.linkname = payload
            tf.addfile(info)
    return buf.getvalue()


def compress(raw: bytes, window_log: int | None = None) -> bytes:
    import zstandard

    if window_log is None:
        return zstandard.ZstdCompressor(level=3).compress(raw)
    params = zstandard.ZstdCompressionParameters(window_log=window_log)
    return zstandard.ZstdCompressor(compression_params=params).compress(raw)


def write_archive(tmp_path: Path, tar_bytes: bytes, *, window_log=None, **ref_over):
    from rlvr.v3.artifacts import ArtifactRef

    blob = compress(tar_bytes, window_log)
    path = tmp_path / "artifact.tar.zst"
    path.write_bytes(blob)
    fields = dict(
        artifact_role="workspace",
        artifact_format="tar_zst_v1",
        sha256=hashlib.sha256(blob).hexdigest(),
        compressed_size_bytes=len(blob),
        expanded_size_bytes=len(tar_bytes),
    )
    fields.update(ref_over)
    return path, ArtifactRef(**fields)


def good_entries():
    return [
        ("pkg", "dir", b"", 0o700),
        ("pkg/run.sh", "file", b"#!/bin/sh\n", 0o4755),
        ("pkg/data.txt", "file", b"data", 0o600),
        ("empty", "file", b"", 0o644),
        ("docs", "dir", b"", 0o755),
        ("docs/" + NFC_E + ".md", "file", b"# e", 0o644),
    ]


def extract(tmp_path, tar_bytes, limits=None, *, window_log=None, **ref_over):
    m = _mod()
    path, ref = write_archive(tmp_path, tar_bytes, window_log=window_log, **ref_over)
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)  # a prior rejection in the same tmp_path leaves it empty
    assert scratch.is_dir() and list(scratch.iterdir()) == []
    dest = tmp_path / "dest"
    report = m.extract_archive(
        path, ref, dest, limits or _limits(), scratch_dir=scratch
    )
    return report, dest, scratch


def assert_rejected(tmp_path, tar_bytes, limits=None, *, window_log=None, **ref_over):
    m = _mod()
    with pytest.raises(m.ArchiveError):
        extract(tmp_path, tar_bytes, limits, window_log=window_log, **ref_over)
    assert not (tmp_path / "dest").exists()
    assert list((tmp_path / "scratch").iterdir()) == []


# --------------------------------------------------------------------------- #
# verify_archive_file
# --------------------------------------------------------------------------- #
def test_limits_require_every_field():
    m = _mod()
    with pytest.raises(TypeError):
        m.ArchiveLimits(max_compressed_bytes=1)


def test_verify_accepts_exact_size_and_digest(tmp_path):
    m = _mod()
    path, ref = write_archive(tmp_path, make_tar(good_entries()))
    assert m.verify_archive_file(path, ref, _limits()) is None


@pytest.mark.parametrize("delta", [-1, 1], ids=["short", "long"])
def test_verify_rejects_compressed_size_mismatch(tmp_path, delta):
    m = _mod()
    path, ref = write_archive(tmp_path, make_tar(good_entries()))
    ref = type(ref)(**{**ref.model_dump(), "compressed_size_bytes": ref.compressed_size_bytes + delta})
    with pytest.raises(m.ArchiveError):
        m.verify_archive_file(path, ref, _limits())


def test_verify_rejects_compressed_size_over_limit(tmp_path):
    m = _mod()
    path, ref = write_archive(tmp_path, make_tar(good_entries()))
    with pytest.raises(m.ArchiveError):
        m.verify_archive_file(path, ref, _limits(max_compressed_bytes=ref.compressed_size_bytes - 1))


def test_verify_rejects_digest_mismatch(tmp_path):
    m = _mod()
    path, ref = write_archive(tmp_path, make_tar(good_entries()), sha256="b" * 64)
    with pytest.raises(m.ArchiveError):
        m.verify_archive_file(path, ref, _limits())


def test_verify_rejects_wrong_format(tmp_path):
    m = _mod()
    path, ref = write_archive(tmp_path, make_tar(good_entries()))
    ref = ref.model_copy(update={"artifact_format": "zip_v1"})
    with pytest.raises((m.ArchiveError, ValueError)):
        m.verify_archive_file(path, ref, _limits())


# --------------------------------------------------------------------------- #
# zstd stream bounds
# --------------------------------------------------------------------------- #
def test_truncated_stream_is_rejected(tmp_path):
    m = _mod()
    from rlvr.v3.artifacts import ArtifactRef

    tar_bytes = make_tar(good_entries())
    blob = compress(tar_bytes)[:-7]
    path = tmp_path / "artifact.tar.zst"
    path.write_bytes(blob)
    ref = ArtifactRef(
        artifact_role="workspace",
        artifact_format="tar_zst_v1",
        sha256=hashlib.sha256(blob).hexdigest(),
        compressed_size_bytes=len(blob),
        expanded_size_bytes=len(tar_bytes),
    )
    (tmp_path / "scratch").mkdir()
    with pytest.raises(m.ArchiveError):
        m.extract_archive(path, ref, tmp_path / "dest", _limits(), scratch_dir=tmp_path / "scratch")
    assert not (tmp_path / "dest").exists()


@pytest.mark.parametrize("suffix", [b"trailing", compress(b"second frame")])
def test_trailing_data_or_a_second_zstd_frame_is_rejected(tmp_path, suffix):
    from rlvr.v3.artifacts import ArtifactRef

    tar_bytes = make_tar(good_entries())
    blob = compress(tar_bytes) + suffix
    path = tmp_path / "artifact.tar.zst"
    path.write_bytes(blob)
    ref = ArtifactRef(
        artifact_role="workspace",
        artifact_format="tar_zst_v1",
        sha256=hashlib.sha256(blob).hexdigest(),
        compressed_size_bytes=len(blob),
        expanded_size_bytes=len(tar_bytes),
    )
    (tmp_path / "scratch").mkdir()
    with pytest.raises(_mod().ArchiveError):
        _mod().extract_archive(
            path,
            ref,
            tmp_path / "dest",
            _limits(),
            scratch_dir=tmp_path / "scratch",
        )


def test_window_size_over_limit_is_rejected(tmp_path):
    import zstandard

    # Content larger than the window so the frame header advertises the full
    # 16 MiB window rather than the content size.
    tar_bytes = make_tar([("zeros", "file", b"\0" * (20 << 20), 0o644)])
    blob = compress(tar_bytes, window_log=24)
    assert zstandard.get_frame_parameters(blob).window_size == 16 << 20
    limits = _limits(
        max_zstd_window_bytes=8 << 20,
        max_expanded_bytes=64 << 20,
        max_file_bytes=64 << 20,
    )
    assert_rejected(tmp_path, tar_bytes, limits, window_log=24)


def test_window_size_within_limit_is_accepted(tmp_path):
    report, _, _ = extract(tmp_path, make_tar(good_entries()), _limits(max_zstd_window_bytes=8 << 20), window_log=20)
    assert report.file_bytes == 4 + 10 + 3


def test_expanded_cap_stops_decompression_bomb(tmp_path):
    zeros = make_tar([("zeros", "file", b"\0" * (32 << 20), 0o644)])
    limits = _limits(max_expanded_bytes=1 << 20, max_file_bytes=64 << 20, max_compressed_bytes=1 << 20)
    assert_rejected(tmp_path, zeros, limits)


@pytest.mark.parametrize("difference", [-1, 1])
def test_advertised_expanded_size_must_match_exactly(tmp_path, difference):
    tar_bytes = make_tar(good_entries())
    with pytest.raises(_mod().ArchiveError):
        extract(tmp_path, tar_bytes, expanded_size_bytes=len(tar_bytes) + difference)


# --------------------------------------------------------------------------- #
# member types
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(("link", "symlink", "pkg/data.txt", 0o777), id="symlink"),
        pytest.param(("link", "hardlink", "pkg/data.txt", 0o644), id="hardlink"),
        pytest.param(("pipe", "fifo", b"", 0o644), id="fifo"),
        pytest.param(("dev", "chr", b"", 0o644), id="chr-device"),
    ],
)
def test_special_member_types_are_rejected(tmp_path, entry):
    assert_rejected(tmp_path, make_tar(good_entries() + [entry]))


def test_sparse_member_is_rejected(tmp_path):
    assert_rejected(tmp_path, make_tar(good_entries() + [("sp", "sparse", b"", 0o644)], fmt=tarfile.GNU_FORMAT))


# --------------------------------------------------------------------------- #
# member names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("/etc/passwd", id="absolute"),
        pytest.param("../escape", id="parent"),
        pytest.param("pkg/../escape", id="parent-inner"),
        pytest.param("./pkg/x", id="dot-prefix"),
        pytest.param("pkg/./x", id="dot-inner"),
        pytest.param("pkg//x", id="empty-segment"),
        pytest.param("pkg\\x", id="backslash"),
        pytest.param("docs/" + NFD_E + ".md", id="nfd"),
    ],
)
def test_unsafe_member_names_are_rejected(tmp_path, name):
    assert_rejected(tmp_path, make_tar(good_entries() + [(name, "file", b"x", 0o644)]))


def test_path_limit_counts_utf8_bytes(tmp_path):
    name = NFC_E * 100  # 100 characters, 200 bytes
    tar_bytes = make_tar([(name, "file", b"x", 0o644)])
    assert_rejected(tmp_path, tar_bytes, _limits(max_path_bytes=150))
    report, _, _ = extract(tmp_path, tar_bytes, _limits(max_path_bytes=200))
    assert report.entries == (name,)


@pytest.mark.parametrize("fmt", [tarfile.GNU_FORMAT, tarfile.PAX_FORMAT], ids=["gnu", "pax"])
def test_long_names_resolved_by_tarfile_are_accepted(tmp_path, fmt):
    name = "d" * 120 + "/" + "f" * 60
    report, dest, _ = extract(tmp_path, make_tar([("d" * 120, "dir", b"", 0o755), (name, "file", b"x", 0o644)], fmt=fmt))
    assert report.entries == ("d" * 120, name)
    assert (dest / name).read_bytes() == b"x"


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def test_duplicate_names_are_rejected(tmp_path):
    assert_rejected(tmp_path, make_tar(good_entries() + [("pkg/data.txt", "file", b"again", 0o644)]))


def test_file_used_as_parent_is_rejected(tmp_path):
    assert_rejected(tmp_path, make_tar([("a", "file", b"x", 0o644), ("a/b", "file", b"y", 0o644)]))


def test_entry_cap_is_enforced(tmp_path):
    entries = [(f"f{i}", "file", b"x", 0o644) for i in range(5)]
    assert_rejected(tmp_path, make_tar(entries), _limits(max_entries=4))
    report, _, _ = extract(tmp_path, make_tar(entries), _limits(max_entries=5))
    assert len(report.entries) == 5


def test_per_file_cap_is_enforced(tmp_path):
    big = make_tar([("big", "file", b"z" * 1025, 0o644)])
    assert_rejected(tmp_path, big, _limits(max_file_bytes=1024))
    report, _, _ = extract(tmp_path, big, _limits(max_file_bytes=1025))
    assert report.file_bytes == 1025


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_extracts_files_and_directories_with_collapsed_modes(tmp_path):
    report, dest, scratch = extract(tmp_path, make_tar(good_entries()))

    assert report.entries == tuple(name for name, *_ in good_entries())
    assert report.file_bytes == 10 + 4 + 0 + 3
    assert report.expanded_bytes == len(make_tar(good_entries()))

    assert (dest / "pkg" / "run.sh").read_bytes() == b"#!/bin/sh\n"
    assert (dest / "docs" / (NFC_E + ".md")).read_bytes() == b"# e"
    assert (dest / "empty").stat().st_size == 0

    def mode(rel):
        return stat.S_IMODE((dest / rel).stat().st_mode)

    assert mode("pkg/run.sh") == 0o755  # setuid stripped, exec kept
    assert mode("pkg/data.txt") == 0o644  # 0600 widened to the fixed file mode
    assert mode("pkg") == 0o755  # 0700 collapsed to the fixed dir mode
    assert mode("docs") == 0o755
    assert (dest / "pkg" / "run.sh").stat().st_uid == os.getuid()
    assert list(scratch.iterdir()) == []


def test_extracted_tree_contains_only_listed_entries(tmp_path):
    report, dest, _ = extract(tmp_path, make_tar(good_entries()))
    found = sorted(str(p.relative_to(dest)) for p in dest.rglob("*"))
    assert found == sorted(report.entries)
    assert not any(p.is_symlink() for p in dest.rglob("*"))


def test_existing_destination_is_rejected_untouched(tmp_path):
    m = _mod()
    path, ref = write_archive(tmp_path, make_tar(good_entries()))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep").write_text("keep")
    (tmp_path / "scratch").mkdir()
    with pytest.raises(m.ArchiveError):
        m.extract_archive(path, ref, dest, _limits(), scratch_dir=tmp_path / "scratch")
    assert (dest / "keep").read_text() == "keep"
    assert sorted(p.name for p in dest.iterdir()) == ["keep"]


def test_failure_after_partial_extraction_removes_destination(tmp_path):
    # Valid entries first, symlink last: rejection must happen before any write,
    # and nothing may remain either way.
    entries = good_entries() + [("late", "symlink", "pkg/data.txt", 0o777)]
    assert_rejected(tmp_path, make_tar(entries))


def test_not_a_tar_payload_is_rejected(tmp_path):
    assert_rejected(tmp_path, b"this is not a tar archive" * 100)


# --------------------------------------------------------------------------- #
# Regressions: realistic frame shapes
# --------------------------------------------------------------------------- #
def _write_blob(tmp_path, blob: bytes, expanded: int):
    from rlvr.v3.artifacts import ArtifactRef

    path = tmp_path / "artifact.tar.zst"
    path.write_bytes(blob)
    ref = ArtifactRef(
        artifact_role="workspace",
        artifact_format="tar_zst_v1",
        sha256=hashlib.sha256(blob).hexdigest(),
        compressed_size_bytes=len(blob),
        expanded_size_bytes=expanded,
    )
    (tmp_path / "scratch").mkdir(exist_ok=True)
    return path, ref


def test_window_within_limit_on_content_larger_than_window_extracts(tmp_path):
    import zstandard

    data = b"abc" * (700 << 10)  # ~2 MiB, larger than the 1 MiB window
    tar_bytes = make_tar([("data", "file", data, 0o644)])
    blob = compress(tar_bytes, window_log=20)
    assert zstandard.get_frame_parameters(blob).window_size == 1 << 20
    limits = _limits(max_zstd_window_bytes=8 << 20, max_file_bytes=4 << 20)
    report, dest, _ = extract(tmp_path, tar_bytes, limits, window_log=20)
    assert report.file_bytes == len(data)
    assert (dest / "data").stat().st_size == len(data)


def test_frame_without_content_size_extracts(tmp_path):
    import zstandard

    m = _mod()
    tar_bytes = make_tar(good_entries())
    blob = zstandard.ZstdCompressor(write_content_size=False).compress(tar_bytes)
    path, ref = _write_blob(tmp_path, blob, len(tar_bytes))
    report = m.extract_archive(path, ref, tmp_path / "dest", _limits(), scratch_dir=tmp_path / "scratch")
    assert report.expanded_bytes == len(tar_bytes)
    assert report.entries == tuple(name for name, *_ in good_entries())


def test_truncated_frame_without_content_size_is_rejected(tmp_path):
    import zstandard

    m = _mod()
    tar_bytes = make_tar(good_entries())
    blob = zstandard.ZstdCompressor(write_content_size=False).compress(tar_bytes)[:-10]
    path, ref = _write_blob(tmp_path, blob, len(tar_bytes))
    with pytest.raises(m.ArchiveError):
        m.extract_archive(path, ref, tmp_path / "dest", _limits(), scratch_dir=tmp_path / "scratch")
    assert not (tmp_path / "dest").exists()
    assert list((tmp_path / "scratch").iterdir()) == []
