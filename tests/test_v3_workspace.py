"""V3 workspace cache and per-miner materialization.

Contract for module ``rlvr.v3.workspace``:

    WorkspaceError(RuntimeError)
    cached_workspace_dir(cache_dir, ref) -> Path        cache_dir / ref.sha256
    ensure_cached_workspace(cache_dir, archive_path, ref, limits, *,
                            scratch_dir) -> Path
        returns the cached directory when it already exists without reading
        the archive; otherwise extracts into a temporary directory beside the
        cache key and renames it into place, so a partial tree never appears
        under the key; failure leaves no cache entry and no temporary directory
    materialize_workspace(cached, destination) -> Path
        destination must not exist; copies directories and regular files
        only, never following links, into an inode-independent tree with
        directories 0755 and files 0755 or 0644 by owner-exec bit; any other
        entry is an error and destination is removed

cache_dir, scratch_dir, and the parent of destination must already exist and
are trusted. No retention or locking.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tests.test_v3_archive import _limits, good_entries, make_tar, write_archive

SENTINEL = "cache-owned.txt"


def _mod():
    from rlvr.v3 import workspace

    return workspace


def _dirs(tmp_path):
    cache = tmp_path / "cache"
    scratch = tmp_path / "scratch"
    miners = tmp_path / "miners"
    for d in (cache, scratch, miners):
        d.mkdir()
    return cache, scratch, miners


def _archive(tmp_path, entries=None):
    return write_archive(tmp_path, make_tar(entries or good_entries()))


def ensure(tmp_path, cache, scratch, archive=None, ref=None):
    m = _mod()
    path, real_ref = archive if archive else _archive(tmp_path)
    return m.ensure_cached_workspace(cache, path, ref or real_ref, _limits(), scratch_dir=scratch)


def tree(root: Path) -> dict[str, tuple[str, bytes | None, int]]:
    out = {}
    for p in sorted(root.rglob("*")):
        info = p.lstat()
        kind = "dir" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "other"
        data = p.read_bytes() if kind == "file" else None
        out[str(p.relative_to(root))] = (kind, data, stat.S_IMODE(info.st_mode))
    return out


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #
def test_cache_dir_is_keyed_by_digest_only(tmp_path):
    m = _mod()
    _, ref = _archive(tmp_path)
    cache, _, _ = _dirs(tmp_path)
    path = m.cached_workspace_dir(cache, ref)
    assert path == cache / ref.sha256
    assert path.name == ref.sha256 and len(path.name) == 64


# --------------------------------------------------------------------------- #
# ensure_cached_workspace
# --------------------------------------------------------------------------- #
def test_first_call_extracts_into_cache(tmp_path):
    cache, scratch, _ = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    assert cached == cache / cached.name
    assert (cached / "pkg" / "run.sh").read_bytes() == b"#!/bin/sh\n"
    assert stat.S_IMODE((cached / "pkg" / "run.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((cached / "pkg" / "data.txt").stat().st_mode) == 0o644
    assert sorted(p.name for p in cache.iterdir()) == [cached.name]
    assert list(scratch.iterdir()) == []


def test_second_call_is_a_cache_hit_without_reading_the_archive(tmp_path):
    cache, scratch, _ = _dirs(tmp_path)
    archive = _archive(tmp_path)
    first = ensure(tmp_path, cache, scratch, archive)
    (first / SENTINEL).write_text("owned")
    path, ref = archive
    path.unlink()  # archive no longer readable
    second = ensure(tmp_path, cache, scratch, (tmp_path / "missing.tar.zst", ref))
    assert second == first
    assert (second / SENTINEL).read_text() == "owned"
    assert list(scratch.iterdir()) == []


def test_digest_mismatch_leaves_no_cache_entry_or_temporary(tmp_path):
    m = _mod()
    cache, scratch, _ = _dirs(tmp_path)
    path, ref = _archive(tmp_path)
    bad = ref.model_copy(update={"sha256": "b" * 64})
    with pytest.raises(m.WorkspaceError):
        ensure(tmp_path, cache, scratch, (path, bad))
    assert list(cache.iterdir()) == []
    assert list(scratch.iterdir()) == []


def test_unsafe_archive_leaves_no_cache_entry_or_temporary(tmp_path):
    m = _mod()
    cache, scratch, _ = _dirs(tmp_path)
    archive = _archive(tmp_path, good_entries() + [("link", "symlink", "pkg/data.txt", 0o777)])
    with pytest.raises(m.WorkspaceError):
        ensure(tmp_path, cache, scratch, archive)
    assert list(cache.iterdir()) == []
    assert list(scratch.iterdir()) == []


def test_stale_scratch_directory_does_not_block(tmp_path):
    cache, scratch, _ = _dirs(tmp_path)
    (scratch / "leftover").mkdir()
    (scratch / "leftover" / "x").write_text("x")
    cached = ensure(tmp_path, cache, scratch)
    assert (cached / "empty").exists()
    assert sorted(p.name for p in scratch.iterdir()) == ["leftover"]


@pytest.mark.parametrize("missing", ["cache", "scratch"])
def test_missing_trusted_directory_is_an_error(tmp_path, missing):
    m = _mod()
    cache, scratch, _ = _dirs(tmp_path)
    (cache if missing == "cache" else scratch).rmdir()
    with pytest.raises(m.WorkspaceError):
        ensure(tmp_path, cache, scratch)


# --------------------------------------------------------------------------- #
# materialize_workspace
# --------------------------------------------------------------------------- #
def test_materialized_copy_matches_cache_bytes_and_modes(tmp_path):
    m = _mod()
    cache, scratch, miners = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    dest = m.materialize_workspace(cached, miners / "uid-7")
    assert dest == miners / "uid-7"
    assert tree(dest) == tree(cached)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o755
    assert all(v[0] in ("dir", "file") for v in tree(dest).values())


def test_copy_is_inode_independent_and_cache_stays_immutable(tmp_path):
    m = _mod()
    cache, scratch, miners = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    before = tree(cached)
    dest = m.materialize_workspace(cached, miners / "uid-7")
    for rel, (kind, _, _) in tree(dest).items():
        if kind == "file":
            assert (dest / rel).stat().st_nlink == 1
            assert (dest / rel).stat().st_ino != (cached / rel).stat().st_ino
    (dest / "pkg" / "data.txt").write_bytes(b"patched")
    (dest / "new.txt").write_bytes(b"new")
    os.chmod(dest / "empty", 0o755)
    assert tree(cached) == before


def test_two_miners_get_independent_trees(tmp_path):
    m = _mod()
    cache, scratch, miners = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    a = m.materialize_workspace(cached, miners / "uid-1")
    b = m.materialize_workspace(cached, miners / "uid-2")
    (a / "pkg" / "data.txt").write_bytes(b"a")
    assert (b / "pkg" / "data.txt").read_bytes() == b"data"
    assert tree(b) == tree(cached)


def test_existing_destination_is_refused_untouched(tmp_path):
    m = _mod()
    cache, scratch, miners = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    dest = miners / "uid-7"
    dest.mkdir()
    (dest / "keep").write_text("keep")
    with pytest.raises(m.WorkspaceError):
        m.materialize_workspace(cached, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["keep"]


@pytest.mark.parametrize("plant", ["symlink", "fifo"])
def test_unsafe_cache_entry_aborts_and_removes_destination(tmp_path, plant):
    m = _mod()
    cache, scratch, miners = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    if plant == "symlink":
        os.symlink("/etc/passwd", cached / "zz-link")
    else:
        os.mkfifo(cached / "zz-pipe")
    with pytest.raises(m.WorkspaceError):
        m.materialize_workspace(cached, miners / "uid-7")
    assert not (miners / "uid-7").exists()


def test_missing_cache_tree_is_an_error(tmp_path):
    m = _mod()
    cache, _, miners = _dirs(tmp_path)
    with pytest.raises(m.WorkspaceError):
        m.materialize_workspace(cache / ("a" * 64), miners / "uid-7")
    assert not (miners / "uid-7").exists()


def test_missing_destination_parent_is_an_error(tmp_path):
    m = _mod()
    cache, scratch, miners = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    with pytest.raises(m.WorkspaceError):
        m.materialize_workspace(cached, miners / "nope" / "uid-7")


# --------------------------------------------------------------------------- #
# Integration with the approved slices
# --------------------------------------------------------------------------- #
def test_cache_then_materialize_then_patch_leaves_cache_unchanged(tmp_path):
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    from rlvr.v3.patch import PatchLimits, apply_patch

    m = _mod()
    cache, scratch, miners = _dirs(tmp_path)
    cached = ensure(tmp_path, cache, scratch)
    before = tree(cached)
    dest = m.materialize_workspace(cached, miners / "uid-7")
    patch = (
        b"diff --git a/pkg/data.txt b/pkg/data.txt\n--- a/pkg/data.txt\n+++ b/pkg/data.txt\n"
        b"@@ -1 +1 @@\n-data\n\\ No newline at end of file\n+patched\n"
    )
    version = subprocess.run([git, "--version"], capture_output=True, text=True, check=True).stdout.split()[2]
    result = apply_patch(dest, patch, PatchLimits(max_patch_bytes=1 << 20, git_timeout_s=10),
                         git_binary=git, expected_git_version=version)
    assert result.status == "applied", result.reason
    assert (dest / "pkg" / "data.txt").read_bytes() == b"patched\n"
    assert tree(cached) == before
