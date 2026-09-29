"""Contract for ``rlvr.v3.tree`` — the shared result-tree policy.

    TreeError(ValueError)
    TreeLimits(max_entries, max_file_bytes, max_total_file_bytes, max_path_bytes)
        frozen; every field a positive int (ValueError otherwise)
    TreeReport(entries: tuple[str, ...], file_bytes: int)   frozen; entries are
        the sorted NFC POSIX relative paths of every dir and file; file_bytes is
        the sum of regular-file sizes
    inspect_tree(root, limits: TreeLimits | None, normalize_modes: bool) -> TreeReport
        root must be a real (non-symlink) directory; every entry must be a
        directory or an nlink==1 regular file (no symlinks, fifos, hardlinks);
        no ``.git`` path component; every relative path passes
        rlvr.v3.identity.validate_relative_path (NFC, POSIX, normalized); with
        limits: entry count, per-file, total and UTF-8 path-byte caps apply;
        normalize_modes collapses root/dirs to 0755 and files to 0755 iff
        owner-exec else 0644, otherwise modes are untouched.  Failure raises
        TreeError and never mutates the tree.
    rlvr.v3.patch._walk_tree delegates to inspect_tree(limits=None,
        normalize_modes=not baseline): TreeError → PatchToolError for the
        baseline walk, → rejected PatchResult after apply.
"""

from __future__ import annotations

import os
import stat

import pytest

from tests.test_v3_patch import MODIFY, apply, assert_applied, assert_rejected, fake_git, make_workspace


def _mod():
    from rlvr.v3 import tree

    return tree


def limits(**over):
    base = dict(max_entries=64, max_file_bytes=1 << 20, max_total_file_bytes=4 << 20, max_path_bytes=256)
    return _mod().TreeLimits(**{**base, **over})


def inspect(root, lim=None, normalize=False):
    return _mod().inspect_tree(root, limits=lim, normalize_modes=normalize)


def fails(root, lim=None, normalize=False):
    with pytest.raises(_mod().TreeError):
        inspect(root, lim, normalize)


def make_tree(tmp_path, files=None):
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    if files is None:
        files = {"a.txt": b"abc", "sub/b.sh": b"#!/bin/sh\n", "sub/deep/c": b""}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def mode(path):
    return stat.S_IMODE(path.lstat().st_mode)


def modes(root):
    return {str(p.relative_to(root)): mode(p) for p in root.rglob("*")}


def test_report_lists_sorted_entries_and_file_bytes(tmp_path):
    m = _mod()
    root = make_tree(tmp_path)
    report = inspect(root)
    assert report == m.TreeReport(entries=("a.txt", "sub", "sub/b.sh", "sub/deep", "sub/deep/c"), file_bytes=13)
    with pytest.raises(Exception):
        report.file_bytes = 0
    assert inspect(make_tree(tmp_path / "e", {})) == m.TreeReport(entries=(), file_bytes=0)


def test_limits_are_frozen_positive_ints():
    lim = limits()
    assert (lim.max_entries, lim.max_path_bytes) == (64, 256)
    with pytest.raises(Exception):
        lim.max_entries = 1
    for bad in ({"max_entries": 0}, {"max_file_bytes": -1}, {"max_total_file_bytes": True}, {"max_path_bytes": "1"}):
        with pytest.raises(ValueError):
            limits(**bad)


@pytest.mark.parametrize("kind", ["missing", "file", "symlink-to-dir"])
def test_root_must_be_a_real_directory(tmp_path, kind):
    root = tmp_path / "root"
    if kind == "file":
        root.write_bytes(b"")
    elif kind == "symlink-to-dir":
        make_tree(tmp_path / "real")
        root.symlink_to(tmp_path / "real" / "root", target_is_directory=True)
    fails(root)


@pytest.mark.parametrize("kind", ["symlink", "dir-symlink", "hardlink", "fifo", "dot-git-dir", "nested-dot-git-file",
                                  "backslash", "nfd"])
def test_unsafe_entries_are_rejected_without_mutation(tmp_path, kind):
    root = make_tree(tmp_path)
    os.chmod(root / "a.txt", 0o600)
    if kind == "symlink":
        (root / "sub" / "link").symlink_to("/etc/passwd")
    elif kind == "dir-symlink":
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "hardlink":  # second link lives outside the tree; nlink is what matters
        os.link(root / "a.txt", tmp_path / "outside")
    elif kind == "fifo":
        os.mkfifo(root / "pipe")
    elif kind == "dot-git-dir":
        (root / ".git").mkdir()
    elif kind == "nested-dot-git-file":
        (root / "sub" / ".git").write_bytes(b"gitdir: x")
    elif kind == "backslash":
        (root / "a\\b").write_bytes(b"")
    elif kind == "nfd":
        (root / "é.txt").write_bytes(b"")
    before = modes(root)
    fails(root, normalize=True)
    assert modes(root) == before and mode(root / "a.txt") == 0o600


def test_no_limits_means_no_caps(tmp_path):
    root = make_tree(tmp_path, {f"d{i}/{'p' * 240}": b"x" * 2048 for i in range(70)})
    report = inspect(root, None)
    assert len(report.entries) == 140 and report.file_bytes == 70 * 2048


@pytest.mark.parametrize("field, ok, over", [
    ("max_entries", {f"f{i}": b"" for i in range(5)}, {f"f{i}": b"" for i in range(6)}),
    ("max_file_bytes", {"a": b"x" * 5}, {"a": b"x" * 6}),
    ("max_total_file_bytes", {"a": b"xx", "b": b"xxx"}, {"a": b"xxx", "b": b"xxx"}),
    ("max_path_bytes", {"é" + "a" * 3: b""}, {"é" + "a" * 4: b""}),  # 5 UTF-8 bytes vs 6
], ids=["entries", "file", "total", "path-bytes"])
def test_caps_are_exact_at_the_boundary(tmp_path, field, ok, over):
    lim = limits(**{field: 5})
    inspect(make_tree(tmp_path / "ok", ok), lim)
    fails(make_tree(tmp_path / "over", over), lim)


def test_entry_cap_counts_directories(tmp_path):
    root = make_tree(tmp_path, {"d1/d2/d3/f": b""})  # 3 dirs + 1 file
    inspect(root, limits(max_entries=4))
    fails(root, limits(max_entries=3))


def test_normalize_modes_collapses_exactly(tmp_path):
    root = make_tree(tmp_path)
    os.chmod(root, 0o700)
    os.chmod(root / "sub", 0o777)
    os.chmod(root / "sub" / "b.sh", 0o4700)  # setuid, owner-exec
    os.chmod(root / "a.txt", 0o666)
    os.chmod(root / "sub" / "deep" / "c", 0o2010)  # setgid, group-exec only
    inspect(root, normalize=True)
    assert mode(root) == 0o755
    assert modes(root) == {"a.txt": 0o644, "sub": 0o755, "sub/b.sh": 0o755, "sub/deep": 0o755, "sub/deep/c": 0o644}


def test_without_normalization_modes_are_untouched(tmp_path):
    root = make_tree(tmp_path)
    os.chmod(root / "a.txt", 0o600)
    os.chmod(root / "sub", 0o700)
    inspect(root, limits(), normalize=False)
    assert mode(root / "a.txt") == 0o600 and mode(root / "sub") == 0o700


def test_walk_tree_delegates_to_inspect_tree(tmp_path, monkeypatch):
    from rlvr.v3 import patch

    calls = []
    real = _mod().inspect_tree

    def spy(root, *, limits, normalize_modes):
        calls.append((root, limits, normalize_modes))
        if getattr(spy, "boom", False):
            raise _mod().TreeError("bad tree")
        return real(root, limits=limits, normalize_modes=normalize_modes)

    monkeypatch.setattr(patch, "inspect_tree", spy, raising=True)
    ws = make_workspace(tmp_path)
    assert patch._walk_tree(ws, baseline=True) is None
    assert patch._walk_tree(ws, baseline=False) is None
    assert [c[1:] for c in calls] == [(None, False), (None, True)] and all(c[0] == ws for c in calls)
    spy.boom = True
    with pytest.raises(patch.PatchToolError):
        patch._walk_tree(ws, baseline=True)
    assert isinstance(patch._walk_tree(ws, baseline=False), str)


def test_apply_patch_semantics_survive_delegation(tmp_path):
    from rlvr.v3 import patch

    ws = make_workspace(tmp_path)
    (ws / ".git").mkdir()
    with pytest.raises(patch.PatchToolError):
        apply(ws, MODIFY)
    (ws / ".git").rmdir()
    git = fake_git(tmp_path, version="2.0.0", apply_cmd="/bin/ln -s /etc/passwd link")
    assert_rejected(apply(ws, MODIFY, git=str(git), version="2.0.0"))
    (ws / "link").unlink()
    git = fake_git(tmp_path, version="2.0.0", apply_cmd="/bin/chmod 4755 run.sh; /bin/chmod 0600 f.txt")
    assert_applied(apply(ws, MODIFY, git=str(git), version="2.0.0"))
    assert mode(ws / "run.sh") == 0o755 and mode(ws / "f.txt") == 0o644
