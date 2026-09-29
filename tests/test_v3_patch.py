"""V3 ``unified_diff_v1`` validation and application.

Contract for module ``rlvr.v3.patch``:

    PatchToolError(RuntimeError)   infrastructure: missing or wrong-version git,
                                   spawn failure, timeout, apply failing after a
                                   clean check, unusable workspace or baseline
    PatchLimits(max_patch_bytes, git_timeout_s)      frozen, every field required
    PatchResult(status, reason)    status "applied" (reason exactly "") or
                                   "rejected" (reason non-empty, <= 200 chars)
    apply_patch(workspace, patch: bytes, limits, *, git_binary,
                expected_git_version) -> PatchResult

Static checks on the raw bytes only: size cap, zero bytes is a valid no-op that
never invokes git, NUL, invalid UTF-8, binary-patch markers, and any mode line
outside 100644/100755. Then exactly ``git apply --check -p1 FILE`` followed by
``git apply -p1 FILE`` in the workspace with a clean environment, no retries
and no extra flags. The whole tree is walked afterwards: normalized safe
relative paths, no ``.git`` segment, no backslash, regular files and
directories only, no links, modes collapsed to 0644/0755. ``expected_git_version``
is compared with the version token of ``git --version``.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git not installed")


def git_version() -> str:
    out = subprocess.run([GIT, "--version"], capture_output=True, text=True, check=True).stdout
    return out.split()[2]


def _mod():
    from rlvr.v3 import patch

    return patch


def _limits(**over):
    base = dict(max_patch_bytes=1 << 20, git_timeout_s=10)
    base.update(over)
    return _mod().PatchLimits(**base)


BASE_FILES = {
    "f.txt": b"line one\nline two\nline three\n",
    "sub/g.txt": b"x\n",
    "run.sh": b"#!/bin/sh\n",
}


def make_workspace(root: Path) -> Path:
    ws = root / "ws"
    for rel, data in BASE_FILES.items():
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    os.chmod(ws / "run.sh", 0o755)
    return ws


def snapshot(ws: Path) -> dict[str, bytes]:
    return {str(p.relative_to(ws)): p.read_bytes() for p in sorted(ws.rglob("*")) if p.is_file()}


MODIFY = (
    b"diff --git a/f.txt b/f.txt\n"
    b"--- a/f.txt\n"
    b"+++ b/f.txt\n"
    b"@@ -1,3 +1,3 @@\n"
    b" line one\n"
    b"-line two\n"
    b"+line TWO \xc3\xa9\n"
    b" line three\n"
)


def apply(ws, patch, limits=None, *, git=None, version=None):
    m = _mod()
    return m.apply_patch(
        ws,
        patch,
        limits or _limits(),
        git_binary=git or GIT,
        expected_git_version=version or git_version(),
    )


def assert_rejected(result):
    m = _mod()
    assert isinstance(result, m.PatchResult)
    assert result.status == "rejected"
    assert 0 < len(result.reason) <= 200
    return result


def assert_applied(result):
    assert result.status == "applied"
    assert result.reason == ""
    return result


def fake_git(root: Path, *, version="0.0.0", check_rc=0, apply_rc=0, check_cmd="", apply_cmd="") -> Path:
    """A stand-in git that logs every argv line to root/log and behaves as configured.
    The clean environment has no PATH beyond the git directory, so helpers use absolute paths."""
    log = root / "log"
    script = root / "git"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        'case "$1" in\n'
        f'  --version) echo "git version {version}"; exit 0;;\n'
        "  apply)\n"
        '    case "$2" in\n'
        f"      --check) {check_cmd}\n        exit {check_rc};;\n"
        f"      *) {apply_cmd}\n        exit {apply_rc};;\n"
        "    esac;;\n"
        "esac\n"
        "exit 2\n"
    )
    script.chmod(0o755)
    return script


def argv_lines(root: Path) -> list[list[str]]:
    return [line.split() for line in (root / "log").read_text().splitlines()]


# --------------------------------------------------------------------------- #
# Static byte checks
# --------------------------------------------------------------------------- #
def test_size_cap_counts_bytes_at_boundary(tmp_path):
    ws = make_workspace(tmp_path)
    assert_rejected(apply(ws, MODIFY, _limits(max_patch_bytes=len(MODIFY) - 1)))
    assert snapshot(ws) == BASE_FILES
    assert_applied(apply(ws, MODIFY, _limits(max_patch_bytes=len(MODIFY))))


def test_empty_patch_is_a_no_op_that_never_invokes_git(tmp_path):
    ws = make_workspace(tmp_path)
    result = apply(ws, b"", git=str(tmp_path / "no-such-git"), version="0.0.0")
    assert_applied(result)
    assert snapshot(ws) == BASE_FILES
    assert stat.S_IMODE((ws / "run.sh").stat().st_mode) == 0o755


def test_whitespace_only_patch_is_rejected(tmp_path):
    ws = make_workspace(tmp_path)
    assert_rejected(apply(ws, b"\n \n\t\n"))
    assert snapshot(ws) == BASE_FILES


@pytest.mark.parametrize(
    "patch",
    [
        pytest.param(MODIFY.replace(b"TWO", b"\xff\xfe"), id="invalid-utf8"),
        pytest.param(MODIFY.replace(b"TWO", b"T\x00O"), id="nul"),
        pytest.param(b"diff --git a/bin b/bin\nnew file mode 100644\nGIT binary patch\nliteral 4\nLcmZQzWMT#Y01f~L\n\n", id="git-binary-patch"),
        pytest.param(b"diff --git a/bin b/bin\nBinary files /dev/null and b/bin differ\n", id="binary-files-differ"),
    ],
)
def test_undecodable_or_binary_patches_are_rejected(tmp_path, patch):
    ws = make_workspace(tmp_path)
    assert_rejected(apply(ws, patch))
    assert snapshot(ws) == BASE_FILES


@pytest.mark.parametrize(
    "mode_lines",
    [
        pytest.param(b"new file mode 120000\n", id="symlink-120000"),
        pytest.param(b"new file mode 160000\n", id="submodule-160000"),
        pytest.param(b"new file mode 104755\n", id="setuid-104755"),
        pytest.param(b"old mode 100644\nnew mode 100600\n", id="new-mode-100600"),
    ],
)
def test_special_modes_are_rejected_before_git(tmp_path, mode_lines):
    ws = make_workspace(tmp_path)
    patch = b"diff --git a/thing b/thing\n" + mode_lines + b"--- /dev/null\n+++ b/thing\n@@ -0,0 +1 @@\n+x\n"
    assert_rejected(apply(ws, patch))
    assert snapshot(ws) == BASE_FILES


# --------------------------------------------------------------------------- #
# Real git behaviour
# --------------------------------------------------------------------------- #
def test_modify_applies_exact_content(tmp_path):
    ws = make_workspace(tmp_path)
    assert_applied(apply(ws, MODIFY))
    assert (ws / "f.txt").read_bytes() == b"line one\nline TWO \xc3\xa9\nline three\n"
    assert (ws / "sub" / "g.txt").read_bytes() == b"x\n"


def test_add_delete_and_mode(tmp_path):
    ws = make_workspace(tmp_path)
    patch = (
        b"diff --git a/new.txt b/new.txt\nnew file mode 100755\n--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+new\n"
        b"diff --git a/sub/g.txt b/sub/g.txt\ndeleted file mode 100644\n--- a/sub/g.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        b"diff --git a/f.txt b/f.txt\nold mode 100644\nnew mode 100755\n"
    )
    assert_applied(apply(ws, patch))
    assert (ws / "new.txt").read_bytes() == b"new\n"
    assert not (ws / "sub" / "g.txt").exists()
    mode = lambda rel: stat.S_IMODE((ws / rel).stat().st_mode)
    assert mode("new.txt") == 0o755 and mode("f.txt") == 0o755
    assert not (ws / "sub").exists()  # git prunes the emptied directory


def test_copy_and_rename_headers_are_rejected(tmp_path):
    ws = make_workspace(tmp_path)
    patch = (
        b"diff --git a/run.sh b/tool.sh\n"
        b"similarity index 100%\n"
        b"rename from run.sh\n"
        b"rename to tool.sh\n"
    )
    assert apply(ws, patch).status == "rejected"


@pytest.mark.parametrize(
    "header",
    [b"copy old run.sh", b"copy new tool.sh", b"rename old run.sh", b"rename new tool.sh"],
)
def test_legacy_copy_and_rename_headers_are_rejected(tmp_path, header):
    ws = make_workspace(tmp_path)
    assert apply(ws, header + b"\n").status == "rejected"


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(b"../escape.txt", id="parent-traversal"),
        pytest.param(b".git/config", id="dot-git"),
        pytest.param(b"sub/.git/hooks", id="nested-dot-git"),
    ],
)
def test_git_rejects_unsafe_paths_and_tree_is_untouched(tmp_path, path):
    ws = make_workspace(tmp_path)
    patch = (
        b"diff --git a/" + path + b" b/" + path + b"\nnew file mode 100644\n--- /dev/null\n+++ b/" + path
        + b"\n@@ -0,0 +1 @@\n+pwned\n"
    )
    assert_rejected(apply(ws, patch))
    assert snapshot(ws) == BASE_FILES
    assert not (tmp_path / "escape.txt").exists()


def test_context_mismatch_is_rejected_without_fuzz(tmp_path):
    ws = make_workspace(tmp_path)
    assert_rejected(apply(ws, MODIFY.replace(b" line one\n", b" line one!\n")))
    assert snapshot(ws) == BASE_FILES


def test_crlf_patch_against_lf_file_is_rejected(tmp_path):
    ws = make_workspace(tmp_path)
    assert_rejected(apply(ws, MODIFY.replace(b"\n", b"\r\n")))
    assert snapshot(ws) == BASE_FILES


# --------------------------------------------------------------------------- #
# Isolation from operator configuration and surrounding repositories
# --------------------------------------------------------------------------- #
def test_operator_git_config_cannot_change_the_verdict(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[apply]\n\tignoreWhitespace = change\n\twhitespace = fix\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    ws = make_workspace(tmp_path)
    whitespace_mismatch = MODIFY.replace(b" line one\n", b" line one   \n")
    assert_rejected(apply(ws, whitespace_mismatch))
    assert snapshot(ws) == BASE_FILES


def test_workspace_nested_in_a_foreign_repository_applies_every_path(tmp_path):
    subprocess.run([GIT, "init", "-q", str(tmp_path)], check=True)
    ws = make_workspace(tmp_path / "deep" / "er")
    patch = MODIFY + b"diff --git a/sub/g.txt b/sub/g.txt\n--- a/sub/g.txt\n+++ b/sub/g.txt\n@@ -1 +1 @@\n-x\n+y\n"
    assert_applied(apply(ws, patch))
    assert (ws / "sub" / "g.txt").read_bytes() == b"y\n"
    assert (ws / "f.txt").read_bytes().startswith(b"line one\nline TWO")
    assert not (tmp_path / "f.txt").exists()


# --------------------------------------------------------------------------- #
# Infrastructure failures
# --------------------------------------------------------------------------- #
def test_wrong_git_version_is_a_tool_error(tmp_path):
    m = _mod()
    ws = make_workspace(tmp_path)
    git = fake_git(tmp_path, version="9.9.9")
    with pytest.raises(m.PatchToolError):
        apply(ws, MODIFY, git=str(git), version="2.0.0")
    assert snapshot(ws) == BASE_FILES


def test_missing_git_binary_is_a_tool_error(tmp_path):
    m = _mod()
    ws = make_workspace(tmp_path)
    with pytest.raises(m.PatchToolError):
        apply(ws, MODIFY, git=str(tmp_path / "no-such-git"), version="2.0.0")


def test_timeout_is_a_tool_error(tmp_path):
    m = _mod()
    ws = make_workspace(tmp_path)
    git = fake_git(tmp_path, version="2.0.0", check_cmd="/bin/sleep 5")
    with pytest.raises(m.PatchToolError):
        apply(ws, MODIFY, _limits(git_timeout_s=1), git=str(git), version="2.0.0")


def test_apply_failure_after_clean_check_is_a_tool_error_and_commands_are_exact(tmp_path):
    m = _mod()
    ws = make_workspace(tmp_path)
    git = fake_git(tmp_path, version="2.0.0", check_rc=0, apply_rc=1)
    with pytest.raises(m.PatchToolError):
        apply(ws, MODIFY, git=str(git), version="2.0.0")

    lines = argv_lines(tmp_path)
    assert lines[0] == ["--version"]
    check, real = lines[1], lines[2]
    assert check[:3] == ["apply", "--check", "-p1"] and len(check) == 4
    assert real[:2] == ["apply", "-p1"] and len(real) == 3
    assert check[3] == real[2]  # same patch file, no rewrite between check and apply
    assert len(lines) == 3  # no retries, no fallback strategy


def test_check_rejection_runs_exactly_one_git_apply(tmp_path):
    ws = make_workspace(tmp_path)
    git = fake_git(tmp_path, version="2.0.0", check_rc=1, check_cmd="echo 'error: patch does not apply' >&2")
    result = assert_rejected(apply(ws, MODIFY, git=str(git), version="2.0.0"))
    assert "does not apply" in result.reason
    assert [l[:2] for l in argv_lines(tmp_path)] == [["--version"], ["apply", "--check"]]


def test_long_git_error_is_bounded(tmp_path):
    ws = make_workspace(tmp_path)
    git = fake_git(tmp_path, version="2.0.0", check_rc=1, check_cmd="/usr/bin/head -c 5000 /dev/zero | /usr/bin/tr '\\0' 'e' >&2")
    assert_rejected(apply(ws, MODIFY, git=str(git), version="2.0.0"))


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_unusable_workspace_is_a_tool_error(tmp_path, kind):
    m = _mod()
    ws = tmp_path / "ws"
    if kind == "file":
        ws.write_text("not a directory")
    with pytest.raises(m.PatchToolError):
        apply(ws, MODIFY)


def test_baseline_containing_dot_git_is_a_tool_error(tmp_path):
    m = _mod()
    ws = make_workspace(tmp_path)
    (ws / ".git").mkdir()
    with pytest.raises(m.PatchToolError):
        apply(ws, MODIFY)


# --------------------------------------------------------------------------- #
# Post-apply tree walk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "apply_cmd",
    [
        pytest.param("/bin/ln -s /etc/passwd link", id="symlink"),
        pytest.param("/bin/ln f.txt hard", id="hardlink"),
        pytest.param("/usr/bin/touch 'dir\\\\name'", id="backslash-name"),
        pytest.param("/bin/mkdir -p sub/.git", id="dot-git-segment"),
        pytest.param("/usr/bin/mkfifo pipe", id="fifo"),
    ],
)
def test_post_apply_walk_rejects_unsafe_tree(tmp_path, apply_cmd):
    ws = make_workspace(tmp_path)
    git = fake_git(tmp_path, version="2.0.0", apply_cmd=apply_cmd)
    assert_rejected(apply(ws, MODIFY, git=str(git), version="2.0.0"))


def test_post_apply_walk_collapses_modes(tmp_path):
    ws = make_workspace(tmp_path)
    git = fake_git(tmp_path, version="2.0.0", apply_cmd="/bin/chmod 4755 run.sh; /bin/chmod 0600 f.txt; /bin/chmod 0700 sub")
    assert_applied(apply(ws, MODIFY, git=str(git), version="2.0.0"))
    mode = lambda rel: stat.S_IMODE((ws / rel).stat().st_mode)
    assert mode("run.sh") == 0o755 and mode("f.txt") == 0o644 and mode("sub") == 0o755
