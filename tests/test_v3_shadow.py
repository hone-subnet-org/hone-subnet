"""A candidate may not add files with reserved Python module names."""

from __future__ import annotations

import os
import sys

import pytest

from rlvr.v3.reasons import MinerReason, Stage
from rlvr.v3.shadow import SHADOWED_NAMES, shadowing_files, shadowing_reason, tree_files
from tests.test_v3_grading import eval_repo, eval_term, result, runner  # noqa: F401  (fixture)

REPO = {"README.md", "src/app.py", "pkg/__init__.py", "pkg/core.py", "cmd/main.go", "tests/test_app.py"}


def added(*paths):
    return shadowing_files(REPO, REPO | set(paths))


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        "subprocess.py",
        "unittest/__init__.py",
        "sitecustomize.py",
        "usercustomize.py",
        "evil.pth",
        "src/subprocess.py",  # beside a script or a working directory
        "tests/json.py",
        "subprocess.pyc",
        "subprocess.cpython-312-x86_64-linux-gnu.so",
        "pytest.py",
        "_pytest/__init__.py",
        "cmd/__init__.py",  # an existing plain directory turned into a package
        "pkg/tools/os.py",
        "pkg/types.py",  # even inside a package: a check may run from pkg/
        "pkg/json/__init__.py",
    ],
)
def test_an_added_file_that_would_shadow_a_library_is_found(path):
    assert added(path) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "src/solver.py",
        "unittest/case.py",  # a namespace directory loses to the real package
        "cmd/server/main.go",
        "json.txt",
        "test/__init__.py",  # the stdlib's own test suite is never imported by a runner
        "__pycache__/app.cpython-312.pyc",
    ],
)
def test_an_ordinary_added_file_is_not_found(path):
    assert added(path) == []


@pytest.mark.parametrize("path", ["iniconfig.py", "packaging/__init__.py", "typing_extensions.py"])
def test_what_test_tools_import_is_covered_too(path):
    assert added(path) == [path]


@pytest.mark.parametrize("link", ["unittest/", "src/subprocess.py/", "lib/pytest/", "unittest/__init__.py/", "evil.pth/"])
def test_a_link_to_a_directory_named_like_a_library_is_found(link):
    assert added(link) == [link]


def test_a_file_the_repository_already_ships_is_graded_as_it_is():
    repo = REPO | {"types.py", "logging/__init__.py"}
    assert shadowing_files(repo, repo) == []


def test_findings_are_sorted_and_the_reason_is_bounded():
    paths = [f"d{i}/os.py" for i in range(5)]
    found = added(*reversed(paths))
    assert found == sorted(paths)
    reason = shadowing_reason(found)
    assert "d0/os.py, d1/os.py, d2/os.py and 2 more" in reason and len(reason) <= 200


def test_the_names_are_fixed_not_read_from_the_running_interpreter():
    assert "subprocess" in SHADOWED_NAMES and "unittest" in SHADOWED_NAMES and "test" not in SHADOWED_NAMES
    assert {"pytest", "_pytest", "pip", "sitecustomize"} <= SHADOWED_NAMES
    if sys.version_info[:2] == (3, 12):
        assert set(sys.stdlib_module_names) - {"test"} <= SHADOWED_NAMES


def test_tree_files_lists_files_and_directory_links(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.py").write_text("")
    (tmp_path / "y.py").write_text("")
    (tmp_path / "unittest").symlink_to(tmp_path / "a")
    assert tree_files(tmp_path) == {"a/b/x.py", "y.py", "unittest/"}


def test_a_dangling_link_is_listed_as_a_link(tmp_path):
    (tmp_path / "unittest").symlink_to("/work/out/payload")  # resolves only inside the container
    assert tree_files(tmp_path) == {"unittest/"}
    assert shadowing_files(set(), tree_files(tmp_path)) == ["unittest/"]


def test_an_unreadable_directory_fails_the_scan(tmp_path):
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked").chmod(0)
    try:
        if os.access(tmp_path / "locked", os.R_OK):
            pytest.skip("running with privileges that ignore permissions")
        with pytest.raises(OSError):
            tree_files(tmp_path)
    finally:
        (tmp_path / "locked").chmod(0o755)


def test_a_terminal_script_linking_a_library_name_outside_the_result_tree_fails(tmp_path, runner):
    def script(request):
        (tmp_path / "env" / "out" / "payload").mkdir()
        (tmp_path / "env" / "out" / "payload" / "__init__.py").write_text("")
        (tmp_path / "env" / "unittest").symlink_to(tmp_path / "env" / "out" / "payload")
        return result()

    runner.responses = [script]
    verdict = eval_term(tmp_path, runner)
    assert verdict.status == "failed" and "unittest/" in verdict.reason


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
SHADOW_PATCH = (
    b"diff --git a/subprocess.py b/subprocess.py\n"
    b"new file mode 100644\n"
    b"--- /dev/null\n"
    b"+++ b/subprocess.py\n"
    b"@@ -0,0 +1 @@\n"
    b"+run = None\n"
)


def test_a_patch_adding_a_shadowing_file_is_rejected_before_anything_runs(tmp_path, runner):
    verdict = eval_repo(tmp_path, runner, patch=SHADOW_PATCH)
    assert verdict.status == "rejected" and "subprocess.py" in verdict.reason
    assert verdict.reason_code is MinerReason.PATCH_STATIC_REJECTED and verdict.stage is Stage.PATCH
    assert runner.requests == []  # neither setup nor any check ran


def test_setup_output_is_scanned_because_setup_can_run_patched_code(tmp_path, runner):
    def setup(request):
        (tmp_path / "ws" / "subprocess.py").write_text("")
        return result()

    runner.responses = [setup, result(), result()]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed" and "subprocess.py" in verdict.reason
    assert [c.outcome for c in verdict.checks] == ["skipped", "skipped"]


def test_ordinary_setup_output_is_fine(tmp_path, runner):
    def setup(request):
        (tmp_path / "ws" / "__pycache__").mkdir()
        (tmp_path / "ws" / "__pycache__" / "f.cpython-312.pyc").write_text("")
        return result()

    runner.responses = [setup, result(), result()]
    assert eval_repo(tmp_path, runner).status == "passed"


def test_a_long_offending_path_still_gives_a_bounded_rejection(tmp_path, runner):
    name = "d" * 180
    patch = SHADOW_PATCH.replace(b"subprocess.py", name.encode() + b"/subprocess.py")
    verdict = eval_repo(tmp_path, runner, patch=patch)
    assert verdict.status == "rejected" and len(verdict.reason) <= 200


def test_a_file_written_while_a_check_runs_stops_the_next_check(tmp_path, runner):
    def first_check(request):
        (tmp_path / "ws" / "subprocess.py").write_text("")
        return result()

    runner.responses = [result(), first_check, result()]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed" and "subprocess.py" in verdict.reason
    assert verdict.reason_code is MinerReason.RESULT_TREE_INVALID
    assert [c.outcome for c in verdict.checks] == ["passed", "skipped"]
    assert len(runner.requests) == 2  # the second check never ran


def test_a_file_written_during_the_last_check_fails_it(tmp_path, runner):
    def last_check(request):
        (tmp_path / "ws" / "subprocess.py").write_text("")
        return result()

    runner.responses = [result(), result(), last_check]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed" and "subprocess.py" in verdict.reason
    assert [c.outcome for c in verdict.checks] == ["passed", "failed"]


def test_a_script_that_writes_a_shadowing_file_fails_before_the_checks(tmp_path, runner):
    def script(request):
        (tmp_path / "env" / "out" / "unittest").mkdir()
        (tmp_path / "env" / "out" / "unittest" / "__init__.py").write_text("")
        return result()

    runner.responses = [script]
    verdict = eval_term(tmp_path, runner)
    assert verdict.status == "failed" and "out/unittest/__init__.py" in verdict.reason
    assert verdict.reason_code is MinerReason.RESULT_TREE_INVALID and verdict.stage is Stage.RESULT_TREE
    assert [c.outcome for c in verdict.checks] == ["skipped", "skipped"]
    assert len(runner.requests) == 1  # only the script ran


def test_an_ordinary_script_is_graded_as_before(tmp_path, runner):
    assert eval_term(tmp_path, runner).status == "passed"
