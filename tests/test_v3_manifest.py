"""Contract for ``rlvr.v3.manifest`` — the validator-internal manifest inside
the verifier archive (not the deferred wire manifest).

ManifestError(ValueError); frozen dataclasses Expectation, SetupStep,
InvocationCheck, InspectionCheck, VerifierManifest (see golden test).
Rules: manifest_version == 1; task_type matches caller; setup may be null for
either task type; 1..64 checks, unique ids ^[a-z0-9][a-z0-9_-]{0,31}$; argv
1..64 strings of 1..4096 bytes, no NUL, argv[0] absolute+normalized (inspection
additionally forbids /result and /verify); timeouts 1..300; byte caps 1..8 MiB;
exit_code 0..255; cwd relative, stdin "inputs/...", stdout/stderr "gold/...".
parse_manifest(raw, *, task_type): strict JSON, exact key sets, no coercion.
load_manifest(dir, *, task_type): manifest.json regular non-symlink <= 1 MiB;
top level limited to manifest.json, checks/, inputs/, gold/ (real dirs); refs
resolve to regular non-symlink files; gold larger than its cap is rejected.
"""

from __future__ import annotations

import copy
import json
import os
import shutil

import pytest

REPO, TERM, MIB8, PY = "repository_patch_v1", "terminal_script_v1", 8 << 20, "/usr/bin/python3"
SETUP = {"argv": [PY, "-m", "compileall", "-q", "."], "timeout_s": 60, "max_output_bytes": 65536}

def _mod():
    from rlvr.v3 import manifest
    return manifest

def expect(stdout="gold/c01.stdout", stderr=None, exit_code=0):
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}

def invocation(check_id="c01", **over):
    return {"check_id": check_id, "kind": "invocation", "argv": [PY, "-I", "/work/greet.py"], "cwd": ".",
            "stdin": "inputs/c01.stdin", "timeout_s": 10, "max_stdout_bytes": 65536,
            "max_stderr_bytes": 65536, "expect": expect(), **over}

def inspection(check_id="c02", **over):
    return {"check_id": check_id, "kind": "inspection", "argv": [PY, "-I", "/verify/c02.py"], "timeout_s": 10,
            "max_stdout_bytes": 65536, "max_stderr_bytes": 65536, "expect": expect(stdout="gold/c02.stdout"), **over}

def repo_manifest(**over):
    return {"manifest_version": 1, "task_type": REPO, "setup": copy.deepcopy(SETUP),
            "checks": [invocation(), inspection()], **over}

def term_manifest(**over):
    return {"manifest_version": 1, "task_type": TERM, "setup": None,
            "checks": [inspection("c01", expect=expect()), inspection("c02")], **over}

def dump(doc) -> bytes:
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")

def parse(doc, task_type=None):
    raw = doc if isinstance(doc, bytes) else dump(doc)
    return _mod().parse_manifest(raw, task_type=task_type or (doc["task_type"] if isinstance(doc, dict) else REPO))

def rejected(doc, task_type=None):
    with pytest.raises(_mod().ManifestError):
        parse(doc, task_type)

def with_check(doc, index, **over):
    doc = copy.deepcopy(doc)
    doc["checks"][index].update(over)
    return doc

def without(doc, key, *path):
    doc = copy.deepcopy(doc)
    node = doc
    for step in path:
        node = node[step]
    del node[key]
    return doc

# Builders applying overrides to the setup step / invocation check / inspection check.
TARGETS = {"setup": lambda **o: repo_manifest(setup={**SETUP, **o}), "invocation": lambda **o: with_check(repo_manifest(), 0, **o),
           "inspection": lambda **o: with_check(repo_manifest(), 1, **o)}

def test_repository_manifest_parses_to_exact_frozen_dataclasses():
    m = _mod()
    parsed = parse(repo_manifest())
    assert parsed == m.VerifierManifest(
        manifest_version=1, task_type=REPO,
        setup=m.SetupStep(argv=(PY, "-m", "compileall", "-q", "."), timeout_s=60, max_output_bytes=65536),
        checks=(
            m.InvocationCheck(check_id="c01", argv=(PY, "-I", "/work/greet.py"), cwd=".", stdin="inputs/c01.stdin",
                              timeout_s=10, max_stdout_bytes=65536, max_stderr_bytes=65536,
                              expect=m.Expectation(exit_code=0, stdout="gold/c01.stdout", stderr=None)),
            m.InspectionCheck(check_id="c02", argv=(PY, "-I", "/verify/c02.py"), timeout_s=10,
                              max_stdout_bytes=65536, max_stderr_bytes=65536,
                              expect=m.Expectation(exit_code=0, stdout="gold/c02.stdout", stderr=None))))
    with pytest.raises(Exception):
        parsed.checks[0].timeout_s = 1

def test_terminal_manifest_parses_and_setup_is_optional_for_both_task_types():
    parsed = parse(term_manifest())
    assert parsed.setup is None and [c.check_id for c in parsed.checks] == ["c01", "c02"]
    assert parse(repo_manifest(setup=None)).setup is None
    assert parse(term_manifest(setup=SETUP)).setup.timeout_s == 60

@pytest.mark.parametrize("raw", [
    b"", b"\xef\xbb\xbf" + dump(repo_manifest()), b"[]", dump(repo_manifest())[:-1],
    b'{"manifest_version": 1.0, "task_type": "%s", "setup": null, "checks": []}' % REPO.encode(),
    b'{"manifest_version": 1, "manifest_version": 1, "task_type": "%s", "setup": null, "checks": []}' % REPO.encode(),
    dump(with_check(repo_manifest(), 0, cwd="é")),
], ids=["empty", "bom", "array", "truncated", "float", "dup-key", "nfd"])
def test_non_strict_json_is_a_manifest_error(raw):
    rejected(raw, REPO)

def test_task_type_and_version_are_exact():
    rejected(repo_manifest(), TERM)
    rejected(term_manifest(), REPO)
    rejected(repo_manifest(task_type="other"), "other")
    for version in (2, "1", True):
        rejected(repo_manifest(manifest_version=version))

def test_key_sets_are_exact():
    for doc in (
        without(repo_manifest(), "setup"), repo_manifest(env={}), repo_manifest(setup=[]),
        TARGETS["setup"](cwd="."), without(repo_manifest(), "max_output_bytes", "setup"),
        with_check(repo_manifest(), 0, env={}), without(repo_manifest(), "stdin", "checks", 0),
        with_check(repo_manifest(), 1, cwd="."), with_check(repo_manifest(), 1, stdin=None),
        without(repo_manifest(), "stderr", "checks", 0, "expect"),
        with_check(repo_manifest(), 0, expect={**expect(), "note": ""}),
        with_check(repo_manifest(), 0, kind="Invocation"),
    ):
        rejected(doc)

def test_check_list_and_ids_are_bounded_and_unique():
    for checks in ([], [invocation()] * 2, [inspection(f"c{i:02}") for i in range(65)],
                   [invocation("")], [invocation("-a")], [invocation("A1")], [invocation("a" * 33)], [invocation("a.b")]):
        rejected(repo_manifest(checks=checks))
    ids = ["0", "a" * 32, "a-b_c9"] + [f"c{i:02}" for i in range(61)]
    assert [c.check_id for c in parse(repo_manifest(checks=[inspection(i) for i in ids])).checks] == ids

@pytest.mark.parametrize("build", TARGETS.values(), ids=TARGETS.keys())
def test_argv_common_rules(build):
    for argv in ([], [PY] * 65, ["python3"], ["/usr/bin/../bin/python3"], ["/usr/bin/"], ["//usr/bin/python3"],
                 [PY, ""], [PY, "a\0b"], [PY, "x" * 4097], [PY, 1], PY):
        rejected(build(argv=argv))
    parse(build(argv=[PY] + ["x" * 4096] * 63))
    parse(build(argv=[PY, "/result/out", "/verify/x", "/work"]))  # reserved roots only matter for argv[0]

def test_only_inspection_argv0_is_kept_out_of_result_and_verify():
    for argv0, ok in (("/result/x", False), ("/verify/x", False), ("/verify", False), ("/verifyx/x", True), ("/work/x", True)):
        (parse if ok else rejected)(with_check(repo_manifest(), 1, argv=[argv0]))
        parse(TARGETS["setup"](argv=[argv0]))
        parse(with_check(repo_manifest(), 0, argv=[argv0]))

@pytest.mark.parametrize("target, field, high", [
    ("setup", "timeout_s", 300), ("setup", "max_output_bytes", MIB8), ("invocation", "timeout_s", 300),
    ("inspection", "max_stdout_bytes", MIB8), ("invocation", "max_stderr_bytes", MIB8),
], ids=["setup-timeout", "setup-output", "check-timeout", "check-stdout", "check-stderr"])
def test_numeric_boundaries(target, field, high):
    build = TARGETS[target]
    parse(build(**{field: 1}))
    parse(build(**{field: high}))
    for bad in (0, high + 1, True, "1", None):
        rejected(build(**{field: bad}))

def test_exit_code_boundaries():
    for value, ok in ((0, True), (255, True), (-1, False), (256, False), (False, False), ("0", False)):
        (parse if ok else rejected)(with_check(repo_manifest(), 0, expect=expect(exit_code=value)))

@pytest.mark.parametrize("field, cases", [
    ("cwd", [("src/a", True), ("/work", False), ("..", False), ("a/../b", False), ("", False), ("./a", False),
             ("a/", False), (None, False)]),
    ("stdin", [(None, True), ("inputs/n/a", True), ("gold/a", False), ("inputs", False), ("/inputs/a", False),
               ("inputs/../a", False)]),
])
def test_invocation_cwd_and_stdin_paths(field, cases):
    for value, ok in cases:
        (parse if ok else rejected)(with_check(repo_manifest(), 0, **{field: value}))

def test_expect_gold_paths():
    for field, value, ok in (("stdout", "gold/n/a", True), ("stdout", None, False), ("stdout", "inputs/a", False),
                             ("stdout", "gold", False), ("stdout", "/gold/a", False), ("stdout", "gold//a", False),
                             ("stderr", "gold/c01.stdout", True), ("stderr", "gold/a\0", False), ("stderr", 1, False)):
        (parse if ok else rejected)(with_check(repo_manifest(), 0, expect=expect(**{field: value})))

# -------------------------------------------------------- filesystem binding

def build_dir(tmp_path, doc):
    root = tmp_path / "verifier"
    root.mkdir()
    (root / "manifest.json").write_bytes(dump(doc))
    for check in doc["checks"]:
        for rel in filter(None, [check.get("stdin"), check["expect"]["stdout"], check["expect"]["stderr"]]):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"x\n")
        for argument in check["argv"][1:]:
            if argument.startswith("/verify/"):
                path = root / "checks" / argument[len("/verify/") :]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x\n")
    (root / "checks").mkdir(exist_ok=True)
    return root

def load(root, ok=True, task_type=REPO):
    if ok:
        return _mod().load_manifest(root, task_type=task_type)
    with pytest.raises(_mod().ManifestError):
        _mod().load_manifest(root, task_type=task_type)

def test_load_matches_parse_and_tolerates_missing_optional_dirs(tmp_path):
    document = repo_manifest(checks=[invocation()])
    root = build_dir(tmp_path, document)
    assert load(root) == load(str(root)) == parse(document)
    load(root, False, TERM)
    shutil.rmtree(root / "checks")
    load(root)

def test_load_manifest_must_be_a_regular_file_of_at_most_one_mib(tmp_path):
    root, raw = build_dir(tmp_path, repo_manifest()), dump(repo_manifest())
    (root / "manifest.json").write_bytes(raw + b" " * ((1 << 20) - len(raw)))
    load(root)
    (root / "manifest.json").write_bytes(raw + b" " * ((1 << 20) - len(raw) + 1))
    load(root, False)
    (root / "manifest.json").unlink()
    load(root, False)
    (tmp_path / "manifest.json").write_bytes(raw)
    os.symlink(tmp_path / "manifest.json", root / "manifest.json")
    load(root, False)
    load(tmp_path / "absent", False)

@pytest.mark.parametrize("entry", ["README", "extra/", "Manifest.json", "gold@", "inputs!"])
def test_load_top_level_allowlist(tmp_path, entry):
    root, name = build_dir(tmp_path, repo_manifest()), entry[:-1]
    if entry.endswith("/"):
        (root / name).mkdir()
    elif entry.endswith("@"):  # allowed name, but a symlink to a directory
        shutil.move(root / name, tmp_path / "real")
        os.symlink(tmp_path / "real", root / name)
    elif entry.endswith("!"):  # allowed name, but a regular file
        shutil.rmtree(root / name)
        (root / name).write_bytes(b"")
    else:
        (root / entry).write_bytes(b"")
    load(root, False)

@pytest.mark.parametrize("rel", ["inputs/c01.stdin", "gold/c01.stdout", "gold/c02.stdout"])
def test_load_references_must_be_existing_regular_non_symlink_files(tmp_path, rel):
    root = build_dir(tmp_path, repo_manifest())
    path = root / rel
    path.unlink()
    load(root, False)
    os.symlink(root / "manifest.json", path)
    load(root, False)
    path.unlink()
    path.mkdir()
    load(root, False)
    path.rmdir()
    path.write_bytes(b"")
    load(root)
    shutil.move(path.parent, tmp_path / "elsewhere")  # symlinked parent directory
    os.symlink(tmp_path / "elsewhere", path.parent)
    load(root, False)


def test_load_requires_direct_trusted_checker_references(tmp_path):
    root = build_dir(tmp_path, repo_manifest())
    checker = root / "checks/c02.py"
    checker.unlink()
    load(root, False)

def test_load_rejects_gold_larger_than_its_own_check_cap(tmp_path):
    doc = with_check(repo_manifest(), 0, max_stdout_bytes=4, expect=expect(stderr="gold/e"))
    doc = with_check(doc, 1, max_stderr_bytes=2, expect=expect(stdout="gold/c01.stdout", stderr="gold/e"))
    root = build_dir(tmp_path, doc)
    (root / "gold/c01.stdout").write_bytes(b"abcd")  # fits c01's stdout cap of 4 (c02's is 65536)
    (root / "gold/e").write_bytes(b"ab")
    load(root)
    (root / "gold/e").write_bytes(b"abc")  # exceeds c02's stderr cap of 2
    load(root, False)
