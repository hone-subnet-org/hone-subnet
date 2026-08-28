"""V3 task identity models and canonical task_id.

Contract for module ``rlvr.v3.identity``:

    RepositoryTaskIdentity      schema_version Literal[1]=1,
                                task_type Literal["repository_patch_v1"]
                                (default), task_kind bug_fix|feature,
                                instruction 1..500000, primary_language 1..64,
                                workspace_sha256, verifier_sha256,
                                execution_profile_id (profile token),
                                working_directory (relative path rule),
                                patch_format Literal["unified_diff_v1"]
                                (default), verifier_policy (profile token),
                                authoring_version 1..128
    TerminalScriptTaskIdentity  schema_version Literal[1]=1,
                                task_type Literal["terminal_script_v1"]
                                (default), instruction 1..500000,
                                environment_sha256,
                                verifier_sha256, execution_profile_id,
                                result_tree_path (relative path rule),
                                verifier_policy, authoring_version 1..128
    TaskIdentity                discriminated union on "task_type"
    validate_relative_path(value: str) -> str
                                1..1024 characters, NFC, normalized relative
                                POSIX path; "." is the root; rejects absolute
                                paths, "..", backslashes, NUL, empty
                                segments, trailing slashes, and "./"
                                prefixes or "." segments
    TASK_ID_DOMAIN              b"hone-v3-task-id\\x00"
    compute_task_id(identity) -> str
                                sha256(TASK_ID_DOMAIN + canonical JSON of the
                                fully revalidated model, defaults included),
                                lowercase hex; TypeError for anything that is
                                not an identity model instance

Profile token pattern: ^[a-z0-9][a-z0-9._-]*$, 1..128 characters. Digests
are lowercase 64 hex. Byte and time caps are release policy and never appear
in identity. Runtime existence of paths is not validated here.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata

import pytest
from pydantic import TypeAdapter, ValidationError

HEX = "a" * 64
HEX2 = "b" * 64
HEX3 = "c" * 64
NFC_E = "é"
NFD_E = "é"
assert unicodedata.normalize("NFC", NFD_E) == NFC_E and NFD_E != NFC_E

DOMAIN = b"hone-v3-task-id\x00"

# Independently computed: sha256(DOMAIN + json.dumps(payload, sort_keys=True,
# separators=(",", ":"))) over the ASCII payloads below.
GOLDEN_REPO = "d4bdc18959cba781847e2d44cfee007b17f0389952f11c4e62c5db7a6bbe7c09"
GOLDEN_TERM = "8ecc3bf8cdd6163c3171906d9ee67ec84f3ea279133ea96b80630347a127ce24"


def _mod():
    from rlvr.v3 import identity

    return identity


def repo(**over):
    base = {
        "schema_version": 1,
        "task_type": "repository_patch_v1",
        "task_kind": "bug_fix",
        "instruction": "Fix the failing build.",
        "primary_language": "python",
        "workspace_sha256": HEX,
        "verifier_sha256": HEX2,
        "execution_profile_id": "repo-polyglot-v1",
        "working_directory": ".",
        "patch_format": "unified_diff_v1",
        "verifier_policy": "binary-pass-v1",
        "authoring_version": "factory-1",
    }
    base.update(over)
    return base


def term(**over):
    base = {
        "schema_version": 1,
        "task_type": "terminal_script_v1",
        "instruction": "Create the report tree.",
        "environment_sha256": HEX3,
        "verifier_sha256": HEX2,
        "execution_profile_id": "repo-polyglot-v1",
        "result_tree_path": "out",
        "verifier_policy": "binary-pass-v1",
        "authoring_version": "factory-1",
    }
    base.update(over)
    return base


def independent_task_id(payload: dict) -> str:
    """Reference computation kept separate from rlvr.v3.canonical. Valid for
    payloads whose keys are ASCII (sort_keys then equals UTF-16 order)."""
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(DOMAIN + body).hexdigest()


def _adapter():
    return TypeAdapter(_mod().TaskIdentity)


# --------------------------------------------------------------------------- #
# Valid models, defaults, and exact task_id
# --------------------------------------------------------------------------- #
def test_domain_prefix_is_exact():
    assert _mod().TASK_ID_DOMAIN == DOMAIN
    assert isinstance(_mod().TASK_ID_DOMAIN, bytes)


def test_repository_identity_round_trips():
    m = _mod()
    assert m.RepositoryTaskIdentity(**repo()).model_dump(mode="json") == repo()


def test_terminal_identity_round_trips():
    m = _mod()
    assert m.TerminalScriptTaskIdentity(**term()).model_dump(mode="json") == term()


def test_repository_defaults_are_filled_and_hashed():
    m = _mod()
    minimal = {
        k: v
        for k, v in repo().items()
        if k not in ("schema_version", "task_type", "patch_format")
    }
    model = m.RepositoryTaskIdentity(**minimal)
    assert model.model_dump(mode="json") == repo()
    assert m.compute_task_id(model) == m.compute_task_id(
        m.RepositoryTaskIdentity(**repo())
    )


def test_terminal_defaults_are_filled_and_hashed():
    m = _mod()
    minimal = {
        k: v for k, v in term().items() if k not in ("schema_version", "task_type")
    }
    model = m.TerminalScriptTaskIdentity(**minimal)
    assert model.model_dump(mode="json") == term()
    assert m.compute_task_id(model) == GOLDEN_TERM


def test_repository_task_id_matches_golden():
    m = _mod()
    task_id = m.compute_task_id(m.RepositoryTaskIdentity(**repo()))
    assert task_id == GOLDEN_REPO
    assert task_id == independent_task_id(repo())


def test_terminal_task_id_matches_golden():
    m = _mod()
    task_id = m.compute_task_id(m.TerminalScriptTaskIdentity(**term()))
    assert task_id == GOLDEN_TERM
    assert task_id == independent_task_id(term())


def test_task_id_is_lowercase_hex_sha256():
    m = _mod()
    task_id = m.compute_task_id(m.RepositoryTaskIdentity(**repo()))
    assert len(task_id) == 64
    assert task_id == task_id.lower()
    int(task_id, 16)


def test_task_id_uses_utf8_bytes_for_non_ascii_instruction():
    m = _mod()
    payload = repo(instruction="Fix " + NFC_E)
    assert m.compute_task_id(m.RepositoryTaskIdentity(**payload)) == (
        independent_task_id(payload)
    )


@pytest.mark.parametrize(
    "over",
    [
        {"task_kind": "feature"},
        {"instruction": "Fix the failing build!"},
        {"primary_language": "rust"},
        {"workspace_sha256": HEX3},
        {"verifier_sha256": HEX3},
        {"execution_profile_id": "repo-polyglot-v2"},
        {"working_directory": "src"},
        {"verifier_policy": "binary-pass-v2"},
        {"authoring_version": "factory-2"},
    ],
    ids=lambda o: next(iter(o)),
)
def test_every_repository_field_changes_task_id(over):
    m = _mod()
    base = m.compute_task_id(m.RepositoryTaskIdentity(**repo()))
    changed = m.compute_task_id(m.RepositoryTaskIdentity(**repo(**over)))
    assert changed != base
    assert changed == independent_task_id(repo(**over))


@pytest.mark.parametrize(
    "over",
    [
        {"instruction": "Create the report tree!"},
        {"environment_sha256": HEX},
        {"verifier_sha256": HEX3},
        {"execution_profile_id": "repo-polyglot-v2"},
        {"result_tree_path": "out/data"},
        {"verifier_policy": "binary-pass-v2"},
        {"authoring_version": "factory-2"},
    ],
    ids=lambda o: next(iter(o)),
)
def test_every_terminal_field_changes_task_id(over):
    m = _mod()
    base = m.compute_task_id(m.TerminalScriptTaskIdentity(**term()))
    changed = m.compute_task_id(m.TerminalScriptTaskIdentity(**term(**over)))
    assert changed != base
    assert changed == independent_task_id(term(**over))


def test_task_types_with_shared_values_hash_differently():
    m = _mod()
    a = m.compute_task_id(m.RepositoryTaskIdentity(**repo(instruction="same")))
    b = m.compute_task_id(m.TerminalScriptTaskIdentity(**term(instruction="same")))
    assert a != b


def test_task_id_is_deterministic_across_instances():
    m = _mod()
    a = m.compute_task_id(m.RepositoryTaskIdentity(**repo()))
    b = m.compute_task_id(m.RepositoryTaskIdentity(**dict(reversed(repo().items()))))
    assert a == b


# --------------------------------------------------------------------------- #
# Hashing revalidates: model_copy(update=...) cannot smuggle bad values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "update",
    [
        {"workspace_sha256": "Z" * 64},
        {"working_directory": "/abs"},
        {"execution_profile_id": "Bad Token"},
        {"instruction": ""},
        {"schema_version": 2},
        {"task_type": "terminal_script_v1"},
        {"instruction": NFD_E},
    ],
    ids=lambda u: next(iter(u)),
)
def test_compute_task_id_rejects_smuggled_repository_values(update):
    m = _mod()
    smuggled = m.RepositoryTaskIdentity(**repo()).model_copy(update=update)
    with pytest.raises(ValidationError):
        m.compute_task_id(smuggled)


def test_compute_task_id_rejects_smuggled_terminal_values():
    m = _mod()
    smuggled = m.TerminalScriptTaskIdentity(**term()).model_copy(
        update={"result_tree_path": "../out"}
    )
    with pytest.raises(ValidationError):
        m.compute_task_id(smuggled)


@pytest.mark.parametrize("value", [repo(), None, "repository_patch_v1", b"{}"], ids=["dict", "none", "str", "bytes"])
def test_compute_task_id_raises_type_error_for_non_identity_input(value):
    m = _mod()
    with pytest.raises(TypeError):
        m.compute_task_id(value)


def test_compute_task_id_raises_type_error_for_other_wire_models():
    from rlvr.v3.artifacts import ArtifactRef

    m = _mod()
    ref = ArtifactRef(
        artifact_role="verifier",
        artifact_format="tar_zst_v1",
        sha256=HEX,
        compressed_size_bytes=1,
        expanded_size_bytes=1,
    )
    with pytest.raises(TypeError):
        m.compute_task_id(ref)


# --------------------------------------------------------------------------- #
# Discriminated union
# --------------------------------------------------------------------------- #
def test_union_selects_branch_by_task_type():
    m = _mod()
    adapter = _adapter()
    assert isinstance(adapter.validate_python(repo()), m.RepositoryTaskIdentity)
    assert isinstance(adapter.validate_python(term()), m.TerminalScriptTaskIdentity)


def test_union_schema_is_discriminated_on_task_type():
    schema = _adapter().json_schema()
    mapping = schema["discriminator"]["mapping"]
    assert set(mapping) == {"repository_patch_v1", "terminal_script_v1"}
    assert schema["discriminator"]["propertyName"] == "task_type"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({**repo(), "task_type": "bogus_v1"}, id="unknown-tag"),
        pytest.param({k: v for k, v in repo().items() if k != "task_type"}, id="missing-tag"),
        pytest.param({**repo(), "task_type": 1}, id="int-tag"),
        pytest.param(repo(task_type="terminal_script_v1"), id="repo-fields-terminal-tag"),
        pytest.param(term(task_type="repository_patch_v1"), id="terminal-fields-repo-tag"),
        pytest.param(repo(result_tree_path="out"), id="repo-plus-terminal-field"),
        pytest.param(term(working_directory="."), id="terminal-plus-repo-field"),
        pytest.param(term(task_kind="bug_fix"), id="terminal-plus-task-kind"),
        pytest.param(term(patch_format="unified_diff_v1"), id="terminal-plus-patch-format"),
    ],
)
def test_union_rejects_cross_model_payloads(payload):
    with pytest.raises(ValidationError):
        _adapter().validate_python(payload)


def test_union_round_trips_through_strict_bytes():
    from rlvr.v3.canonical import canonical_json_bytes, parse_strict_json

    m = _mod()
    model = m.TerminalScriptTaskIdentity(**term())
    raw = canonical_json_bytes(model)
    again = _adapter().validate_python(parse_strict_json(raw))
    assert again == model
    assert m.compute_task_id(again) == GOLDEN_TERM


# --------------------------------------------------------------------------- #
# Inherited WireModel behavior
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, builder", [("RepositoryTaskIdentity", repo), ("TerminalScriptTaskIdentity", term)])
def test_identity_models_forbid_extra_and_are_frozen(name, builder):
    m = _mod()
    cls = getattr(m, name)
    with pytest.raises(ValidationError):
        cls(**builder(), extra_field=1)
    instance = cls(**builder())
    with pytest.raises(ValidationError):
        instance.instruction = "changed"


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"schema_version": "1"}, id="schema-str"),
        pytest.param({"schema_version": True}, id="schema-bool"),
        pytest.param({"schema_version": 1.0}, id="schema-float"),
        pytest.param({"instruction": b"x"}, id="instruction-bytes"),
        pytest.param({"workspace_sha256": None}, id="digest-none"),
    ],
)
def test_repository_identity_rejects_coercion(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.RepositoryTaskIdentity(**repo(**over))


# --------------------------------------------------------------------------- #
# Literal, bounds, digest, token, NFC rejection
# --------------------------------------------------------------------------- #
REPO_REJECTS = [
    pytest.param({"schema_version": 2}, id="schema-2"),
    pytest.param({"schema_version": 0}, id="schema-0"),
    pytest.param({"task_type": "repository_patch_v2"}, id="task-type-v2"),
    pytest.param({"task_kind": "chore"}, id="task-kind"),
    pytest.param({"task_kind": ""}, id="task-kind-empty"),
    pytest.param({"patch_format": "git_v1"}, id="patch-format"),
    pytest.param({"instruction": ""}, id="instruction-empty"),
    pytest.param({"instruction": "x" * 500_001}, id="instruction-500001"),
    pytest.param({"primary_language": ""}, id="lang-empty"),
    pytest.param({"primary_language": "p" * 65}, id="lang-65"),
    pytest.param({"authoring_version": ""}, id="authoring-empty"),
    pytest.param({"authoring_version": "v" * 129}, id="authoring-129"),
    pytest.param({"workspace_sha256": HEX.upper()}, id="workspace-upper"),
    pytest.param({"workspace_sha256": HEX[:-1]}, id="workspace-63"),
    pytest.param({"workspace_sha256": HEX + "a"}, id="workspace-65"),
    pytest.param({"verifier_sha256": HEX.upper()}, id="verifier-upper"),
    pytest.param({"verifier_sha256": HEX[:-1]}, id="verifier-63"),
    pytest.param({"verifier_sha256": "g" * 64}, id="verifier-nonhex"),
    pytest.param({"instruction": NFD_E}, id="instruction-nfd"),
    pytest.param({"primary_language": NFD_E}, id="lang-nfd"),
    pytest.param({"authoring_version": NFD_E}, id="authoring-nfd"),
    pytest.param({"working_directory": NFD_E}, id="workdir-nfd"),
]


@pytest.mark.parametrize("over", REPO_REJECTS)
def test_repository_identity_rejects(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.RepositoryTaskIdentity(**repo(**over))


def test_repository_identity_accepts_bounds():
    m = _mod()
    model = m.RepositoryTaskIdentity(
        **repo(
            instruction="x" * 500_000,
            primary_language="p" * 64,
            authoring_version="v" * 128,
            task_kind="feature",
        )
    )
    assert len(model.instruction) == 500_000


TERM_REJECTS = [
    pytest.param({"schema_version": 2}, id="schema-2"),
    pytest.param({"task_type": "terminal_script_v2"}, id="task-type-v2"),
    pytest.param({"instruction": ""}, id="instruction-empty"),
    pytest.param({"instruction": "x" * 500_001}, id="instruction-500001"),
    pytest.param({"authoring_version": ""}, id="authoring-empty"),
    pytest.param({"authoring_version": "v" * 129}, id="authoring-129"),
    pytest.param({"environment_sha256": HEX.upper()}, id="env-upper"),
    pytest.param({"environment_sha256": HEX[:-1]}, id="env-63"),
    pytest.param({"environment_sha256": HEX + "a"}, id="env-65"),
    pytest.param({"verifier_sha256": HEX.upper()}, id="verifier-upper"),
    pytest.param({"verifier_sha256": "g" * 64}, id="verifier-nonhex"),
    pytest.param({"instruction": NFD_E}, id="instruction-nfd"),
    pytest.param({"result_tree_path": NFD_E}, id="tree-nfd"),
]


@pytest.mark.parametrize("over", TERM_REJECTS)
def test_terminal_identity_rejects(over):
    m = _mod()
    with pytest.raises(ValidationError):
        m.TerminalScriptTaskIdentity(**term(**over))


def test_terminal_identity_has_no_cap_fields():
    m = _mod()
    fields = set(m.TerminalScriptTaskIdentity.model_fields)
    assert fields == set(term())
    for name in ("max_script_bytes", "timeout_seconds", "deadline", "max_bytes"):
        assert name not in fields
    assert set(m.RepositoryTaskIdentity.model_fields) == set(repo())


GOOD_TOKENS = ["a", "0", "repo-polyglot-v1", "a.b_c-1", "a" * 128]
BAD_TOKENS = [
    pytest.param("", id="empty"),
    pytest.param("A", id="upper"),
    pytest.param("-a", id="leading-dash"),
    pytest.param(".a", id="leading-dot"),
    pytest.param("_a", id="leading-underscore"),
    pytest.param("a b", id="space"),
    pytest.param("a/b", id="slash"),
    pytest.param("a:b", id="colon"),
    pytest.param("a@sha256:" + HEX, id="image-ref"),
    pytest.param("a" * 129, id="129"),
    pytest.param("a" + NFC_E, id="non-ascii"),
    pytest.param("a\n", id="trailing-newline"),
]


@pytest.mark.parametrize("field", ["execution_profile_id", "verifier_policy"])
@pytest.mark.parametrize("token", GOOD_TOKENS)
def test_profile_tokens_accepted_on_both_models(field, token):
    m = _mod()
    assert getattr(m.RepositoryTaskIdentity(**repo(**{field: token})), field) == token
    assert getattr(m.TerminalScriptTaskIdentity(**term(**{field: token})), field) == token


@pytest.mark.parametrize("field", ["execution_profile_id", "verifier_policy"])
@pytest.mark.parametrize("token", BAD_TOKENS)
def test_profile_tokens_rejected_on_both_models(field, token):
    m = _mod()
    with pytest.raises(ValidationError):
        m.RepositoryTaskIdentity(**repo(**{field: token}))
    with pytest.raises(ValidationError):
        m.TerminalScriptTaskIdentity(**term(**{field: token}))


# --------------------------------------------------------------------------- #
# Normalized relative path rule (shared by working_directory/result_tree_path)
# --------------------------------------------------------------------------- #
GOOD_PATHS = [
    ".",
    "src",
    "src/lib",
    ".hidden",
    "a.b/c-d_e",
    "..." ,
    "a..b",
    "out/" + NFC_E,
    "a" * 1024,
    "a/" * 511 + "aa",
]
BAD_PATHS = [
    pytest.param("", id="empty"),
    pytest.param("/", id="root-abs"),
    pytest.param("/abs", id="abs"),
    pytest.param("//abs", id="double-slash-abs"),
    pytest.param("a\x00b", id="embedded-nul"),
    pytest.param("\x00", id="nul-only"),
    pytest.param("a/\x00", id="nul-segment"),
    pytest.param("..", id="parent"),
    pytest.param("../x", id="parent-prefix"),
    pytest.param("a/..", id="parent-suffix"),
    pytest.param("a/../b", id="parent-inner"),
    pytest.param("a\\b", id="backslash"),
    pytest.param("\\", id="backslash-only"),
    pytest.param("a//b", id="empty-segment"),
    pytest.param("a/", id="trailing-slash"),
    pytest.param("./", id="dot-trailing-slash"),
    pytest.param("./src", id="dot-prefix"),
    pytest.param("src/.", id="dot-suffix"),
    pytest.param("src/./x", id="dot-inner"),
    pytest.param("./.", id="dot-dot-segments"),
    pytest.param("a" * 1025, id="1025"),
    pytest.param("out/" + NFD_E, id="nfd"),
]


@pytest.mark.parametrize("value", GOOD_PATHS, ids=lambda v: v[:16])
def test_relative_path_helper_accepts(value):
    assert _mod().validate_relative_path(value) == value


@pytest.mark.parametrize("value", BAD_PATHS)
def test_relative_path_helper_rejects(value):
    with pytest.raises(ValueError):
        _mod().validate_relative_path(value)


def test_relative_path_helper_returns_input_unchanged():
    assert _mod().validate_relative_path("src/lib") == "src/lib"
    assert _mod().validate_relative_path(".") == "."


FIELD_PATH_REJECTS = ["/abs", "//abs", "../x", "a\\b", "a//b", "a/", "./src", "", "a\x00b"]


@pytest.mark.parametrize("value", FIELD_PATH_REJECTS)
def test_working_directory_uses_relative_rule(value):
    m = _mod()
    with pytest.raises(ValidationError):
        m.RepositoryTaskIdentity(**repo(working_directory=value))


@pytest.mark.parametrize("value", FIELD_PATH_REJECTS)
def test_result_tree_path_uses_relative_rule(value):
    m = _mod()
    with pytest.raises(ValidationError):
        m.TerminalScriptTaskIdentity(**term(result_tree_path=value))


def test_paths_accept_root_and_nested_on_both_models():
    m = _mod()
    assert m.RepositoryTaskIdentity(**repo(working_directory="src/pkg")).working_directory == "src/pkg"
    assert m.TerminalScriptTaskIdentity(**term(result_tree_path=".")).result_tree_path == "."


# --------------------------------------------------------------------------- #
# Only the two exact identity classes hash; subclasses are programming errors
# --------------------------------------------------------------------------- #
def test_compute_task_id_rejects_subclasses_without_added_fields():
    m = _mod()

    class RepoSame(m.RepositoryTaskIdentity):
        pass

    class TermSame(m.TerminalScriptTaskIdentity):
        pass

    with pytest.raises(TypeError):
        m.compute_task_id(RepoSame(**repo()))
    with pytest.raises(TypeError):
        m.compute_task_id(TermSame(**term()))


def test_compute_task_id_rejects_subclasses_with_added_fields():
    m = _mod()

    class RepoExtra(m.RepositoryTaskIdentity):
        extra_note: str = "x"

    class TermExtra(m.TerminalScriptTaskIdentity):
        extra_note: str = "x"

    with pytest.raises(TypeError):
        m.compute_task_id(RepoExtra(**repo()))
    with pytest.raises(TypeError):
        m.compute_task_id(TermExtra(**term()))


def test_identity_schema_version_is_the_shared_wire_alias():
    from rlvr.v3 import wire

    m = _mod()
    assert m.SchemaVersion is wire.SchemaVersion
    for cls, builder in ((m.RepositoryTaskIdentity, repo), (m.TerminalScriptTaskIdentity, term)):
        with pytest.raises(ValidationError):
            cls(**builder(schema_version=True))
