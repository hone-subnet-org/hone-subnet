from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from rlvr.neurons.demo_miner import (
    BedrockClient,
    DemoMiner,
    DemoMinerSettings,
    ModelCompletion,
    build_model_messages,
    extract_submission,
)
from rlvr.v3.api import MinerTaskRequest, MinerTaskResponse
from rlvr.v3.miner_workspace import WorkspaceReader
from rlvr.v3.trajectory import parse_trajectory
from tests.test_v3_api import lease, slot_set


class WalletLike:
    class Hotkey:
        def __init__(self, address: str):
            self.ss58_address = address

    def __init__(self, address: str):
        self.hotkey = self.Hotkey(address)


def settings(**updates) -> DemoMinerSettings:
    return DemoMinerSettings(_env_file=None, bedrock_api_key="key", **updates)


@pytest.fixture(autouse=True)
def _no_reasks_by_default(request, monkeypatch):
    # Loop tests that predate the pre-upload check use empty diffs; keep them focused.
    if "reask" not in request.node.name:
        monkeypatch.setattr("rlvr.neurons.demo_miner.SUBMISSION_REASK_LIMIT", 0)


def task(hotkey="miner") -> MinerTaskRequest:
    value = lease()
    return MinerTaskRequest(
        protocol_version=3,
        challenge_id=value["challenge_id"],
        task_id=value["task_id"],
        identity=value["identity"],
        workspace=value["workspace"],
        workspace_url=value["workspace_url"],
        expires_at=2**53 - 1,
        slots=slot_set(hotkey=hotkey),
    )


def test_prompt_and_submission_are_v3_shapes():
    messages = build_model_messages(task())
    assert "Git unified diff" in messages[0]["content"]
    assert messages[1]["content"] == "Fix it"
    assert extract_submission("```diff\n--- a/a\n+++ b/a\n```") == b"--- a/a\n+++ b/a\n"
    assert extract_submission("```diff\n context line \n```") == b" context line \n"
    assert extract_submission("```diff\n+value\r\n```") == b"+value\r\n"
    assert extract_submission(" --- a/a\n+++ b/a\n") == b"--- a/a\n+++ b/a\n"  # provider-added leading space
    # a raw diff that adds a fenced block keeps the fence as content
    fenced = "--- a/README.md\n+++ b/README.md\n@@ -1 +1,4 @@\n x\n+```python\n+print('hi')\n+```\n"
    assert extract_submission(fenced) == fenced.encode()
    assert extract_submission("  #!/bin/bash\necho hi\n") == b"#!/bin/bash\necho hi\n"


def test_extracted_repository_patch_remains_git_applicable(tmp_path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    patch = extract_submission(
        "```diff\n"
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```"
    )
    completed = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=tmp_path,
        input=patch,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


async def test_provider_response_records_exact_bodies_and_five_alternatives():
    token = {
        "token": "x",
        "bytes": [120],
        "token_id": 7,
        "logprob": -0.1,
        "top_logprobs": [
            {"token": value, "bytes": [ord(value)], "token_id": index, "logprob": -index}
            for index, value in enumerate("abcde")
        ],
    }

    async def handler(request):
        body = json.loads(request.content)
        assert body["logprobs"] is True and body["top_logprobs"] == 5
        assert body["include_reasoning"] is True
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "x", "reasoning_content": "r"}, "logprobs": {"content": [token]}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BedrockClient(
        settings(bedrock_base_url="https://provider.invalid"), http=http
    )
    result = await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    assert result.generated_bytes == b"x"
    assert len(result.tokens[0]["top_logprobs"]) == 5
    assert json.loads(result.request_body)["model"] == "us.moonshotai.kimi-k3"


def _records(pieces: list[str]) -> list[dict]:
    alternatives = [{"token": "a", "bytes": [97], "token_id": 1, "logprob": -1.0}] * 5
    return [
        {"token": piece, "bytes": list(piece.encode()), "token_id": 1, "logprob": -0.5, "top_logprobs": alternatives}
        for piece in pieces
    ]


async def _complete_with(records: list[dict] | None, content: str = " 391", reasoning: str | None = " thinking"):
    async def handler(request):
        message = {"content": content}
        if reasoning is not None:
            message["reasoning"] = reasoning
        choice = {"finish_reason": "stop", "message": message}
        if records is not None:
            choice["logprobs"] = {"content": records}
        return httpx.Response(200, json={"choices": [choice]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BedrockClient(settings(bedrock_base_url="https://provider.invalid"), http=http)
    try:
        return await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    finally:
        await http.aclose()


async def test_provider_tokens_cover_reasoning_and_answer():
    result = await _complete_with(_records(["thinking", "</think>", "391", "<|im_end|>"]))
    assert result.generated_bytes == b"thinking</think>391<|im_end|>"
    assert result.output == " 391" and result.reasoning == " thinking"


async def test_provider_tokens_cover_a_multiword_answer():
    records = _records(["thinking", "</think>", "the", " answer", " is", " 391", "<|im_end|>"])
    result = await _complete_with(records, content=" the answer is 391\n")
    assert result.output == " the answer is 391\n"


async def test_provider_tokens_are_recorded_as_returned_even_when_they_differ_from_the_text():
    result = await _complete_with(_records(["thinking", "</think>", "392", "<|im_end|>"]))
    assert result.output == " 391" and result.generated_bytes == b"thinking</think>392<|im_end|>"


NATIVE_CALL = (
    " <|tool_calls_section_begin|> <|tool_call_begin|> functions.list_files:0 "
    "<|tool_call_argument_begin|> {\"path\": \".\", \"offset\": 0} <|tool_call_end|> <|tool_calls_section_end|>"
)


@pytest.mark.parametrize("output, expected", [
    ("```workspace\n{\"tool\":\"read_file\",\"path\":\"a.py\",\"offset\":0}\n```", {"tool": "read_file", "path": "a.py", "offset": 0}),
    (NATIVE_CALL, {"path": ".", "offset": 0, "tool": "list_files"}),
    (NATIVE_CALL.replace("list_files", "ReadFile").replace('"path": "."', '"file_path": "slug/core.py"'), {"file_path": "slug/core.py", "offset": 0, "tool": "read_file"}),
    (NATIVE_CALL.replace("list_files", "command").replace('"offset": 0', '"tool": "read_file"'), {"path": ".", "tool": "read_file"}),
    (NATIVE_CALL.replace('{"path": ".", "offset": 0}', "{not json}"), {"tool": "list_files"}),
    ("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", None),
    # markup inside a submission stays a submission
    ("--- a/t.py\n+++ b/t.py\n@@ -1 +1,2 @@\n a\n+MARK = '" + NATIVE_CALL.strip() + "'\n", None),
    ("```diff\n" + NATIVE_CALL + "\n```", None),
    # non-object arguments keep the tool name only
    (NATIVE_CALL.replace('{"path": ".", "offset": 0}', "[]"), {"tool": "list_files"}),
    # the first call wins
    (NATIVE_CALL.replace("<|tool_calls_section_end|>", " <|tool_call_begin|> functions.read_file:1 <|tool_call_argument_begin|> {\"path\": \"x\"} <|tool_call_end|> <|tool_calls_section_end|>"), {"path": ".", "offset": 0, "tool": "list_files"}),
])
def test_extract_tool_call_reads_the_fence_or_native_markup(output, expected):
    from rlvr.neurons.demo_miner import extract_tool_call

    found = extract_tool_call(output)
    assert (None if found is None else json.loads(found)) == expected


async def test_demo_uploads_submission_and_genuine_canonical_trajectory(monkeypatch, tmp_path):
    completion = ModelCompletion(
        request_body=b"{}",
        response_body=b'{"ok":true}',
        output="```diff\n\n```",
        reasoning="No change is needed.",
        generated_bytes=b"x",
        tokens=[{
            "token_bytes_b64": "eA==",
            "token_id": None,
            "logprob": 0,
            "top_logprobs": [
                {"token_bytes_b64": "eA==", "token_id": None, "logprob": 0}
            ] * 5,
        }],
    )

    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s, tools=()):
            self.calls += 1
            if self.calls == 1:
                assert "Task working directory" in messages[-1]["content"]
                return replace(
                    completion,
                    output='```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```',
                    request_body=json.dumps({"messages": messages}).encode(),
                )
            assert "existing repository contents" in messages[-1]["content"]
            return replace(completion, request_body=json.dumps({"messages": messages}).encode())

        async def aclose(self):
            return None

    captured = {}

    (tmp_path / "a.txt").write_text("existing repository contents")

    @asynccontextmanager
    async def workspace(*_args):
        yield WorkspaceReader(tmp_path)

    async def upload(_http, request, submission, trajectory, *, allowed_origins):
        captured.update(submission=submission, trajectory=trajectory, origins=allowed_origins)
        return MinerTaskResponse(
            protocol_version=3,
            challenge_id=request.challenge_id,
            task_id=request.task_id,
            response_type=request.identity.task_type,
            submission={
                "artifact_role": "patch",
                "artifact_format": "unified_diff_v1",
                "upload_id": request.slots.submission.upload_id,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "size_bytes": 0,
            },
            trajectory={
                "artifact_role": "trajectory",
                "artifact_format": "trajectory_v1",
                "upload_id": request.slots.trajectory.upload_id,
                "sha256": "1" * 64,
                "size_bytes": len(trajectory),
            },
        )

    monkeypatch.setattr("rlvr.neurons.demo_miner.upload_miner_result", upload)
    monkeypatch.setattr("rlvr.neurons.demo_miner.open_miner_workspace", workspace)
    miner = DemoMiner(settings(), Provider(), wallet=WalletLike("miner"))
    response = await miner.solve(task(), 30)
    trajectory = parse_trajectory(captured["trajectory"])
    assert response.response_type == "repository_patch_v1"
    assert captured["submission"] == b""
    assert trajectory.miner_hotkey == "miner"
    assert trajectory.events[-1].submission_sha256 == trajectory.submission_sha256
    turns = [event for event in trajectory.events if event.event_type == "model_turn"]
    assert len(turns) == 2
    assert b"existing repository contents" in base64.b64decode(turns[-1].request_body_b64)
    assert [event.sequence for event in trajectory.events] == list(range(len(trajectory.events)))
    reads = [event for event in trajectory.events if event.event_type == "tool_result"]
    assert b"existing repository contents" in base64.b64decode(reads[0].output_body_b64)


async def test_bedrock_client_requires_https():
    client = BedrockClient(settings(bedrock_base_url="http://provider.invalid"))
    try:
        with pytest.raises(RuntimeError, match="HTTPS"):
            _ = client.completion_url
    finally:
        await client.aclose()


def completion_for(output):
    encoded = base64.b64encode(output.encode()).decode()
    return ModelCompletion(
        request_body=b"{}", response_body=b"{}", output=output, reasoning="Inspect the file.",
        generated_bytes=output.encode(),
        tokens=[{
            "token_bytes_b64": encoded, "token_id": None, "logprob": 0,
            "top_logprobs": [{"token_bytes_b64": encoded, "token_id": None, "logprob": 0}] * 5,
        }],
    )


@pytest.mark.parametrize("keep_calling", [False, True])
async def test_workspace_tool_budget_allows_one_final_turn(tmp_path, keep_calling):
    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s, tools=()):
            self.calls += 1
            if self.calls == 2:
                assert "Tool budget exhausted" in messages[-1]["content"]
                if not keep_calling:
                    return completion_for("```diff\n\n```")
            return completion_for('```workspace\n{"tool":"list_files","path":".","offset":0}\n```')

    import time

    provider = Provider()
    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), provider)
    if keep_calling:
        with pytest.raises(ValueError, match="tool call limit"):
            await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    else:
        completion, _, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
        assert extract_submission(completion.output) == b""
    assert provider.calls == 2


async def test_workspace_reads_share_one_model_deadline(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", SimpleNamespace(monotonic=lambda: clock[0]))
    budgets = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            budgets.append(timeout_s)
            clock[0] += 25
            if len(budgets) == 1:
                return completion_for('```workspace\n{"tool":"read_file","path":"missing","offset":0}\n```')
            assert "error" in messages[-1]["content"]
            return completion_for("```diff\n\n```")

    miner = DemoMiner(settings(), Provider())
    await miner._generate(task(), WorkspaceReader(tmp_path), 200)
    assert budgets == [100, 75]
    with pytest.raises(TimeoutError):
        await miner._generate(task(), WorkspaceReader(tmp_path), 100)
    assert len(budgets) == 2


async def test_provider_call_is_bounded_by_total_time(monkeypatch):
    async def handler(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BedrockClient(settings(bedrock_base_url="https://provider.invalid", bedrock_max_retries=0), http=http)
    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await client.complete([{"role": "user", "content": "x"}], timeout_s=0.2)
    await http.aclose()
    assert time.monotonic() - started < 2


async def test_provider_reply_without_reasoning_is_recorded_with_empty_reasoning():
    result = await _complete_with(_records(["391", "<|im_end|>"]), content=" 391", reasoning=None)
    assert result.reasoning == "" and result.output == " 391"


async def test_malformed_reasoning_is_rejected_not_blanked():
    with pytest.raises(RuntimeError, match="invalid response"):
        await _complete_with(_records(["391", "<|im_end|>"]), content=" 391", reasoning=False)


async def test_empty_reply_is_recorded_and_the_loop_reasks(tmp_path):
    import time

    empty = await _complete_with(None, content="", reasoning="thinking only")
    assert empty.output == "" and empty.tool_calls == ()
    replies = [empty, completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```")]
    (tmp_path / "a.txt").write_text("a\n")
    prompts = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            prompts.append(messages[-1]["content"])
            return replies.pop(0)

    miner = DemoMiner(settings(), Provider())
    _, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission.startswith(b"--- a/a.txt") and prompts[-1].startswith("That reply was not accepted:")


async def test_provider_reply_without_logprobs_is_recorded_without_tokens():
    result = await _complete_with(None, content=" 391")
    assert result.tokens == [] and result.generated_bytes == b"" and result.output == " 391"


async def test_present_but_malformed_logprobs_are_still_rejected():
    records = _records(["391"])
    records[0]["top_logprobs"] = records[0]["top_logprobs"][:3]
    with pytest.raises(RuntimeError, match="invalid response"):
        await _complete_with(records, content=" 391")


@pytest.mark.parametrize("records", [{}, "", 0])
async def test_logprobs_with_wrong_type_content_are_rejected(records):
    with pytest.raises(RuntimeError, match="invalid response"):
        await _complete_with(records, content=" 391")


def _tool_call_message(*calls):
    return {"content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)
    ]}


async def test_provider_tool_calls_are_parsed_and_the_request_declares_tools():
    seen = {}

    async def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": _tool_call_message(("list_files", {"path": ".", "offset": 0}))}]})

    from rlvr.neurons.demo_miner import WORKSPACE_TOOLS

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BedrockClient(settings(bedrock_base_url="https://provider.invalid"), http=http)
    result = await client.complete([{"role": "user", "content": "x"}], timeout_s=10, tools=WORKSPACE_TOOLS)
    await http.aclose()
    assert result.tool_calls == (("call_0", "list_files", '{"path": ".", "offset": 0}'),)
    assert result.output == "" and result.tokens == []
    assert [tool["function"]["name"] for tool in seen["body"]["tools"]] == ["list_files", "find_files", "read_file"]


def completion_with_calls(*calls):
    base = completion_for("")
    return dataclasses.replace(base, tool_calls=tuple((f"call_{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)))


async def test_loop_executes_provider_tool_calls_and_answers_in_tool_messages(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("hello")
    replies = [
        completion_with_calls(("find_files", {"path": ".", "offset": 0}), ("read_file", {"path": "sub/a.txt", "offset": 0})),
        completion_for("```diff\n\n```"),
    ]
    seen_messages = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            seen_messages.append([dict(m) for m in messages])
            return replies.pop(0)

    import time

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=2), Provider())
    completion, events, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert extract_submission(completion.output) == b""
    second = seen_messages[1]
    assistant = next(m for m in second if m["role"] == "assistant")
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_0", "call_1"]
    tool_replies = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_replies] == ["call_0", "call_1"]
    assert json.loads(tool_replies[0]["content"])["files"] == ["sub/a.txt"]
    assert json.loads(tool_replies[1]["content"])["content"] == "hello"
    assert "Tool budget exhausted" in second[-1]["content"]  # both calls used the budget of two
    call_ids = [e["call_id"] for e in events if e["event_type"] == "tool_call"]
    assert call_ids == ["workspace-0", "workspace-1"]


async def test_loop_reasks_when_the_submission_is_not_a_valid_patch(tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n<|tool_call_begin|> functions.read_file:0 {\"path\": \"a.txt\"}\n```"
    good = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```"
    replies = [completion_for(bad), completion_for(good)]
    prompts = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            prompts.append(messages[-1]["content"])
            return replies.pop(0)

    import time

    miner = DemoMiner(settings(), Provider())
    _, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission.startswith(b"--- a/a.txt")
    assert prompts[-1].startswith("That reply was not accepted:")


async def test_loop_reask_gives_up_after_the_limit_and_uploads_the_last_reply(tmp_path):
    bad = completion_for("```diff\nnot a diff\n```")

    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s, tools=()):
            self.calls += 1
            return bad

    import time

    provider = Provider()
    miner = DemoMiner(settings(), provider)
    completion, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert completion is bad and provider.calls == 3  # first try plus two re-asks
    assert submission == b"not a diff\n"  # uploaded as is


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
async def test_check_submission_uses_git_and_repairs_only_what_git_accepts(tmp_path):
    import time

    (tmp_path / "a.txt").write_text("a\n")
    miner = DemoMiner(settings(), object())
    deadline = time.monotonic() + 30
    good = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```"
    wrong_context = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    assert await miner._check_submission(task(), good, WorkspaceReader(tmp_path), deadline) == (extract_submission(good), None)
    submission, problem = await miner._check_submission(task(), wrong_context, WorkspaceReader(tmp_path), deadline)
    assert submission == extract_submission(wrong_context) and problem and "error" in problem
    assert (await miner._check_submission(task(), "```diff\n\x00\n```", WorkspaceReader(tmp_path), deadline))[1] == "patch contains a NUL byte"


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
async def test_check_submission_ignores_git_config_inside_the_workspace(tmp_path):
    import time

    (tmp_path / "a.txt").write_text("a\n")
    (tmp_path / ".gitattributes").write_text("* filter=evil\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(f'[filter "evil"]\n\tclean = touch {tmp_path}/FILTER_RAN\n')
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    miner = DemoMiner(settings(), object())
    good = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```"
    assert (await miner._check_submission(task(), good, WorkspaceReader(tmp_path), time.monotonic() + 30))[1] is None
    assert not (tmp_path / "FILTER_RAN").exists()


def test_find_files_is_recursive_in_directory_order_and_paged(tmp_path, monkeypatch):
    from rlvr.v3 import miner_workspace

    for name in ("b/z.txt", "b/a/x.txt", "a.txt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "b")
    monkeypatch.setattr(miner_workspace, "LIST_ENTRIES", 2)
    reader = WorkspaceReader(tmp_path)
    # a directory's files come before its subdirectories; the symlink is skipped
    first = json.loads(reader.execute({"tool": "find_files", "path": ".", "offset": 0}))
    assert first == {"files": ["a.txt", "b/z.txt"], "next_offset": 2}
    second = json.loads(reader.execute({"tool": "find_files", "path": ".", "offset": 2}))
    assert second == {"files": ["b/a/x.txt"], "next_offset": None}
    assert "error" in json.loads(reader.execute({"tool": "find_files", "path": "a.txt", "offset": 0}))


@pytest.mark.parametrize("header, body, expected", [
    ("@@ -1,8 +1,17 @@", [" a", "-b", "+c", "+d"], "@@ -1,2 +1,3 @@"),
    ("@@ -1 +1 @@ fn main()", ["-a", "+b"], "@@ -1,1 +1,1 @@ fn main()"),
    ("@@ -3,2 +3,2 @@", [" x", "-y", "+z", "\\ No newline at end of file"], "@@ -3,2 +3,2 @@"),
])
def test_recount_hunks_fixes_only_the_counts(header, body, expected):
    from rlvr.neurons.demo_miner import recount_hunks

    patch = "--- a/f\n+++ b/f\n" + header + "\n" + "\n".join(body) + "\n"
    fixed = recount_hunks(patch.encode()).decode()
    assert fixed == "--- a/f\n+++ b/f\n" + expected + "\n" + "\n".join(body) + "\n"


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
async def test_recount_is_applied_only_when_git_accepts_the_repair(tmp_path):
    import time

    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    miner = DemoMiner(settings(), object())
    miscounted = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,9 +1,12 @@\n one\n-two\n+TWO\n+more\n three\n```"
    submission, problem = await miner._check_submission(task(), miscounted, WorkspaceReader(tmp_path), time.monotonic() + 30)
    assert problem is None and submission.startswith(b"--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,4 @@\n")


def test_recount_leaves_a_multi_file_diff_without_git_headers_alone_when_valid():
    from rlvr.neurons.demo_miner import recount_hunks

    # Two files, no "diff --git" lines: the second file's headers start with - and +.
    patch = b"--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n--- a/y\n+++ b/y\n@@ -1 +1 @@\n-c\n+d\n"
    assert recount_hunks(patch) != patch  # the naive recount would change it ...
    # ... which is why the miner only uses a recount that git has accepted.


async def test_a_repair_is_not_used_when_git_could_not_run(tmp_path, monkeypatch):
    import time

    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    miscounted = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,9 +1,12 @@\n one\n-two\n+TWO\n+more\n three\n```"
    calls = []

    def fake_git_check(git, patch, workspace, timeout):
        calls.append(patch)
        if len(calls) == 1:
            return True, "error: corrupt patch at line 3"
        raise_timeout = __import__("subprocess").TimeoutExpired
        raise raise_timeout(cmd="git", timeout=timeout)

    def guarded(git, patch, workspace, timeout):
        try:
            return fake_git_check(git, patch, workspace, timeout)
        except __import__("subprocess").SubprocessError:
            return False, None

    monkeypatch.setattr("rlvr.neurons.demo_miner.shutil.which", lambda _: "/usr/bin/git")
    monkeypatch.setattr("rlvr.neurons.demo_miner._git_check", guarded)
    miner = DemoMiner(settings(), object())
    submission, problem = await miner._check_submission(task(), miscounted, WorkspaceReader(tmp_path), time.monotonic() + 30)
    assert submission == extract_submission(miscounted)  # the raw bytes, not the unverified repair
    assert problem == "error: corrupt patch at line 3"
    assert len(calls) == 2


async def test_no_git_check_runs_when_the_deadline_has_passed(tmp_path, monkeypatch):
    import time

    (tmp_path / "a.txt").write_text("a\n")
    monkeypatch.setattr("rlvr.neurons.demo_miner.shutil.which", lambda _: "/usr/bin/git")
    monkeypatch.setattr("rlvr.neurons.demo_miner._git_check", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    miner = DemoMiner(settings(), object())
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    assert await miner._check_submission(task(), bad, WorkspaceReader(tmp_path), time.monotonic() - 1) == (extract_submission(bad), None)


async def test_reask_is_skipped_and_the_reply_uploaded_when_time_runs_out(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", SimpleNamespace(monotonic=lambda: clock[0]))
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    calls = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            calls.append(timeout_s)
            clock[0] += 60  # the model reply consumed the rest of the budget
            return completion_for(bad)

    async def failing_check(request, output, reader, deadline):
        return extract_submission(output), "error: patch failed: a.txt:1"

    miner = DemoMiner(settings(), Provider())
    monkeypatch.setattr(miner, "_check_submission", failing_check)
    _, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), 150)
    assert submission == extract_submission(bad) and calls == [50]  # no second call, no TimeoutError


async def test_reask_call_timeout_uploads_the_previous_reply(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    calls = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            calls.append(len(messages))
            if len(calls) == 1:
                return completion_for(bad)
            raise TimeoutError("Bedrock request deadline exceeded")

    async def failing_check(request, output, reader, deadline):
        return extract_submission(output), "error: patch failed: a.txt:1"

    import time

    miner = DemoMiner(settings(), Provider())
    monkeypatch.setattr(miner, "_check_submission", failing_check)
    _, events, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission == extract_submission(bad) and len(calls) == 2
    assert [e["event_type"] for e in events] == ["model_turn"]  # the failed correction attempt left no event


async def test_reask_then_tool_call_then_timeout_still_uploads_the_checked_reply(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    replies = [completion_for(bad), completion_for('```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```')]

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            if replies:
                return replies.pop(0)
            raise TimeoutError("Bedrock request deadline exceeded")

    async def failing_check(request, output, reader, deadline):
        return extract_submission(output), "error: patch failed: a.txt:1"

    import time

    miner = DemoMiner(settings(), Provider())
    monkeypatch.setattr(miner, "_check_submission", failing_check)
    _, events, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission == extract_submission(bad)
    assert [e["event_type"] for e in events] == ["model_turn"]


async def test_over_budget_tool_call_reasks_for_the_submission_instead_of_failing(tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    call = completion_for('```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```')
    diff = completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```")
    replies = [call, call, diff]  # second call is over a budget of one
    prompts = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            prompts.append(messages[-1]["content"])
            return replies.pop(0)

    import time

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), Provider())
    _, events, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission.startswith(b"--- a/a.txt")
    assert sum(1 for e in events if e["event_type"] == "tool_call") == 1  # the over-budget call was not run
    assert sum(1 for e in events if e["event_type"] == "model_turn") == 3  # every model reply is still recorded
    assert "budget is exhausted" in prompts[-1]


async def test_over_budget_tool_calls_fail_after_the_reask_limit(tmp_path):
    call = completion_for('```workspace\n{"tool":"list_files","path":".","offset":0}\n```')

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            return call  # never stops calling tools

    import time

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), Provider())
    with pytest.raises(ValueError, match="tool call limit"):
        await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
