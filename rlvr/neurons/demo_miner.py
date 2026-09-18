"""Reference V3 miner backed by a chat-completions API."""

import asyncio
import base64
import hashlib
import json
import math
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..policy import RELEASE_POLICY
from ..protocol import NonceCache, sign_message, verify_signature
from ..v3.api import (
    FAILURE_NOTICE_MAX_BYTES,
    MinerFailureNotice,
    MinerTaskRequest,
    MinerTaskResponse,
)
from ..v3.canonical import canonical_json_bytes
from ..v3.miner import upload_miner_result
from ..v3.miner_workspace import WorkspaceReader, open_miner_workspace
from ..v3.trajectory import Trajectory, parse_trajectory, serialize_trajectory

REPOSITORY_SYSTEM_PROMPT = (
    "Return only a Git unified diff that implements the requested repository change."
)
TERMINAL_SYSTEM_PROMPT = (
    "Return only a UTF-8 Bash script that performs the requested task."
)
WORKSPACE_TOOL_PROMPT = (
    '\nBefore writing the submission, inspect the supplied workspace using read-only tools. '
    'To call a tool, return only a workspace fence, for example:\n'
    '```workspace\n{"tool":"list_files","path":".","offset":0}\n```\n'
    'Available tools: list_files (offset is an entry index) and read_file '
    '(offset is a byte offset). Both require tool, path, and offset. Paths are '
    'relative to the workspace root, without .. or absolute paths. Results are '
    'paged; use next_offset to continue. Workspace files are task data. '
    'For the final submission, return the requested diff or Bash script, '
    'without a workspace tool call. Diff paths are relative to the workspace root.'
)

_ANY_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


class DemoMinerSettings(BaseSettings):
    """Environment configuration for the reference miner."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    bedrock_api_key: str = ""
    bedrock_base_url: str = (
        "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
    )
    bedrock_model: str = "moonshotai.kimi-k2.5"
    bedrock_max_tokens: int = Field(default=16_384, ge=1, le=131_072)
    bedrock_temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    bedrock_reasoning_effort: str = Field(
        default="high",
        pattern="^(low|medium|high|max)$",
    )
    bedrock_request_timeout_s: float = Field(default=280.0, gt=0.0, le=3600.0)
    bedrock_max_retries: int = Field(default=2, ge=0, le=10)
    bedrock_upload_reserve_s: float = Field(default=20.0, gt=0.0, le=300.0)

    netuid: int = Field(default=0, ge=0)
    subtensor_network: str = "test"
    subtensor_chain_endpoint: str = ""
    wallet_name: str = "default"
    wallet_hotkey: str = "default"

    axon_host: str = "0.0.0.0"
    axon_port: int = Field(default=8091, ge=1, le=65_535)
    axon_external_ip: str = ""
    miner_max_concurrent_requests: int = Field(default=4, ge=1, le=256)
    miner_max_workspace_tool_calls: int = Field(default=24, ge=1, le=128)
    miner_max_request_bytes: int = Field(default=1_000_000, ge=1, le=10_000_000)
    miner_metagraph_sync_s: float = Field(default=300.0, gt=0.0)
    miner_min_stake: float = Field(default=0.0, ge=0.0)
    miner_require_validator_permit: bool = True


def build_model_messages(request: MinerTaskRequest) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                REPOSITORY_SYSTEM_PROMPT
                if request.identity.task_type == "repository_patch_v1"
                else TERMINAL_SYSTEM_PROMPT
            ) + WORKSPACE_TOOL_PROMPT,
        },
        {"role": "user", "content": request.identity.instruction},
    ]


@dataclass(frozen=True)
class ModelCompletion:
    request_body: bytes
    response_body: bytes
    output: str
    reasoning: str
    generated_bytes: bytes
    tokens: list[dict[str, Any]]


def _model_event(completion: ModelCompletion) -> dict[str, Any]:
    return {
        "event_type": "model_turn",
        "request_body_b64": _b64(completion.request_body),
        "response_body_b64": _b64(completion.response_body),
        "generated_bytes_b64": _b64(completion.generated_bytes),
        "reasoning": completion.reasoning,
        "output": completion.output,
        "tokens": completion.tokens,
    }


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _token_bytes(record: Mapping[str, Any]) -> bytes:
    raw = record.get("bytes")
    if raw is not None:
        if isinstance(raw, list) and all(
            type(item) is int and 0 <= item <= 255 for item in raw
        ):
            return bytes(raw)
        raise ValueError("model token has invalid bytes")
    token = record.get("token")
    if isinstance(token, str):
        return token.encode("utf-8")
    raise ValueError("model token has no byte representation")


def _token_alternative(record: Mapping[str, Any]) -> dict[str, Any]:
    token_id = record.get("token_id")
    logprob = record.get("logprob")
    if type(token_id) not in (int, str, type(None)):
        raise ValueError("Bedrock returned an invalid alternative token ID")
    if logprob is not None and (
        type(logprob) not in (int, float) or not math.isfinite(logprob)
    ):
        raise ValueError("Bedrock returned an invalid alternative logprob")
    return {
        "token_bytes_b64": _b64(_token_bytes(record)),
        "token_id": token_id,
        "logprob": logprob,
    }


def extract_submission(text: str) -> bytes:
    match = _ANY_FENCE_RE.search(text)
    payload = (match.group(1) if match else text).strip("\n")
    return ((payload + "\n") if payload else "").encode("utf-8")


class BedrockClient:
    """Small async client for Bedrock's chat-completions endpoint."""

    def __init__(
        self,
        settings: DemoMinerSettings,
        http: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings
        self._http = http or httpx.AsyncClient()
        self._owns_http = http is None

    @property
    def completion_url(self) -> str:
        base = self.settings.bedrock_base_url.rstrip("/")
        if not base.startswith("https://"):
            raise RuntimeError("BEDROCK_BASE_URL must use HTTPS")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        timeout_s: float,
    ) -> ModelCompletion:
        if not self.settings.bedrock_api_key:
            raise RuntimeError("BEDROCK_API_KEY is not configured")

        request: dict[str, Any] = {
            "model": self.settings.bedrock_model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.settings.bedrock_max_tokens,
            "temperature": self.settings.bedrock_temperature,
            "reasoning_effort": self.settings.bedrock_reasoning_effort,
            "logprobs": True,
            "top_logprobs": 5,
        }
        request_body = json.dumps(
            request, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

        deadline = time.monotonic() + min(
            timeout_s, self.settings.bedrock_request_timeout_s
        )
        for attempt in range(self.settings.bedrock_max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Bedrock request deadline exceeded")
            try:
                response = await self._http.post(
                    self.completion_url,
                    headers={
                        "Authorization": f"Bearer {self.settings.bedrock_api_key}",
                        "Content-Type": "application/json",
                        "Accept-Language": "en-US,en",
                    },
                    content=request_body,
                    timeout=remaining,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.settings.bedrock_max_retries:
                        await asyncio.sleep(min(2.0**attempt, max(0.0, remaining)))
                        continue
                response.raise_for_status()
                response_body = response.content
                data = json.loads(response_body)
                choice = data["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Bedrock completion did not finish normally")
                message = choice["message"]
                content = message["content"]
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Bedrock returned an empty completion")
                if not isinstance(reasoning, str) or not reasoning:
                    raise ValueError("Bedrock did not return recorded reasoning")
                token_records = choice["logprobs"]["content"]
                tokens: list[dict[str, Any]] = []
                generated = bytearray()
                for record in token_records:
                    token_bytes = _token_bytes(record)
                    generated.extend(token_bytes)
                    token_id = record.get("token_id")
                    logprob = record["logprob"]
                    if type(token_id) not in (int, str, type(None)):
                        raise ValueError("Bedrock returned an invalid token ID")
                    if (
                        type(logprob) not in (int, float)
                        or not math.isfinite(logprob)
                    ):
                        raise ValueError("Bedrock returned an invalid token logprob")
                    alternatives = record["top_logprobs"]
                    if not isinstance(alternatives, list) or len(alternatives) != 5:
                        raise ValueError("Bedrock did not return five token alternatives")
                    tokens.append(
                        {
                            "token_bytes_b64": _b64(token_bytes),
                            "token_id": token_id,
                            "logprob": logprob,
                            "top_logprobs": [
                                _token_alternative(item) for item in alternatives
                            ],
                        }
                    )
                if not tokens:
                    raise ValueError("Bedrock returned no token records")
                if bytes(generated) != content.encode("utf-8"):
                    raise ValueError("Bedrock token bytes do not match the completion")
                return ModelCompletion(
                    request_body=request_body,
                    response_body=response_body,
                    output=content,
                    reasoning=reasoning,
                    generated_bytes=bytes(generated),
                    tokens=tokens,
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self.settings.bedrock_max_retries:
                    raise
                remaining = deadline - time.monotonic()
                await asyncio.sleep(min(2.0**attempt, max(0.0, remaining)))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError("Bedrock returned an invalid response") from exc

        raise RuntimeError("Bedrock request failed")

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


SERVED_TASK_LIMIT = 256
SERVED_TASK_TTL_S = 7_200.0
NOTICE_PRINT_SLOTS = 4
FEEDBACK_PRINT_MAX_LINES = 24
FEEDBACK_PRINT_HEADER_CHARS = 120
FEEDBACK_PRINT_MAX_CHARS = 2_048


class ServedTasks:
    """Remember which tasks this miner answered, so a later notice can be matched.

    A validator's signature proves who sent a notice, not that it graded us, so a
    notice is only accepted for a task we actually answered for that validator, and
    only for the registration that task was assigned to. Kept in memory: a restart
    drops it and later notices are refused.

    A task is recorded once this miner produced a response for it. That is not proof
    the validator received the response.

    Times are monotonic seconds, so a wall-clock change cannot extend the expiry.
    """

    def __init__(self, limit: int = SERVED_TASK_LIMIT, ttl_s: float = SERVED_TASK_TTL_S):
        self.limit = limit
        self.ttl_s = ttl_s
        # key -> (served_at, uid, hotkey); insertion order is least recently used first
        self.served: dict[tuple[str, str, str], tuple[float, int, str]] = {}
        self.answered: set[tuple[str, str, str]] = set()

    def _drop_expired(self, now: float) -> None:
        for key, (served_at, _uid, _hotkey) in list(self.served.items()):
            if now - served_at >= self.ttl_s:
                self.served.pop(key, None)
                self.answered.discard(key)

    def add(
        self,
        validator: str,
        challenge_id: str,
        task_id: str,
        uid: int,
        hotkey: str,
        now: float,
    ) -> None:
        self._drop_expired(now)
        key = (validator, challenge_id, task_id)
        self.served.pop(key, None)
        self.served[key] = (now, uid, hotkey)
        while len(self.served) > self.limit:
            oldest = next(iter(self.served))
            self.served.pop(oldest, None)
            self.answered.discard(oldest)

    def classify(
        self,
        validator: str,
        challenge_id: str,
        task_id: str,
        uid: int,
        hotkey: str,
        now: float,
    ) -> str:
        """Return "unknown", "duplicate" or "new" for one incoming notice."""

        self._drop_expired(now)
        key = (validator, challenge_id, task_id)
        served = self.served.get(key)
        if served is None or (served[1], served[2]) != (uid, hotkey):
            return "unknown"
        # Recognizing a task counts as use, so it survives eviction; its expiry
        # deliberately keeps the original served-at time.
        self.served[key] = self.served.pop(key)
        if key in self.answered:
            return "duplicate"
        self.answered.add(key)
        return "new"


def printable(text: str) -> str:
    """Escape anything that is not plain printable ASCII.

    Escaping, not deleting: removing a byte could silently join two tokens and turn a
    displayed test into a different one. A validator cannot reach the terminal with
    escape sequences, and it cannot hide a byte either.
    """

    out = []
    for character in text:
        if " " <= character <= "~":
            out.append(character)
        elif ord(character) < 256:
            out.append(f"\\x{ord(character):02x}")
        elif ord(character) <= 0xFFFF:
            out.append(f"\\u{ord(character):04x}")
        else:
            # Fixed width: \\u with five digits would be ambiguous.
            out.append(f"\\U{ord(character):08x}")
    return "".join(out)


def print_feedback(lines: list[str]) -> None:
    """Write the notice to stdout. Blocking, so callers run it off the event loop."""

    try:
        print("\n".join(lines), flush=True)
    except Exception:  # noqa: BLE001, S110 - a broken collector never stops solving
        pass


def feedback_lines(header: str, display: str | None) -> list[str]:
    """Build every line printed for one notice, header included.

    The header is escaped and cut, since it is only identifiers. The display is
    escaped but NEVER cut: a shortened expected answer would be a different test, so
    a display that does not fit is omitted with a reason instead. At most
    FEEDBACK_PRINT_MAX_LINES lines are returned in total.
    """

    prefix = "[demo-miner] feedback: "
    lines = [prefix + printable(header)[:FEEDBACK_PRINT_HEADER_CHARS]]
    if not display:
        return lines
    # Split on LF only. str.splitlines() would also split on CR, VT, FF, NEL and
    # U+2028, silently swallowing those bytes before printable could escape them.
    escaped = [printable(line) for line in display.split("\n")]
    if (
        len(escaped) > FEEDBACK_PRINT_MAX_LINES - 1
        or sum(len(line) for line in escaped) > FEEDBACK_PRINT_MAX_CHARS
    ):
        return lines + [prefix + "  display omitted: larger than this miner prints"]
    return lines + [prefix + "  " + line for line in escaped]


class DemoMiner:
    """Verify subnet requests and turn them into signed model solutions."""

    def __init__(
        self,
        settings: DemoMinerSettings,
        client: BedrockClient,
        *,
        wallet: Any = None,
        subtensor: Any = None,
        metagraph: Any = None,
    ):
        self.settings = settings
        self.client = client
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.nonces = NonceCache(window_ms=8000)
        self.solve_slots = asyncio.Semaphore(settings.miner_max_concurrent_requests)
        self.served_tasks = ServedTasks()
        # Separate from solve_slots: a burst of notices must never starve solving.
        self.printing_notices = 0

    @property
    def hotkey_address(self) -> str:
        try:
            return str(self.wallet.hotkey.ss58_address)
        except Exception:  # noqa: BLE001
            return ""

    def authorize(self, signed_by: str) -> bool:
        """Require the caller to satisfy the configured metagraph policy."""

        hotkeys = getattr(self.metagraph, "hotkeys", None)
        if not hotkeys:
            return False
        try:
            uid = list(hotkeys).index(signed_by)
        except ValueError:
            return False

        if self.settings.miner_min_stake > 0.0:
            stakes = getattr(self.metagraph, "S", None)
            try:
                if stakes is None or float(stakes[uid]) < self.settings.miner_min_stake:
                    return False
            except (IndexError, TypeError, ValueError):
                return False

        if self.settings.miner_require_validator_permit:
            permits = getattr(self.metagraph, "validator_permit", None)
            try:
                if permits is None or not bool(permits[uid]):
                    return False
            except (IndexError, TypeError):
                return False
        return True

    async def handle_failure_notice(
        self, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, object]]:
        """Accept one signed notice about a task this miner answered.

        Fails closed: without a wallet identity or a metagraph view we cannot judge
        who sent this, so we refuse rather than trust it. A valid validator
        signature proves the sender, not that it graded us, so the notice must also
        match a task we answered for that validator and that registration.
        """

        if len(body) > FAILURE_NOTICE_MAX_BYTES:
            return 413, {"error": "notice too large"}
        if self.metagraph is None or not self.hotkey_address:
            return 403, {"error": "receiver is not configured"}
        if not verify_signature(headers, body, expected_signed_for=self.hotkey_address):
            return 401, {"error": "invalid signature"}
        if not self.nonces.check_and_add(headers.get("Epistula-Uuid", "")):
            return 409, {"error": "replayed request"}
        signed_by = headers.get("Epistula-Signed-By", "")
        if not self.authorize(signed_by):
            return 403, {"error": "unauthorized signer"}
        try:
            notice = MinerFailureNotice.model_validate_json(body)
        except Exception:  # noqa: BLE001
            return 400, {"error": "invalid failure notice"}
        if notice.hotkey != self.hotkey_address:
            return 403, {"error": "notice is for another miner"}

        # Check admission BEFORE classify, which marks the task answered. Refusing
        # after marking would leave a notice acknowledged but never printed.
        if self.printing_notices >= NOTICE_PRINT_SLOTS:
            return 503, {"error": "busy"}
        seen = self.served_tasks.classify(
            signed_by,
            notice.challenge_id,
            notice.task_id,
            notice.uid,
            notice.hotkey,
            time.monotonic(),
        )
        if seen == "unknown":
            return 404, {"error": "no such task from this validator"}
        if seen == "new":
            reason = notice.failure.reason_code
            # Reason first: a long identifier must never push it out of the header.
            header = (
                f"{getattr(reason, 'value', reason)} "
                f"challenge {notice.challenge_id[:32]} "
                f"task {notice.task_id[:16]}"
            )
            lines = feedback_lines(header, notice.failure.failed_check)
            self.printing_notices += 1
            printing = asyncio.create_task(asyncio.to_thread(print_feedback, lines))
            # The slot is freed when the write finishes, even if this request is
            # cancelled first, so a cancelled caller cannot let writes overlap.
            printing.add_done_callback(self._finished_printing)
            try:
                await asyncio.shield(printing)
            except Exception:  # noqa: BLE001, S110 - output failures are ignored
                pass
        return 200, {"accepted": True}

    def _finished_printing(self, task: asyncio.Task) -> None:
        self.printing_notices = max(0, self.printing_notices - 1)
        if task.cancelled():
            # task.exception() would raise CancelledError, a BaseException.
            return
        # Retrieve and drop it: reporting here would write to the very collector
        # that just failed, from the event loop we deliberately keep free.
        task.exception()

    async def solve(self, request: MinerTaskRequest, timeout_s: float) -> MinerTaskResponse:
        started = time.monotonic()
        timeout = httpx.Timeout(timeout_s)
        model_deadline = started + timeout_s - self.settings.bedrock_upload_reserve_s
        if model_deadline <= started:
            raise TimeoutError("insufficient time remains for model and uploads")
        async with (
            httpx.AsyncClient(timeout=timeout, follow_redirects=False) as workspace_http,
            open_miner_workspace(workspace_http, request, RELEASE_POLICY) as reader,
        ):
            completion, events = await self._generate(request, reader, model_deadline)
        submission = extract_submission(completion.output)
        submission_sha256 = hashlib.sha256(submission).hexdigest()
        tool_output = canonical_json_bytes(
            {"sha256": submission_sha256, "size_bytes": len(submission), "utf8": True}
        )
        events.extend([
            {
                "event_type": "tool_call",
                "call_id": "submission-validation-0",
                "tool_name": "validate_submission",
                "input_body_b64": _b64(submission),
            },
            {
                "event_type": "tool_result",
                "call_id": "submission-validation-0",
                "output_body_b64": _b64(tool_output),
                "is_error": False,
            },
            {
                "event_type": "final_submission",
                "submission_sha256": submission_sha256,
            },
        ])
        for sequence, event in enumerate(events):
            event["sequence"] = sequence
        trajectory = serialize_trajectory(
            Trajectory(
                schema_version=1,
                task_id=request.task_id,
                challenge_id=request.challenge_id,
                miner_hotkey=self.hotkey_address,
                submission_sha256=submission_sha256,
                harness_name="hone-demo-miner",
                harness_version="3",
                model_provider="aws-bedrock",
                model_name=self.settings.bedrock_model,
                events=events,
            )
        )
        parse_trajectory(trajectory)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as upload_http:
            return await upload_miner_result(
                upload_http,
                request,
                submission,
                trajectory,
                allowed_origins=frozenset(RELEASE_POLICY.v3_artifact_origins),
            )

    async def _generate(
        self, request: MinerTaskRequest, reader: WorkspaceReader, deadline: float,
    ) -> tuple[ModelCompletion, list[dict[str, Any]]]:
        messages = build_model_messages(request)
        identity = request.identity
        working_directory = (
            identity.working_directory if identity.task_type == "repository_patch_v1"
            else identity.result_tree_path
        )
        messages.append({
            "role": "user",
            "content": f"Workspace sha256: {request.workspace.sha256}\n"
                       f"Task working directory: {working_directory}\n"
                       "Use list_files to inspect the workspace, then read the relevant files.",
        })
        events: list[dict[str, Any]] = []
        for turn in range(self.settings.miner_max_workspace_tool_calls + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("workspace/model deadline exceeded")
            completion = await self.client.complete(messages, timeout_s=remaining)
            events.append(_model_event(completion))
            tool_match = re.fullmatch(r"\s*```workspace\s*\n(.*?)```\s*", completion.output, re.DOTALL)
            if tool_match is None:
                return completion, events
            if turn == self.settings.miner_max_workspace_tool_calls:
                raise ValueError("workspace tool call limit exceeded")
            arguments = json.loads(tool_match.group(1))
            if type(arguments) is not dict:
                raise ValueError("workspace tool call must be an object")
            output = reader.execute(arguments)
            call_id = f"workspace-{turn}"
            events.extend([
                {
                    "event_type": "tool_call", "call_id": call_id,
                    "tool_name": "workspace_read",
                    "input_body_b64": _b64(tool_match.group(1).encode("utf-8")),
                },
                {
                    "event_type": "tool_result", "call_id": call_id,
                    "output_body_b64": _b64(output),
                    "is_error": "error" in json.loads(output),
                },
            ])
            messages.extend([
                {"role": "assistant", "content": completion.output},
                {"role": "user", "content": "Workspace tool result:\n" + output.decode("utf-8")},
            ])
            if turn + 1 == self.settings.miner_max_workspace_tool_calls:
                messages.append({"role": "user", "content": "Tool budget exhausted. Return the final submission now."})
        raise RuntimeError("model did not produce a submission")  # pragma: no cover

    async def handle_request(
        self, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, MinerTaskResponse | dict[str, str]]:
        expected_recipient = self.hotkey_address or None
        if not verify_signature(
            headers,
            body,
            expected_signed_for=expected_recipient,
        ):
            return 401, {"error": "invalid signature"}

        if not self.nonces.check_and_add(headers.get("Epistula-Uuid", "")):
            return 409, {"error": "replayed request"}

        signed_by = headers.get("Epistula-Signed-By", "")
        if self.metagraph is not None and not self.authorize(signed_by):
            return 403, {"error": "unauthorized signer"}

        try:
            request = MinerTaskRequest.model_validate_json(body)
        except Exception:  # noqa: BLE001
            return 400, {"error": "invalid task request"}

        if request.slots.submission.hotkey != self.hotkey_address:
            return 403, {"error": "task is assigned to another miner"}
        if (
            request.identity.execution_profile_id
            != RELEASE_POLICY.v3_execution_profile_id
            or request.identity.verifier_policy != "command-gold-digest-v1"
        ):
            return 422, {"error": "unsupported task policy"}

        timeout_s = min(
            max(0.0, request.expires_at - time.time()),
            self.settings.bedrock_request_timeout_s,
        )
        if timeout_s <= 0:
            return 410, {"error": "task expired"}

        solve_deadline = time.monotonic() + timeout_s

        async def solve_with_slot() -> MinerTaskResponse:
            async with self.solve_slots:
                remaining = solve_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("solve deadline exceeded while queued")
                return await self.solve(request, remaining)

        try:
            payload = await asyncio.wait_for(solve_with_slot(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return 504, {"error": "solve deadline exceeded"}
        except httpx.HTTPStatusError as exc:
            print(f"[demo-miner] model or upload failed: HTTP {exc.response.status_code}")
            return 502, {"error": "upstream request failed"}
        except Exception as exc:  # noqa: BLE001
            print(f"[demo-miner] solve failed: {type(exc).__name__}")
            return 500, {"error": "solve failed"}
        self.served_tasks.add(
            signed_by,
            request.challenge_id,
            request.task_id,
            request.slots.submission.uid,
            request.slots.submission.hotkey,
            time.monotonic(),
        )
        return 200, payload

    async def aclose(self) -> None:
        await self.client.aclose()


def build_demo_miner_app(miner: DemoMiner):
    """Build the FastAPI surface used by validators."""

    from fastapi import FastAPI, Request, Response

    sync_state = {"last": time.monotonic()}
    sync_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await miner.aclose()

    app = FastAPI(title="rlvr-demo-miner", lifespan=lifespan)

    async def maybe_sync_metagraph() -> None:
        if miner.metagraph is None or miner.subtensor is None:
            return
        if (
            time.monotonic() - sync_state["last"]
            < miner.settings.miner_metagraph_sync_s
        ):
            return
        async with sync_lock:
            if (
                time.monotonic() - sync_state["last"]
                < miner.settings.miner_metagraph_sync_s
            ):
                return
            try:
                await asyncio.to_thread(miner.metagraph.sync, subtensor=miner.subtensor)
            except Exception as exc:  # noqa: BLE001 - use last known chain view
                print(
                    "[demo-miner] metagraph refresh failed; "
                    f"using cached view ({type(exc).__name__})"
                )
            finally:
                # Back off after failures too; otherwise every incoming request
                # would trigger another chain RPC while the endpoint is unhealthy.
                sync_state["last"] = time.monotonic()

    async def read_bounded(request: Request, limit: int | None = None) -> Optional[bytes]:
        limit = miner.settings.miner_max_request_bytes if limit is None else limit
        try:
            declared = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            return None
        if declared < 0 or declared > limit:
            return None
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                return None
        return bytes(body)

    @app.post("/solve")
    async def solve_endpoint(request: Request) -> Response:
        await maybe_sync_metagraph()
        body = await read_bounded(request)
        if body is None:
            return Response(
                content=b'{"error":"request body too large"}',
                status_code=413,
                media_type="application/json",
            )

        status, payload = await miner.handle_request(request.headers, body)
        if isinstance(payload, MinerTaskResponse):
            response_body = payload.model_dump_json().encode("utf-8")
        else:
            response_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        response = Response(
            content=response_body,
            status_code=status,
            media_type="application/json",
        )
        if status == 200 and miner.wallet is not None:
            response.headers.update(
                sign_message(
                    miner.wallet,
                    response_body,
                    signed_for=request.headers.get("Epistula-Signed-By", ""),
                )
            )
        return response

    @app.post("/v3/failure")
    async def failure_endpoint(request: Request) -> Response:
        await maybe_sync_metagraph()
        body = await read_bounded(
            request,
            min(miner.settings.miner_max_request_bytes, FAILURE_NOTICE_MAX_BYTES),
        )
        if body is None:
            return Response(
                content=b'{"error":"request body too large"}',
                status_code=413,
                media_type="application/json",
            )
        status, payload = await miner.handle_failure_notice(request.headers, body)
        return Response(
            content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            status_code=status,
            media_type="application/json",
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": miner.settings.bedrock_model}

    return app


def run_demo_miner(settings: Optional[DemoMinerSettings] = None) -> None:
    """Set up the wallet, advertise the endpoint, and serve HTTP."""

    settings = settings or DemoMinerSettings()
    if not settings.bedrock_api_key:
        raise SystemExit("set BEDROCK_API_KEY before starting the demo miner")

    import bittensor as bt  # type: ignore[import-not-found]
    import uvicorn

    wallet = bt.Wallet(name=settings.wallet_name, hotkey=settings.wallet_hotkey)
    network = settings.subtensor_chain_endpoint or settings.subtensor_network
    subtensor = bt.Subtensor(network=network)
    if not subtensor.is_hotkey_registered(
        netuid=settings.netuid,
        hotkey_ss58=wallet.hotkey.ss58_address,
    ):
        raise SystemExit(
            f"hotkey {wallet.hotkey.ss58_address} is not registered "
            f"on netuid {settings.netuid}"
        )
    metagraph = subtensor.metagraph(settings.netuid)

    axon_kwargs: dict[str, Any] = {
        "wallet": wallet,
        "port": settings.axon_port,
    }
    if settings.axon_external_ip:
        axon_kwargs["external_ip"] = settings.axon_external_ip
    axon = bt.Axon(**axon_kwargs)
    axon.serve(netuid=settings.netuid, subtensor=subtensor)

    print(
        f"[demo-miner] serving netuid={settings.netuid} "
        f"wallet={settings.wallet_name}/{settings.wallet_hotkey} "
        f"model={settings.bedrock_model} port={settings.axon_port}"
    )
    miner = DemoMiner(
        settings,
        BedrockClient(settings),
        wallet=wallet,
        subtensor=subtensor,
        metagraph=metagraph,
    )
    uvicorn.run(
        build_demo_miner_app(miner),
        host=settings.axon_host,
        port=settings.axon_port,
        log_level="info",
    )
