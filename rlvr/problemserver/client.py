"""Validator-side HTTPS client for the private problem-source protocol."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from ..protocol import sign_message
from .api import (
    ChallengeCommitRequest,
    ChallengeCommitResponse,
    ChallengeFeedback,
    ChallengeLeaseRequest,
    ChallengeRevealRequest,
    ChallengeRevealResponse,
    MinerSubmission,
    PublicChallenge,
)


def require_secure_problem_url(url: str, allow_insecure_http: bool = False) -> None:
    parsed = urlsplit(url)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if allow_insecure_http and parsed.scheme == "http" and parsed.netloc:
        return
    raise ValueError(
        "PROBLEM_SERVER_URL must use https://; set "
        "PROBLEM_SERVER_ALLOW_INSECURE_HTTP=true only for local testing"
    )


# Remote-supplied diagnostic text reaches operator log lines verbatim, so its
# length is bounded here. Note that Retry-After is deliberately NOT bounded:
# see parse_retry_after.
_MAX_DETAIL_CHARS = 200
_RETRY_AFTER_CUSHION_S = 1.0

# Transient conditions worth another attempt: the server could not complete the
# request, but nothing about the request itself was refused. 5xx joins these by
# range. One source of truth, because post_result retries exactly this set and
# status_category must label exactly the same set as UNAVAILABLE.
_RETRYABLE_STATUSES = frozenset({408, 425})


class LeaseCategory(str, Enum):
    """Why one lease attempt did or did not yield a challenge.

    The non-success causes were previously indistinguishable at the call site:
    an exhausted problem pool, a paced lease, an unauthorized validator, a
    rejected request, a failing server and an unreachable one all collapsed to
    ``None``. Each demands a different operator response, so each is its own
    category — in particular UNAVAILABLE (the server answered, and the answer
    was a failure) is never merged with TRANSPORT (nothing answered at all).
    """

    LEASED = "leased"
    EMPTY = "empty"
    PACED = "paced"
    DENIED = "denied"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    TRANSPORT = "transport"
    MALFORMED = "malformed"


_CATEGORY_REASONS = {
    LeaseCategory.LEASED: "challenge leased",
    LeaseCategory.EMPTY: "problem source has no challenge available",
    LeaseCategory.PACED: "lease paced by the problem source",
    LeaseCategory.DENIED: "validator not authorized to lease",
    LeaseCategory.REJECTED: "problem source rejected the lease request",
    LeaseCategory.UNAVAILABLE: "problem source could not serve the lease",
    LeaseCategory.TRANSPORT: "problem source unreachable",
    LeaseCategory.MALFORMED: "problem source returned an unusable challenge",
}


def bounded_detail(text: str) -> str:
    """Collapse remote text into one bounded, printable log line."""
    collapsed = " ".join(text.split())
    printable = "".join(ch for ch in collapsed if ch.isprintable())
    if len(printable) > _MAX_DETAIL_CHARS:
        return printable[: _MAX_DETAIL_CHARS - 3] + "..."
    return printable


def response_detail(response: httpx.Response) -> str:
    """Best-effort server explanation, tolerant of non-JSON and malformed bodies."""
    body = response.content.decode("utf-8", "replace").strip()
    if body.startswith("{"):
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001 - a broken error body is still a detail
            payload = None
        if isinstance(payload, dict):
            for key in ("detail", "error", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return bounded_detail(value)
    return bounded_detail(body)


# RFC 9110 delta-seconds is 1*DIGIT, where DIGIT is ASCII 0-9 and nothing else.
# `int()` is far more permissive than that grammar — it accepts a leading sign,
# underscores, and every Unicode decimal digit — so the header is matched
# against the grammar first and only then converted.
_DELTA_SECONDS = re.compile(r"[0-9]+")


def parse_retry_after(response: httpx.Response) -> Optional[int]:
    """Parse Retry-After fail-closed: whole non-negative seconds or nothing.

    The HTTP-date form and every malformed value stay ``None`` so a caller can
    fall back to its own pacing instead of acting on a misread deadline. A
    syntactically valid value is returned exactly as sent, however large: it is
    a server-directed pacing instruction, and silently shortening it would make
    the validator retry earlier than the server asked. Bounding what we obey is
    a pacing policy decision that belongs to the caller, not to the parser.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    # Only SP and HTAB are optional whitespace in HTTP; str.strip() would also
    # eat Unicode whitespace that has no business in a header value.
    candidate = raw.strip(" \t")
    if not _DELTA_SECONDS.fullmatch(candidate):
        return None
    return int(candidate)


def next_lease_not_before(
    retry_after_s: int, *, now: Optional[float] = None
) -> float:
    """Convert whole-second pacing into a cushioned monotonic deadline."""
    current = time.monotonic() if now is None else float(now)
    return current + retry_after_s + _RETRY_AFTER_CUSHION_S


def status_category(status: int) -> LeaseCategory:
    if status == 204:
        return LeaseCategory.EMPTY
    if status == 429:
        return LeaseCategory.PACED
    if status in (401, 403):
        return LeaseCategory.DENIED
    # Reached only after post_result has retried this status to exhaustion:
    # the server kept answering, and kept failing.
    if status >= 500 or status in _RETRYABLE_STATUSES:
        return LeaseCategory.UNAVAILABLE
    # Everything else, 404 included. A 404 could mean an exhausted pool or a
    # wrong route, which call for opposite operator responses, so it keeps its
    # raw status and the server's own words rather than a confident guess.
    return LeaseCategory.REJECTED


@dataclass(frozen=True)
class LeaseOutcome:
    """Typed result of one lease attempt.

    ``status`` is ``None`` exactly when no HTTP response survived transport
    retries. Callers never have to infer the cause from that alone: ``category``
    states it outright. Remote and local details are bounded before they can
    reach an operator log.
    """

    category: LeaseCategory
    challenge: Optional[PublicChallenge] = None
    status: Optional[int] = None
    detail: str = ""
    retry_after_s: Optional[int] = None
    retry_after_from_prior_response: bool = False
    transport_error: str = ""
    detail_is_server: bool = True

    @property
    def leased(self) -> bool:
        return self.challenge is not None

    def describe(self) -> str:
        """One-line, cause-specific explanation for validator logs."""
        parts = [_CATEGORY_REASONS[self.category]]
        if self.status is not None:
            status_label = "last HTTP" if self.transport_error else "HTTP"
            parts.append(f"{status_label} {self.status}")
        if self.retry_after_s is not None:
            prior = (
                " (from an earlier response)"
                if self.retry_after_from_prior_response
                else ""
            )
            parts.append(f"retry after {self.retry_after_s}s{prior}")
        if self.detail and self.detail != self.transport_error:
            label = "server said" if self.detail_is_server else "detail"
            parts.append(f"{label}: {self.detail}")
        if self.transport_error:
            parts.append(f"transport error: {self.transport_error}")
        return "; ".join(parts)


@dataclass(frozen=True)
class PostOutcome:
    """One bounded POST result, including an exhausted retryable response."""

    response: Optional[httpx.Response] = None
    transport_error: str = ""
    protocol_error: str = ""
    retry_after_s: Optional[int] = None
    retry_after_from_prior_response: bool = False
    retry_exhausted: bool = False


class ProblemServerClient:
    def __init__(
        self,
        url: str,
        wallet,
        http: httpx.AsyncClient,
        *,
        allow_insecure_http: bool = False,
        timeout_s: float = 60.0,
        retries: int = 3,
        max_response_bytes: int = 2 * 1024 * 1024,
    ):
        require_secure_problem_url(url, allow_insecure_http)
        self._url = url.rstrip("/")
        self._wallet = wallet
        self._http = http
        self._timeout = timeout_s
        self._retries = max(1, int(retries))
        self._max_response_bytes = max(1, int(max_response_bytes))

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        declared = response.headers.get("content-length")
        if declared:
            try:
                if int(declared) < 0 or int(declared) > self._max_response_bytes:
                    raise RuntimeError("problem-server response exceeds byte limit")
            except ValueError as error:
                raise RuntimeError("invalid problem-server content-length") from error
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self._max_response_bytes:
                raise RuntimeError("problem-server response exceeds byte limit")
        return bytes(body)

    async def post(self, path: str, body: bytes) -> Optional[httpx.Response]:
        outcome = await self.post_result(path, body)
        # Preserve the pre-existing contract for commit/reveal/feedback: an
        # exhausted retryable HTTP response remains a failed POST there. Lease
        # consumes the richer result so it can expose the final wire status.
        if outcome.retry_exhausted or outcome.protocol_error:
            return None
        return outcome.response

    async def post_result(
        self, path: str, body: bytes
    ) -> PostOutcome:
        """POST with retries while retaining the final bounded HTTP response.

        A retryable response and a transport exception are deliberately
        different outcomes. A late transport exception retains the last HTTP
        evidence while still identifying transport as the final failure.
        """
        last_response: Optional[httpx.Response] = None
        last_protocol_error = ""
        last_retry_after_s: Optional[int] = None
        last_retry_after_attempt: Optional[int] = None
        for attempt in range(1, self._retries + 1):
            headers = sign_message(self._wallet, body)
            headers["Content-Type"] = "application/json"
            try:
                async with self._http.stream(
                    "POST",
                    f"{self._url}{path}",
                    content=body,
                    headers=headers,
                    timeout=self._timeout,
                ) as streamed:
                    status_code = streamed.status_code
                    response_headers = streamed.headers
                    request = streamed.request
                    try:
                        content = await self._read_bounded(streamed)
                    except RuntimeError as error:
                        # Headers and status arrived, so this is a local protocol
                        # refusal, not a transport failure. Do not retry: a 200
                        # lease may already have consumed a scarce challenge.
                        rewrapped_headers = {
                            name: value
                            for name, value in response_headers.items()
                            if name.lower()
                            not in (
                                "content-encoding",
                                "content-length",
                                "transfer-encoding",
                            )
                        }
                        response = httpx.Response(
                            status_code=status_code,
                            headers=rewrapped_headers,
                            content=b"",
                            request=request,
                        )
                        last_response = response
                        last_protocol_error = bounded_detail(str(error))
                        parsed_retry_after = parse_retry_after(response)
                        if parsed_retry_after is not None:
                            last_retry_after_s = parsed_retry_after
                            last_retry_after_attempt = attempt
                        retryable = (
                            status_code >= 500
                            or status_code in _RETRYABLE_STATUSES
                        )
                        if retryable and attempt < self._retries:
                            await asyncio.sleep(
                                min(0.25 * (2 ** (attempt - 1)), 2.0)
                            )
                            continue
                        return PostOutcome(
                            response=response,
                            protocol_error=last_protocol_error,
                            retry_after_s=last_retry_after_s,
                            retry_after_from_prior_response=(
                                last_retry_after_attempt is not None
                                and last_retry_after_attempt != attempt
                            ),
                            retry_exhausted=retryable,
                        )
                # Return a normal in-memory response only after the streaming
                # cap has been enforced; callers keep their existing parsing API.
                # `content` is already transport-decoded by aiter_bytes, so the
                # framing headers must NOT ride along: keeping content-encoding
                # would make httpx re-apply the decoder to plaintext and raise
                # DecodingError behind any gzip-enabled server or proxy.
                rewrapped_headers = {
                    name: value
                    for name, value in response_headers.items()
                    if name.lower()
                    not in ("content-encoding", "content-length", "transfer-encoding")
                }
                response = httpx.Response(
                    status_code=status_code,
                    headers=rewrapped_headers,
                    content=content,
                    request=request,
                )
                last_response = response
                parsed_retry_after = parse_retry_after(response)
                if parsed_retry_after is not None:
                    last_retry_after_s = parsed_retry_after
                    last_retry_after_attempt = attempt
                retryable = status_code >= 500 or status_code in _RETRYABLE_STATUSES
                if retryable:
                    if attempt == self._retries:
                        print(
                            f"[validator] WARN: problem server {path} failed: "
                            f"HTTP {status_code}"
                        )
                        return PostOutcome(
                            response=response,
                            retry_after_s=last_retry_after_s,
                            retry_after_from_prior_response=(
                                last_retry_after_attempt is not None
                                and last_retry_after_attempt != attempt
                            ),
                            retry_exhausted=True,
                        )
                    await asyncio.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0))
                    continue
                return PostOutcome(response=response)
            except Exception as e:  # noqa: BLE001 - retry transient network faults
                if attempt == self._retries:
                    print(f"[validator] WARN: problem server {path} failed: {e}")
                    return PostOutcome(
                        response=last_response,
                        transport_error=str(e).strip() or type(e).__name__,
                        protocol_error=last_protocol_error,
                        retry_after_s=last_retry_after_s,
                        retry_after_from_prior_response=(
                            last_retry_after_attempt is not None
                        ),
                        retry_exhausted=True,
                    )
                await asyncio.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0))
        return PostOutcome(transport_error="retries exhausted")

    async def lease(self) -> LeaseOutcome:
        """Attempt one lease and report the outcome, never a bare ``None``."""
        # Stable body across HTTP retries makes generation idempotent.
        body = ChallengeLeaseRequest(request_id=uuid4().hex).model_dump_json().encode()
        post = await self.post_result("/v1/challenges/lease", body)
        response = post.response
        if response is None:
            transport_error = bounded_detail(post.transport_error)
            return LeaseOutcome(
                category=LeaseCategory.TRANSPORT,
                detail=transport_error,
                transport_error=transport_error,
                detail_is_server=False,
            )
        status = response.status_code
        # Parsed for every response, not just 429: a server may pace with any
        # status, and an unparseable header simply stays None.
        retry_after_s = (
            post.retry_after_s
            if post.retry_after_s is not None
            else parse_retry_after(response)
        )
        if post.transport_error:
            return LeaseOutcome(
                category=LeaseCategory.TRANSPORT,
                status=status,
                detail=post.protocol_error or response_detail(response),
                retry_after_s=retry_after_s,
                retry_after_from_prior_response=(
                    post.retry_after_from_prior_response
                ),
                transport_error=bounded_detail(post.transport_error),
                detail_is_server=not bool(post.protocol_error),
            )
        if post.protocol_error:
            category = (
                LeaseCategory.MALFORMED
                if status == 200
                else status_category(status)
            )
            return LeaseOutcome(
                category=category,
                status=status,
                detail=post.protocol_error,
                retry_after_s=retry_after_s,
                retry_after_from_prior_response=(
                    post.retry_after_from_prior_response
                ),
                detail_is_server=False,
            )
        if status != 200:
            return LeaseOutcome(
                category=status_category(status),
                status=status,
                detail=response_detail(response),
                retry_after_s=retry_after_s,
                retry_after_from_prior_response=(
                    post.retry_after_from_prior_response
                ),
            )
        try:
            challenge = PublicChallenge.model_validate_json(response.content)
        except Exception as e:  # noqa: BLE001
            return LeaseOutcome(
                category=LeaseCategory.MALFORMED,
                status=status,
                detail=bounded_detail(str(e)),
                retry_after_s=retry_after_s,
                detail_is_server=False,
            )
        return LeaseOutcome(
            category=LeaseCategory.LEASED,
            challenge=challenge,
            status=status,
            retry_after_s=retry_after_s,
            retry_after_from_prior_response=post.retry_after_from_prior_response,
        )

    async def commit(
        self, challenge_id: str, submissions: list[MinerSubmission]
    ) -> Optional[ChallengeCommitResponse]:
        body = ChallengeCommitRequest(
            challenge_id=challenge_id, submissions=submissions
        ).model_dump_json().encode()
        response = await self.post("/v1/challenges/commit", body)
        if response is None:
            return None
        try:
            return ChallengeCommitResponse.model_validate_json(response.content)
        except Exception as e:  # noqa: BLE001
            print(f"[validator] WARN: invalid commit receipt: {e}")
            return None

    async def reveal(
        self, challenge_id: str, commit_token: str
    ) -> Optional[ChallengeRevealResponse]:
        body = ChallengeRevealRequest(
            challenge_id=challenge_id, commit_token=commit_token
        ).model_dump_json().encode()
        response = await self.post("/v1/challenges/reveal", body)
        if response is None or response.status_code != 200:
            return None
        try:
            reveal = ChallengeRevealResponse.model_validate_json(response.content)
        except Exception as e:  # noqa: BLE001
            print(f"[validator] WARN: invalid test reveal: {e}")
            return None
        return reveal

    async def feedback(self, feedback: ChallengeFeedback) -> bool:
        body = feedback.model_dump_json().encode()
        response = await self.post("/v1/challenges/feedback", body)
        if response is None or response.status_code != 200:
            return False
        try:
            return bool(response.json().get("accepted"))
        except Exception:  # noqa: BLE001
            return False
