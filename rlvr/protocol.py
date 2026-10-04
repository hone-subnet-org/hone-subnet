"""Wire protocol + epistula-style request signing.

This module defines the two pydantic models that cross the wire between the
validator and miners:

* ``TaskRequest``   — validator -> miner: the problem to solve.
* ``SolutionPayload`` — miner -> validator: the candidate solution.

It also provides ``sign_message`` / ``verify_signature``, an epistula-style
(BitMind SN34) request-signing scheme. On a live deployment the signatures are
produced with a real ``substrateinterface``/``bittensor`` ``Keypair`` (sr25519).
When that stack is NOT importable (e.g. local simulation, CI), we transparently
fall back to a deterministic **HMAC-SHA256** signature keyed by the signer's
hotkey string, so the whole protocol round-trips offline without the chain.

Verification is **fail-closed**: the HMAC fallback is only honored when the real
crypto stack is genuinely unavailable AND the claimed signer does not look like a
real ss58 key. If the signer looks like an ss58/real key, or the live crypto
verification raises, verification returns ``False`` rather than silently
downgrading to HMAC — an attacker cannot force the cheap path by spoofing the
``Epistula-Signed-By`` header.

The header layout mirrors BitMind's epistula v2 so a future swap to real crypto
needs no call-site changes:

    Epistula-Version            -> "2"
    Epistula-Timestamp          -> ms since epoch
    Epistula-Uuid               -> per-request nonce
    Epistula-Signed-By          -> signer hotkey (ss58 / opaque id)
    Epistula-Signed-For         -> optional recipient hotkey
    Epistula-Request-Signature  -> "0x" + hex signature over
                                   "{sha256(body)}.{uuid}.{timestamp}.{signed_for}"
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import OrderedDict
from typing import Any, Literal, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .types import SolutionResponse, TestCase

# --------------------------------------------------------------------------- #
# Guarded crypto import. If the bittensor/substrate stack is present we use a
# real sr25519 Keypair; otherwise everything degrades to an HMAC fallback so the
# simulation runs with zero chain dependencies.
# --------------------------------------------------------------------------- #
# Modern bittensor (>=8) ships the Rust-backed `bittensor_wallet` Keypair and no
# longer depends on `substrateinterface`, so we try every known provider before
# falling back. All three expose the same Keypair(ss58_address=...).verify() API.
_Keypair = None  # type: ignore
for _provider in ("substrateinterface", "bittensor_wallet", "bittensor"):
    try:  # pragma: no cover - exercised only when a chain stack is installed
        _Keypair = getattr(__import__(_provider), "Keypair")
        break
    except Exception:  # noqa: BLE001 - any import failure -> try the next
        _Keypair = None  # type: ignore
_HAVE_CRYPTO = _Keypair is not None


EPISTULA_VERSION = "2"
PROMPT_VARIANT_SPACE = 2**31

# Module-level HMAC key. This is NOT a security boundary in fallback mode — it
# just makes the signature deterministic and verifiable without a real keypair.
# In fallback mode the "hotkey" string itself is folded into the HMAC key so two
# different signers produce different signatures.
_FALLBACK_NAMESPACE = b"rlvr-epistula-fallback-v1"


# --------------------------------------------------------------------------- #
# Wire models
# --------------------------------------------------------------------------- #
class TaskRequest(BaseModel):
    """Validator -> miner. Exactly the information a solver is allowed to see.

    Hidden evaluation cases are deliberately NOT included; only
    ``public_examples`` (if any) ship to the miner.
    """

    model_config = ConfigDict(extra="forbid")

    problem_id: str = Field(min_length=1, max_length=256)
    language: Literal["python", "rust"]
    statement: str = Field(min_length=1, max_length=500_000)
    entrypoint: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    public_examples: list[TestCase] = Field(default_factory=list, max_length=1000)
    deadline_s: float = Field(default=60.0, gt=0.0, le=3600.0)
    # Variant id forwarded unchanged to response providers. Its interpretation
    # is outside the wire contract.
    prompt_variant: int = Field(default=0, ge=0, le=PROMPT_VARIANT_SPACE - 1)


class SolutionPayload(BaseModel):
    """Miner -> validator. The candidate solution to a ``TaskRequest``."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str = Field(min_length=1, max_length=256)
    code: str
    raw_response: str = ""


class SignedSolution(BaseModel):
    """Byte-preserving, end-to-end signed miner response artifact."""

    response_body: str = ""
    response_headers: dict[str, str] = Field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    responded: bool = False
    signed_age_s: Optional[float] = None

    def to_solution(self, canonical_problem_id: str) -> SolutionResponse:
        if self.error or not self.response_body:
            return SolutionResponse(
                problem_id=canonical_problem_id,
                code="",
                raw_response=self.error or "<empty miner response>",
                latency_ms=self.latency_ms,
            )
        try:
            payload = SolutionPayload.model_validate_json(self.response_body)
        except Exception as e:  # noqa: BLE001 - invalid artifacts fail closed
            return SolutionResponse(
                problem_id=canonical_problem_id,
                code="",
                raw_response=f"<invalid signed miner response: {e}>",
                latency_ms=self.latency_ms,
            )
        return SolutionResponse(
            problem_id=canonical_problem_id,
            code=payload.code,
            raw_response=payload.raw_response,
            latency_ms=self.latency_ms,
        )


# --------------------------------------------------------------------------- #
# Signing helpers
# --------------------------------------------------------------------------- #
def _body_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _signing_message(body_hash: str, uuid: str, timestamp: int, signed_for: str) -> str:
    return f"{body_hash}.{uuid}.{timestamp}.{signed_for}"


def _hmac_key(signed_by: str) -> bytes:
    return hashlib.sha256(_FALLBACK_NAMESPACE + signed_by.encode("utf-8")).digest()


def _hmac_sign(signed_by: str, message: str) -> str:
    digest = hmac.new(_hmac_key(signed_by), message.encode("utf-8"), hashlib.sha256)
    return "0x" + digest.hexdigest()


def _keypair_address(hotkey: Any) -> str:
    """Best-effort extraction of an address/identifier from a hotkey object.

    Accepts a substrate ``Keypair`` (``.ss58_address``), a wallet-like object
    with ``.hotkey``, or a plain string.
    """
    if hotkey is None:
        return ""
    if isinstance(hotkey, str):
        return hotkey
    # wallet -> wallet.hotkey -> keypair
    inner = getattr(hotkey, "hotkey", None)
    if inner is not None and inner is not hotkey:
        return _keypair_address(inner)
    addr = getattr(hotkey, "ss58_address", None)
    if addr:
        return str(addr)
    return str(hotkey)


def sign_message(
    hotkey: Any,
    body: bytes,
    signed_for: Optional[str] = None,
) -> dict[str, str]:
    """Produce epistula-style signing headers for ``body``.

    Args:
        hotkey: a substrate ``Keypair`` (live) OR a plain string id (fallback)
            OR a wallet-like object exposing ``.hotkey``.
        body:   the raw request bytes that are being signed.
        signed_for: optional recipient hotkey id (binds the signature to a peer).

    Returns a dict of ``Epistula-*`` headers. The signature covers
    ``{sha256(body)}.{uuid}.{timestamp}.{signed_for}``.
    """
    if not isinstance(body, (bytes, bytearray)):
        raise TypeError("body must be bytes")

    timestamp = round(time.time() * 1000)
    uuid = str(uuid4())
    body_hash = _body_hash(bytes(body))
    signed_for = signed_for or ""
    signed_by = _keypair_address(hotkey)
    message = _signing_message(body_hash, uuid, timestamp, signed_for)

    # A bittensor wallet is not itself a signer — its `.hotkey` Keypair is. Unwrap
    # so passing a wallet uses real crypto instead of silently falling back to
    # HMAC (which a live peer would then reject, since Signed-By is a real ss58).
    signer = hotkey
    inner = getattr(signer, "hotkey", None)
    if inner is not None and inner is not signer and hasattr(inner, "sign"):
        signer = inner

    if _HAVE_CRYPTO and not isinstance(signer, str) and hasattr(signer, "sign"):
        # Live path: real sr25519 signature.
        signature = "0x" + signer.sign(message).hex()  # pragma: no cover
    else:
        # Fallback path: deterministic HMAC keyed by the signer id.
        signature = _hmac_sign(signed_by, message)

    return {
        "Epistula-Version": EPISTULA_VERSION,
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": signed_by,
        "Epistula-Signed-For": signed_for,
        "Epistula-Request-Signature": signature,
    }


def _hmac_sign_bytes(signed_by: str, message: bytes) -> str:
    digest = hmac.new(_hmac_key(signed_by), bytes(message), hashlib.sha256)
    return "0x" + digest.hexdigest()


def sign_detached(hotkey: Any, message: bytes) -> str:
    """Sign ``message`` with the hotkey, the way request headers are signed.

    Live path: sr25519 by the hotkey Keypair (a wallet is unwrapped to it).
    Fallback path, only for an opaque signer id when no crypto stack is
    present: the HMAC the request signer uses. Anything else raises, since a
    signature nobody can verify must never be posted.
    """
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    signed_by = _keypair_address(hotkey)
    signer = hotkey
    inner = getattr(signer, "hotkey", None)
    if inner is not None and inner is not signer and hasattr(inner, "sign"):
        signer = inner
    if not isinstance(signer, str) and hasattr(signer, "sign"):
        if not _HAVE_CRYPTO:
            raise RuntimeError("a keypair signer needs the crypto stack")
        signature = "0x" + signer.sign(bytes(message)).hex()  # pragma: no cover - live crypto
        # A keypair of the wrong kind, or one that does not match the address,
        # would produce a signature no reader accepts. Check before it leaves.
        if not verify_detached(signed_by, message, signature):  # pragma: no cover - live crypto
            raise RuntimeError("the hotkey's signature does not verify against its address")
        return signature  # pragma: no cover - live crypto
    if _HAVE_CRYPTO or _looks_like_ss58(signed_by):
        raise RuntimeError("detached signing needs a keypair, not a bare address")
    return _hmac_sign_bytes(signed_by, message)


def verify_detached(signed_by: str, message: bytes, signature: str) -> bool:
    """Verify a detached signature, failing closed exactly like request headers.

    A real ss58 signer is only ever verified with real crypto; an opaque signer
    only with the HMAC fallback, and only when no crypto stack is present.
    """
    if not isinstance(message, (bytes, bytearray)) or not isinstance(signature, str):
        return False
    if not isinstance(signed_by, str) or not signed_by or not signature.startswith("0x"):
        return False
    if _looks_like_ss58(signed_by):
        if not _HAVE_CRYPTO:
            return False
        try:  # pragma: no cover - live crypto path
            keypair = _Keypair(ss58_address=signed_by)  # type: ignore[call-arg,misc]
            raw = bytes(message)
            sig = bytes.fromhex(signature[2:])
            if not keypair.verify(raw, sig):
                return False
            # Keypair.verify also accepts a signature made over
            # "<Bytes>" + message + "</Bytes>" (the browser-wallet wrapping).
            # Such a signature verifies against the wrapped bytes too; one over
            # the exact message does not. Accept only the exact message.
            return not keypair.verify(b"<Bytes>" + raw + b"</Bytes>", sig)
        except Exception:  # noqa: BLE001 - fail closed
            return False
    if _HAVE_CRYPTO:
        return False
    return hmac.compare_digest(_hmac_sign_bytes(signed_by, message), signature)


def verify_signature(
    headers: Mapping[str, str],
    body: bytes,
    *,
    allowed_delta_ms: int = 8000,
    now_ms: Optional[int] = None,
    expected_signed_for: Optional[str] = None,
) -> bool:
    """Verify epistula-style headers against ``body``.

    Returns ``True`` iff the signature is valid, the body hash matches, and the
    timestamp is within ``allowed_delta_ms`` of ``now_ms`` (default: wall clock).

    Recipient binding: if ``expected_signed_for`` is provided (non-empty), the
    request's ``Epistula-Signed-For`` MUST equal it — a request signed for peer
    A is rejected when verified by peer B.

    Verification is **fail-closed** with respect to the crypto path. When the
    claimed ``Epistula-Signed-By`` looks like a real ss58 key:
      * if the live crypto stack is available, only the sr25519 verification is
        honored, and any exception in that path yields ``False`` (no HMAC
        downgrade);
      * if the live crypto stack is NOT available, verification fails closed
        (we refuse to validate a real-key request with the HMAC fallback).
    The HMAC fallback is honored only for non-ss58 (opaque sim) signers when the
    real crypto stack is genuinely unavailable.
    """
    if not isinstance(body, (bytes, bytearray)):
        return False
    try:
        version = headers["Epistula-Version"]
        signature = headers["Epistula-Request-Signature"]
        timestamp = int(headers["Epistula-Timestamp"])
        uuid = headers["Epistula-Uuid"]
        signed_by = headers["Epistula-Signed-By"]
    except (KeyError, TypeError, ValueError):
        return False

    signed_for = headers.get("Epistula-Signed-For", "") or ""
    if (
        version != EPISTULA_VERSION
        or not isinstance(signature, str)
        or not isinstance(signed_by, str)
        or not signed_by
        or not isinstance(uuid, str)
        or not uuid
        or allowed_delta_ms < 0
    ):
        return False

    # Recipient binding: reject a request that was not signed for us.
    if expected_signed_for:
        if signed_for != expected_signed_for:
            return False

    now = now_ms if now_ms is not None else round(time.time() * 1000)
    # Reject stale requests (replay protection). Allow modest clock skew either way.
    if timestamp + allowed_delta_ms < now:
        return False
    if timestamp - allowed_delta_ms > now:
        return False

    body_hash = _body_hash(bytes(body))
    message = _signing_message(body_hash, uuid, timestamp, signed_for)

    # If the signer looks like a real ss58 key, the request MUST be verified with
    # real crypto. We never let an attacker downgrade a real-key request to the
    # HMAC path by setting Signed-By to an ss58 address.
    if _looks_like_ss58(signed_by):
        if not (_HAVE_CRYPTO and signature.startswith("0x")):
            return False  # fail closed: no crypto stack (or non-hex sig) -> reject
        try:  # pragma: no cover - live crypto path
            keypair = _Keypair(ss58_address=signed_by)  # type: ignore[call-arg,misc]
            sig_bytes = bytes.fromhex(signature[2:])
            return bool(keypair.verify(message, sig_bytes))
        except Exception:  # noqa: BLE001 - fail closed, never fall back to HMAC
            return False

    # Opaque (non-ss58) signer: HMAC fallback, but only when real crypto is truly
    # unavailable. With a live crypto stack present we do not accept HMAC at all.
    if _HAVE_CRYPTO:
        return False
    expected = _hmac_sign(signed_by, message)
    return hmac.compare_digest(expected, signature)


def _looks_like_ss58(value: str) -> bool:
    """Heuristic: ss58 addresses are base58, ~47-48 chars, no '0x' prefix."""
    alphabet = frozenset(
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    )
    return (
        isinstance(value, str)
        and 46 <= len(value) <= 49
        and not value.startswith("0x")
        and all(char in alphabet for char in value)
    )


def crypto_available() -> bool:
    """True if the real substrate/bittensor signing stack is importable."""
    return _HAVE_CRYPTO


# --------------------------------------------------------------------------- #
# Replay / nonce cache
# --------------------------------------------------------------------------- #
class NonceCache:
    """Bounded, time-windowed cache of seen Epistula-Uuids for replay defense.

    A signature stays valid for ``allowed_delta_ms`` of clock skew, which is
    exactly the window in which a captured-and-replayed request would still pass
    the timestamp check. This cache records each request's uuid for that window
    so a replay (same uuid, still-fresh timestamp) is rejected.

    ``check_and_add(uuid, now_ms)`` returns ``True`` for a first-seen uuid when
    capacity is available (and records it). Replays and new requests arriving
    while every slot is occupied by a fresh nonce fail closed. Entries older
    than ``window_ms`` are pruned lazily on each call. Thread-safe.
    """

    def __init__(self, window_ms: int = 8000, max_entries: int = 100_000):
        self.window_ms = max(0, int(window_ms))
        self.max_entries = max(1, int(max_entries))
        self._seen: "OrderedDict[str, int]" = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_ms
        # Entries are inserted in (roughly) increasing time order; drop the
        # stale prefix. Fresh entries are never evicted merely to make room:
        # doing so would let a signed request flood churn the cache and replay
        # an evicted nonce while its signature timestamp is still valid.
        while self._seen:
            uuid, ts = next(iter(self._seen.items()))
            if ts < cutoff:
                self._seen.popitem(last=False)
            else:
                break

    def check_and_add(self, uuid: str, now_ms: Optional[int] = None) -> bool:
        """Record a fresh UUID, rejecting replays and capacity saturation."""
        if not isinstance(uuid, str) or not uuid:
            return False
        now = now_ms if now_ms is not None else round(time.time() * 1000)
        with self._lock:
            self._prune(now)
            if uuid in self._seen:
                return False
            if len(self._seen) >= self.max_entries:
                return False
            self._seen[uuid] = now
            return True
