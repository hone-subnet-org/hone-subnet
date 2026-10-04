"""Number, sign and keep each round for the shared ledger until the server has it.

With pooled scoring a lost round is a lost round for every validator, so a
signed round is written to disk before the first send and re-sent, byte for
byte, until the server accepts it, already holds it, or the round's grading
window has closed. Feedback is a separate call and is not touched here.

Ordering on disk: the round's record is written first, then the counter, so a
crash in between repeats neither. The next number is always one past the
highest number on disk, whether that is the counter or a kept record. One
lock in the process and one lock on the directory keep numbering, discovery
and removal in order, whoever holds a handle to the directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..protocol import _keypair_address
from .api import RoundVerdict, SignedRound, serialize_signed_round
from .verdicts import sign_round, verify_round

Delivery = Literal["delivered", "retry", "dropped"]

SEND_TIMEOUT_S = 60.0  # one attempt, connect to last byte; a server trickling a reply cannot hold the worker
RETRY_S = 60.0  # between passes while a round is still kept

_RECORD = re.compile(r"^round-(\d{12,16})\.json$")  # zero-padded to 12; the wire allows up to 2**53
RECORD_FORMAT = 1  # bump, and migrate in _load, if a kept record ever changes shape


@dataclass(frozen=True)
class KeptRound:
    seq: int
    body: bytes  # the exact bytes signed and sent, every time
    expires_at: int


class RoundLedger:
    def __init__(self, signer: Any, directory: str | os.PathLike[str]) -> None:
        if signer is None:
            raise ValueError("the ledger needs the validator's wallet to sign rounds")
        self.signer = signer
        self.address = _keypair_address(signer)
        self.directory = Path(directory)  # created on first write, so startup never fails on it
        self._lock = threading.RLock()
        self._depth = 0
        self._handle: int | None = None
        self._wake: asyncio.Event | None = None

    # ---- one holder at a time, in this process and across processes
    @contextlib.contextmanager
    def _held(self) -> Iterator[None]:
        with self._lock:
            if self._depth == 0:
                _ensure_directory(self.directory)
                handle = os.open(self.directory, os.O_RDONLY)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                except BaseException:
                    os.close(handle)
                    raise
                self._handle = handle
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._handle is not None:
                    handle, self._handle = self._handle, None
                    try:
                        fcntl.flock(handle, fcntl.LOCK_UN)
                    finally:
                        os.close(handle)  # closing drops the lock even if unlocking failed

    # ---- one worker sends everything kept, oldest first
    async def run(self, client: Any) -> None:
        """Send kept rounds until the server has them: a pass right away, a
        pass whenever a round is prepared, and a pass every RETRY_S while
        anything is still kept. Cancel it before the HTTP client closes."""
        self._wake = asyncio.Event()
        while True:
            self._wake.clear()  # a wake during the pass asks for another pass
            try:
                still_pending = await self.retry_pending(client)
            except Exception:  # noqa: BLE001 - the worker outlives any one failure
                still_pending = True
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=RETRY_S if still_pending else None)

    def notify(self) -> None:
        """A round was prepared: send it now, without holding the round up."""
        if self._wake is not None:
            self._wake.set()

    # ---- the validator's own round sequence
    def last_seq(self) -> int:
        """The highest number used so far: the counter, or a kept record the
        counter did not catch up with before a crash."""
        with self._held():
            counter = 0
            try:
                counter = int((self.directory / "seq").read_text(encoding="utf-8").strip() or 0)
            except (FileNotFoundError, ValueError):
                pass
            kept = [_seq_of(path) for path in self.pending()]
            return max([counter, *kept])

    # ---- a round, numbered, signed and kept
    def prepare(self, challenge_id: str, task_id: str, verdicts: list[RoundVerdict], *, expires_at: int) -> KeptRound:
        ordered = sorted(verdicts, key=lambda item: item.uid)
        with self._held():
            seq = self.last_seq() + 1
            report = SignedRound(
                protocol_version=3,
                challenge_id=challenge_id,
                task_id=task_id,
                round_seq=seq,
                verdicts=ordered,
                signature=sign_round(self.signer, challenge_id, task_id, seq, ordered),
            )
            body = serialize_signed_round(report)
            record = {"format": RECORD_FORMAT, "expires_at": int(expires_at), "body": body.decode("utf-8")}
            _write_atomic(self._path(seq), json.dumps(record).encode("utf-8"))
            _write_atomic(self.directory / "seq", str(seq).encode("utf-8"))
            return KeptRound(seq, body, int(expires_at))

    def _path(self, seq: int) -> Path:
        return self.directory / f"round-{seq:012d}.json"

    def pending(self) -> list[Path]:
        """Kept records, oldest first. Anything else in the directory is not a
        record: leftovers of interrupted writes are removed, other files ignored."""
        with self._held():
            if not self.directory.is_dir():
                return []
            # A record left by a run that died between its rename and the
            # directory sync is visible but not yet durable: make it so before
            # anyone acts on it.
            _fsync_chain(self.directory)
            records = []
            for path in sorted(self.directory.iterdir()):
                if _RECORD.match(path.name):
                    records.append(path)
                elif path.name.startswith(".") and path.name.endswith(".tmp"):
                    path.unlink(missing_ok=True)
            return records

    def _load(self, path: Path) -> KeptRound | None:
        """A kept record, checked as the server would check it: the filename's
        number, the signature by this validator's own hotkey, the window."""
        seq = _seq_of(path)
        raw = path.read_bytes()  # a read error is the caller's to retry, not a bad record
        try:
            record = json.loads(raw.decode("utf-8"))
            if record.get("format") != RECORD_FORMAT:
                raise ValueError(f"record format {record.get('format')!r}")
            body = record["body"].encode("utf-8")
            if type(record["expires_at"]) is not int:
                raise ValueError("expires_at")
            report = SignedRound.model_validate_json(body)
            if report.round_seq != seq:
                raise ValueError("record number")
            if not verify_round(self.address, report.signature, report.challenge_id, report.task_id, seq, report.verdicts):
                raise ValueError("signature")
            return KeptRound(seq, body, record["expires_at"])
        except Exception as error:  # noqa: BLE001 - contents the server would refuse are never resent
            print(f"[validator] WARN: dropping kept round {path.name}: {type(error).__name__}: {error}")
            self._forget(seq)
            return None

    async def deliver(self, client: Any, kept: KeptRound, *, now: float | None = None) -> Delivery:
        """One attempt with the kept bytes. The record goes away once the
        server has the round, or once sending it again could never help."""
        if kept.expires_at <= (time.time() if now is None else now):
            print(f"[validator] WARN: round {kept.seq} expired before the server accepted it")
            await asyncio.to_thread(self._forget, kept.seq)
            return "dropped"
        try:
            status = await asyncio.wait_for(client.signed_round_status(kept.body), timeout=SEND_TIMEOUT_S)
        except asyncio.TimeoutError:
            print(f"[validator] WARN: round {kept.seq}: no answer within {SEND_TIMEOUT_S:g} s; kept for retry")
            status = None
        if status == 409:
            # The server holds a round for this number or this challenge
            # already, or the challenge was never ours. Sending again cannot
            # help. Repeated conflicts mean the counter file was lost: rounds
            # are refused until the numbering passes the server's.
            print(f"[validator] WARN: round {kept.seq} conflicts with what the server holds (HTTP 409); stopped")
        if status in (200, 409):
            await asyncio.to_thread(self._forget, kept.seq)
            return "delivered"
        if status in (400, 410):
            why = "was refused as invalid" if status == 400 else "missed the grading window"
            print(f"[validator] WARN: round {kept.seq} {why} (HTTP {status}); dropped")
            await asyncio.to_thread(self._forget, kept.seq)
            return "dropped"
        return "retry"

    async def retry_pending(self, client: Any, *, now: float | None = None) -> bool:
        """Resend every kept round once. Returns True while any is still kept.
        One bad round never stops the others."""
        remaining = False
        for path in await asyncio.to_thread(self.pending):
            try:
                kept = await asyncio.to_thread(self._load, path)
                if kept is None:
                    continue
                outcome = await self.deliver(client, kept, now=now)
            except Exception:  # noqa: BLE001 - one bad round or send; try again next pass
                outcome = "retry"
            remaining = remaining or outcome == "retry"
        return remaining

    def _forget(self, seq: int) -> None:
        """Remove a kept record, first making sure the counter has caught up
        with it: a record is the only proof a number was used after a crash."""
        with self._held():
            counter = 0
            try:
                counter = int((self.directory / "seq").read_text(encoding="utf-8").strip() or 0)
            except (FileNotFoundError, ValueError):
                pass
            if seq > counter:
                _write_atomic(self.directory / "seq", str(seq).encode("utf-8"))
            else:
                # The counter already covers this number, but it may have been
                # renamed into place by a run that died before syncing the
                # directory. Make it durable before the record goes.
                _fsync_chain(self.directory)
            self._path(seq).unlink(missing_ok=True)


def _seq_of(path: Path) -> int:
    match = _RECORD.match(path.name)
    if match is None:
        raise ValueError(f"not a kept round: {path.name}")
    return int(match.group(1))


def _write_atomic(path: Path, data: bytes) -> None:
    """Write, fsync, rename, then fsync the directory chain, so neither the
    data nor the name is lost to a crash. A write that fails leaves nothing."""
    _ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_chain(path.parent)


def _fsync_chain(directory: Path) -> None:
    """Sync the outbox directory and the two directories above it, every
    time: an earlier run may have created one of them and died before its
    entry was synced. The outbox lives at data/feedback-outbox under the
    validator's working directory, so this reaches a directory that existed
    long before the ledger did; anything above that is the operator's
    filesystem."""
    for level in (directory, directory.parent, directory.parent.parent):
        _fsync_directory(level)


def _ensure_directory(directory: Path) -> None:
    """Create every missing level of ``directory`` one at a time, fsyncing
    each parent, so no level can be lost to a crash."""
    missing = []
    current = directory
    while not current.is_dir():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for level in reversed(missing):
        level.mkdir(mode=0o700, exist_ok=True)
        _fsync_directory(level.parent)


def _fsync_directory(directory: Path) -> None:
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
