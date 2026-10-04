"""Pooled scoring over the shared ledger of signed rounds.

Other validators' rounds are admitted when their signature verifies and the
signer is validating and holds enough stake. Every admitted round counts
equally, one validator contributes at most a fixed number of rounds per
miner inside the window, and a miner's score is the plain mean of its own
local observations and the admitted ones. With nothing admitted for a miner
the score is exactly the local one, so a validator that cannot reach the
ledger scores as it always did.

Replay is refused by number alone: each validator's rounds are numbered
upward, and the pool keeps, per validator, the numbers it has taken inside a
fixed slack below the newest. The server's timestamp only decides whether a
round is inside the scoring window.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import LedgerPage, LedgerRound, RoundVerdict
from .verdicts import verify_round

STATE_FORMAT = 1
FUTURE_SLACK_S = 600  # a round recorded further ahead of our clock than this is refused
# A validator's numbers only go up, one per round, and a round the server
# receives late arrives a few numbers behind its newest, never a hundred. Per
# validator the pool remembers the numbers taken inside this slack below the
# newest; a number already taken, or further below, is an old round shown
# again, however fresh its recorded_at says it is.
SEQ_SLACK = 100
Entry = tuple[float, float, int]  # recorded_at, payment, round_seq


def round_payments(
    verdicts: Iterable[RoundVerdict], *, speed_half_life_ms: float, speed_floor: float
) -> dict[tuple[int, str], float]:
    """What local scoring would have paid each miner for this round: a miss
    is 0, a pass is 1 scaled by how far behind the round's fastest pass it
    came. A pass with no usable latency gets the floor."""
    items = list(verdicts)
    fastest = min(
        (item.response_latency_ms for item in items if item.passed and item.response_latency_ms), default=None
    )
    floor = min(1.0, max(0.0, float(speed_floor)))
    half_life = float(speed_half_life_ms)
    payments: dict[tuple[int, str], float] = {}
    for item in items:
        if not item.passed:
            payments[(item.uid, item.hotkey)] = 0.0
        elif fastest is None or not math.isfinite(half_life) or half_life <= 0:
            payments[(item.uid, item.hotkey)] = 1.0
        elif not item.response_latency_ms:
            payments[(item.uid, item.hotkey)] = floor
        else:
            delay = max(0, item.response_latency_ms - fastest)
            payments[(item.uid, item.hotkey)] = floor + (1.0 - floor) * (2.0 ** (-delay / half_life))
    return payments


def parse_recorded_at(value: str) -> float:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class LedgerPool:
    def __init__(
        self,
        *,
        own_hotkey: str,
        cap_per_validator: int,
        window_s: int,
        speed_half_life_ms: float,
        speed_floor: float,
    ) -> None:
        self.own_hotkey = own_hotkey
        self.cap = int(cap_per_validator)
        self.window_s = int(window_s)
        self.speed_half_life_ms = float(speed_half_life_ms)
        self.speed_floor = float(speed_floor)
        self.cursor: str | None = None
        # validator hotkey -> (uid, miner hotkey) -> newest entries, oldest first
        self.entries: dict[str, dict[tuple[int, str], deque[Entry]]] = {}
        # validator hotkey -> the round numbers taken inside the slack below its newest; never forgotten
        self.seen: dict[str, set[int]] = {}

    # ---- ingest
    def ingest(
        self,
        page: LedgerPage,
        *,
        admitted: Callable[[str], bool],
        hotkeys: Sequence[str],
        now: float,
    ) -> tuple[int, int]:
        """Take one page. Returns (rounds taken, rounds dropped). Our own
        rounds and numbers already taken are skipped silently; a round that
        is refused, unverified or numbered below the slack is dropped. Only
        verdicts about a miner registered as named right now are kept, so a
        round can never create more buckets than there are registrations."""
        taken = dropped = 0
        current = {(uid, hotkey) for uid, hotkey in enumerate(hotkeys)}
        self.settle(hotkeys, now, admitted=admitted)
        for item in page.rounds:
            validator, seq = item.validator_hotkey, item.round_seq
            if validator == self.own_hotkey or seq in self.seen.get(validator, ()):
                continue
            newest = self.newest_seq(validator)
            if seq + SEQ_SLACK <= newest:
                dropped += 1  # numbered far below what we already hold: an old round shown again
                continue
            try:
                recorded = parse_recorded_at(item.recorded_at)
            except ValueError:
                recorded = math.nan
            if not 0 <= recorded <= now + FUTURE_SLACK_S or not admitted(validator) or not self._verified(item):
                dropped += 1
                continue
            self._remember(validator, seq)
            if recorded <= now - self.window_s:
                continue  # verified and remembered, but already outside the window: nothing to score
            per_miner = self.entries.setdefault(validator, {})
            for registration, payment in round_payments(
                item.verdicts, speed_half_life_ms=self.speed_half_life_ms, speed_floor=self.speed_floor
            ).items():
                if registration not in current:
                    continue
                bucket = per_miner.setdefault(registration, deque(maxlen=self.cap))
                # the newest cap entries by the server's clock, whatever order pages arrive in
                kept = sorted([*bucket, (recorded, payment, seq)], key=lambda e: (e[0], e[2]))[-self.cap :]
                bucket.clear()
                bucket.extend(kept)
            taken += 1
        self.cursor = page.next_cursor or self.cursor
        return taken, dropped

    def newest_seq(self, validator: str) -> int:
        return max(self.seen.get(validator, ()), default=0)

    def _remember(self, validator: str, seq: int) -> None:
        numbers = self.seen.get(validator, set()) | {seq}
        newest = max(numbers)
        self.seen[validator] = {n for n in numbers if n + SEQ_SLACK > newest}

    @staticmethod
    def _verified(item: LedgerRound) -> bool:
        return verify_round(
            item.validator_hotkey, item.signature, item.challenge_id, item.task_id, item.round_seq, item.verdicts
        )

    def prune(self, now: float) -> None:
        """Forget every entry older than the window."""
        cutoff = now - self.window_s
        for per_miner in self.entries.values():
            for registration in list(per_miner):
                kept = [entry for entry in per_miner[registration] if entry[0] > cutoff]
                if kept:
                    per_miner[registration] = deque(kept, maxlen=self.cap)
                else:
                    del per_miner[registration]
        for validator in [v for v, per_miner in self.entries.items() if not per_miner]:
            del self.entries[validator]

    def settle(
        self, hotkeys: Sequence[str], now: float, *, admitted: Callable[[str], bool] | None = None
    ) -> None:
        """Bring the pool to the present: forget what is outside the window,
        what belongs to registrations that no longer exist and, when told who
        is admitted now, the entries of validators that no longer are. The
        numbers taken are never forgotten, so a validator that leaves and
        returns cannot have its old rounds shown again. Call before asking
        whether anything is pooled."""
        self.prune(now)
        self._reset_replaced(hotkeys)
        if admitted is not None:
            for validator in [v for v in self.entries if not admitted(v)]:
                del self.entries[validator]  # its numbers stay remembered, so nothing comes back twice

    def _reset_replaced(self, hotkeys: Sequence[str]) -> None:
        """A uid whose hotkey changed starts from nothing, as it does locally:
        buckets kept for that uid under any other hotkey are dropped."""
        current = dict(enumerate(hotkeys))
        for per_miner in self.entries.values():
            for uid, hotkey in list(per_miner):
                if current.get(uid) != hotkey:
                    del per_miner[(uid, hotkey)]

    # ---- scoring
    def pooled_observations(self, uid: int, hotkey: str) -> list[float]:
        """Other validators' payments for this registration, inside the window."""
        payments: list[float] = []
        for per_miner in self.entries.values():
            bucket = per_miner.get((uid, hotkey))
            if bucket:
                payments.extend(payment for _, payment, _ in bucket)
        return payments

    def scores(
        self,
        hotkeys: Sequence[str],
        local: Mapping[int, Sequence[tuple[float, float]]],
        *,
        min_samples: int,
        now: float,
        admitted: Callable[[str], bool] | None = None,
    ) -> list[float]:
        """Per-uid score: the mean of local and pooled payments, divided by at
        least min_samples. A miner nobody else has graded scores exactly by
        the local rule. A miner others have graded counts this validator like
        any other: only its newest cap observations inside the window enter."""
        self.settle(hotkeys, now, admitted=admitted)
        result: list[float] = []
        for uid, hotkey in enumerate(hotkeys):
            payments = self._window(uid, hotkey, local, now)
            result.append(sum(payments) / max(len(payments), min_samples) if payments else 0.0)
        return result

    def _window(
        self, uid: int, hotkey: str, local: Mapping[int, Sequence[tuple[float, float]]], now: float
    ) -> list[float]:
        pooled = self.pooled_observations(uid, hotkey)
        if not pooled:
            return [payment for _, payment in local.get(uid, ())]
        cutoff = now - self.window_s
        own = [payment for observed_at, payment in local.get(uid, ()) if observed_at > cutoff]
        return own[-self.cap :] + pooled

    def observation_count(
        self,
        hotkeys: Sequence[str],
        local: Mapping[int, Sequence[tuple[float, float]]],
        *,
        now: float,
        admitted: Callable[[str], bool] | None = None,
    ) -> int:
        """The largest window any miner has: the evidence gate."""
        self.settle(hotkeys, now, admitted=admitted)
        return max((len(self._window(uid, hotkey, local, now)) for uid, hotkey in enumerate(hotkeys)), default=0)

    @property
    def active(self) -> bool:
        return any(self.entries.values())

    # ---- persistence
    def save(self, path: str | os.PathLike[str]) -> None:
        state = {
            "format": STATE_FORMAT,
            "cursor": self.cursor,
            "seen": {validator: sorted(numbers) for validator, numbers in self.seen.items()},
            "entries": {
                validator: [[uid, hotkey, list(bucket)] for (uid, hotkey), bucket in per_miner.items()]
                for validator, per_miner in self.entries.items()
            },
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(state, separators=(",", ":")))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)  # the numbers taken must not roll back with a crash
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def load(self, path: str | os.PathLike[str], *, now: float) -> bool:
        """Restore a saved pool. Anything not exactly as written, including a
        payment outside [0, 1] or a non-finite time, means starting empty."""
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as error:
            print(f"[validator] WARN: ledger pool state unreadable, starting empty: {type(error).__name__}")
            return False
        try:
            if not isinstance(state, dict) or state.get("format") != STATE_FORMAT:
                raise ValueError("format")
            cursor = state["cursor"]
            if cursor is not None and (type(cursor) is not str or not 0 < len(cursor) <= 512):
                raise ValueError("cursor")
            seen: dict[str, set[int]] = {}
            for validator, numbers in state["seen"].items():
                if type(numbers) is not list or not 0 < len(numbers) <= SEQ_SLACK:
                    raise ValueError("seen")
                seen[_hotkey(validator)] = {_seq(n) for n in numbers}
            entries: dict[str, dict[tuple[int, str], deque[Entry]]] = {}
            for validator, buckets in state["entries"].items():
                per_miner: dict[tuple[int, str], deque[Entry]] = {}
                for uid, hotkey, items in buckets:
                    if type(uid) is not int or uid < 0:
                        raise ValueError("uid")
                    bucket: deque[Entry] = deque(maxlen=self.cap)
                    for recorded, payment, seq in items:
                        if type(payment) not in (int, float) or not 0.0 <= float(payment) <= 1.0:
                            raise ValueError("payment")
                        bucket.append((_time(recorded), float(payment), _seq(seq)))
                    per_miner[(uid, _hotkey(hotkey))] = bucket
                entries[_hotkey(validator)] = per_miner
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            print(f"[validator] WARN: ledger pool state malformed, starting empty: {type(error).__name__}")
            return False
        self.cursor, self.seen, self.entries = cursor, seen, entries
        self.prune(now)
        return True


def _hotkey(value: Any) -> str:
    if type(value) is not str or not 0 < len(value) <= 256:
        raise ValueError("hotkey")
    return value


def _seq(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("seq")
    return value


def _time(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("time")
    return float(value)


def validating_with_stake(metagraph: Any, hotkey: str, *, min_share: float) -> bool:
    """Admission: the hotkey holds a validator permit, has non-zero validator
    trust, and holds at least min_share of the subnet's stake."""
    try:
        uid = list(metagraph.hotkeys).index(hotkey)
        permits = metagraph.validator_permit
        trust = metagraph.validator_trust
        stakes = [float(value) for value in metagraph.S]
    except (ValueError, AttributeError, TypeError):
        return False
    total = sum(stakes)
    if uid >= len(permits) or uid >= len(trust) or uid >= len(stakes) or total <= 0:
        return False
    return bool(permits[uid]) and float(trust[uid]) > 0.0 and stakes[uid] / total >= min_share
