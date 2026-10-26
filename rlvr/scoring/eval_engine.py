"""Count-bounded score histories -> L1-normalized weights.

Design choices:
  - Each uid retains its latest ``max_samples`` completed-problem payments.
  - A score is the mean payment raised to ``score_exponent``, then scaled by
    ``min(observations / min_samples, 1)``. The exponent widens the gap
    between better and worse miners; among miners past the ramp it never
    changes their order. The ramp stops a few lucky first answers from
    earning a full score, and stays linear because the exponent is applied
    before it, not after.
  - Only completed challenges advance histories. Non-serving miners receive one
    zero per completed challenge, never for server pacing or empty rounds.
    Set ``decay=False`` to freeze histories on silence.
  - Deregistration is handled by :meth:`sync`: when a uid's hotkey changes the slot
    belongs to a NEW miner, so its inherited history is reset. Hotkeys are tracked
    per uid as they are observed (via :meth:`update` or :meth:`set_hotkeys`).
  - The score vector can GROW if the metagraph adds UIDs (e.g. after a sync), and
    :meth:`get_weights` can be aligned to a provided metagraph size ``n`` so the
    weight vector length matches what ``set_weights`` expects.
  - Everything is nan-safe: nan/inf scores are treated as 0 before normalization,
    and an all-zero score vector yields all-zero weights (not nan).
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np


def history_score(
    rewards: Iterable[float], min_samples: int, exponent: float
) -> float:
    """Mean payment to the power ``exponent``, ramped linearly up to ``min_samples``."""
    values = list(rewards)
    if not values:
        return 0.0
    quality = max(sum(values) / len(values), 0.0)
    ramp = min(len(values) / min_samples, 1.0)
    # numpy overflows to inf instead of raising; get_weights maps inf to 0.
    with np.errstate(over="ignore"):
        return float(np.power(quality, exponent) * ramp)


class EvalEngine:
    """Per-uid bounded payment history -> normalized weights."""

    def __init__(
        self,
        num_uids: int,
        window_seconds: float,
        max_samples: int,
        min_samples: int,
        decay: bool = True,
        clock: Callable[[], float] = time.time,
        *,
        score_exponent: float = 1.0,
    ):
        if num_uids < 0:
            raise ValueError("num_uids must be non-negative")
        if not np.isfinite(window_seconds) or float(window_seconds) <= 0.0:
            raise ValueError("window_seconds must be finite and positive")
        if (
            isinstance(max_samples, bool)
            or int(max_samples) != max_samples
            or int(max_samples) < 1
        ):
            raise ValueError("max_samples must be a positive integer")
        if (
            isinstance(min_samples, bool)
            or int(min_samples) != min_samples
            or not 1 <= int(min_samples) <= int(max_samples)
        ):
            raise ValueError("min_samples must be between 1 and max_samples")
        if (
            isinstance(score_exponent, bool)
            or not np.isfinite(score_exponent)
            or float(score_exponent) <= 0.0
        ):
            raise ValueError("score_exponent must be finite and positive")
        self.window_seconds = float(window_seconds)
        self.max_samples = int(max_samples)
        self.min_samples = int(min_samples)
        self.decay = bool(decay)
        self.score_exponent = float(score_exponent)
        self._clock = clock
        self.scores: np.ndarray = np.zeros(int(num_uids), dtype=np.float64)
        # Sparse so "never observed" remains distinct from "observed zero".
        self.histories: dict[int, list[tuple[float, float]]] = {}
        # Last hotkey observed at each uid; used to detect deregistration. Sparse:
        # only uids we've actually seen a hotkey for appear here.
        self.hotkeys: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    def _ensure_capacity(self, uid: int) -> None:
        """Grow the score vector (zero-filled) to cover `uid` if needed."""
        if uid < 0:
            raise ValueError(f"uid must be non-negative, got {uid}")
        if uid >= self.scores.size:
            grown = np.zeros(uid + 1, dtype=np.float64)
            grown[: self.scores.size] = self.scores
            self.scores = grown

    def _timestamp(self, observed_at: float | None = None) -> float:
        value = float(self._clock() if observed_at is None else observed_at)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("observation timestamp must be finite and non-negative")
        return value

    def _refresh_score(self, uid: int) -> None:
        history = self.histories.get(uid, [])
        self.scores[uid] = history_score(
            (reward for _, reward in history), self.min_samples, self.score_exponent
        )

    def _record(self, uid: int, reward: float, observed_at: float) -> None:
        """Append one bounded observation and refresh the derived score."""
        uid = int(uid)
        self._ensure_capacity(uid)
        value = float(reward)
        if not np.isfinite(value) or value < 0.0:
            value = 0.0
        history = self.histories.setdefault(uid, [])
        history.append((observed_at, value))
        if len(history) > self.max_samples:
            del history[: len(history) - self.max_samples]
        self._refresh_score(uid)

    def _restore(
        self,
        num_uids: int,
        histories: dict[int, list[tuple[float, float]]],
        hotkeys: dict[int, str],
    ) -> None:
        """Install fully validated persisted state and recompute all scores."""
        self.scores = np.zeros(int(num_uids), dtype=np.float64)
        self.histories = {}
        for uid, values in histories.items():
            kept = [
                (float(timestamp), float(reward))
                for timestamp, reward in values[-self.max_samples :]
            ]
            if not kept:
                continue
            self.histories[int(uid)] = kept
            self._refresh_score(int(uid))
        self.hotkeys = dict(hotkeys)

    def set_hotkeys(self, hotkeys: dict[int, str]) -> None:
        """Record the current hotkey for each uid without scoring side effects."""
        for uid, hotkey in hotkeys.items():
            uid = int(uid)
            self._ensure_capacity(uid)
            self.hotkeys[uid] = str(hotkey)

    def sync(self, hotkeys: dict[int, str], n: int | None = None) -> list[int]:
        """Reconcile against the current metagraph hotkeys.

        For every uid whose hotkey differs from the one we last tracked, the slot
        has been reused by a NEW miner: reset its score so it starts from 0. The
        new hotkey is then recorded. Optionally resize to metagraph size ``n``.

        Returns the list of uids that were reset (for logging/tests).
        """
        reset: list[int] = []
        for uid, hotkey in hotkeys.items():
            uid = int(uid)
            hotkey = str(hotkey)
            self._ensure_capacity(uid)
            prev = self.hotkeys.get(uid)
            if prev is not None and prev != hotkey:
                self.scores[uid] = 0.0
                self.histories.pop(uid, None)
                reset.append(uid)
            self.hotkeys[uid] = hotkey
        if n is not None:
            self.resize(n)
        return reset

    def resize(self, n: int) -> None:
        """Align the score vector to metagraph size ``n``.

        Grows (zero-filled) or shrinks (truncating dropped tail uids). Shrinking
        also forgets tracked hotkeys for the removed uids.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        cur = self.scores.size
        if n == cur:
            return
        if n > cur:
            grown = np.zeros(n, dtype=np.float64)
            grown[:cur] = self.scores
            self.scores = grown
        else:
            self.scores = self.scores[:n].copy()
            self.histories = {
                uid: history for uid, history in self.histories.items() if uid < n
            }
            self.hotkeys = {uid: hk for uid, hk in self.hotkeys.items() if uid < n}

    def update(
        self,
        rewards: dict[int, float],
        hotkeys: dict[int, str] | None = None,
        dispatched: "set[int] | None" = None,
        observed_at: float | None = None,
    ) -> float:
        """Record one completed-challenge observation for the relevant uids.

        Responders append their payment. Non-finite or negative rewards are
        clamped to 0.0 so a bad round cannot poison persisted state.

        Silence semantics depend on ``dispatched``:
          * ``dispatched=None`` (full-pool dispatch, the sim default): when
            ``decay`` is enabled, every previously observed uid absent from
            ``rewards`` appends zero.
          * ``dispatched`` provided (subset dispatch): absence from ``rewards``
            usually just means "not sampled for this problem", which must NOT
            be punished. Only uids that were dispatched-but-silent append zero;
            everyone else is frozen until their turn in the rotation.

        ``hotkeys`` (optional, {uid: hotkey}) records the hotkey seen this round so
        a later :meth:`sync` can detect deregistration.
        """
        if hotkeys:
            self.set_hotkeys(hotkeys)
        timestamp = self._timestamp(observed_at)
        if not rewards and dispatched is None:
            return timestamp
        responders: set[int] = set()
        for uid, reward in rewards.items():
            uid = int(uid)
            responders.add(uid)
            self._record(uid, reward, timestamp)
        if not self.decay:
            return timestamp
        if dispatched is not None:
            # Subset dispatch: only sampled-but-silent uids get a zero.
            for uid in {int(u) for u in dispatched} - responders:
                self._record(uid, 0.0, timestamp)
            return timestamp
        # Full-pool semantics: every observed non-responder gets one zero.
        for uid in set(self.histories) - responders:
            self._record(uid, 0.0, timestamp)
        return timestamp

    def record_nonserving(
        self,
        active_uids: "Iterable[int]",
        observed_at: float | None = None,
    ) -> None:
        """Append one zero for every observed uid not currently serving.

        Call exactly once after each successfully completed challenge. The
        active set and this absent set must be disjoint, so every uid advances
        by at most one history slot per challenge. Never-observed registrations
        stay at score zero without accumulating pre-start failures.
        """
        if not self.decay or not self.histories:
            return
        timestamp = self._timestamp(observed_at)
        active = {
            int(uid)
            for uid in active_uids
            if 0 <= int(uid) < self.scores.size
        }
        for uid in set(self.histories) - active:
            self._record(uid, 0.0, timestamp)

    def get_weights(self, n: int | None = None) -> "np.ndarray":
        """L1-normalized weights from the current scores; nan-safe.

        nan/inf scores -> 0; negative scores -> 0 (weights are a distribution).
        An all-zero (or empty) score vector yields an all-zero weight vector.
        Reading weights does not mutate history. Only completed challenges add
        observations, and only the count cap removes the oldest observation.

        ``n`` (optional) aligns the returned vector to a metagraph size so its
        length matches what ``set_weights`` expects: a larger ``n`` is zero-padded,
        a smaller ``n`` truncates trailing uids. Normalization is computed over the
        aligned (clamped) scores so the kept weights still sum to 1.
        """
        scores = np.nan_to_num(
            self.scores, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64)
        if n is not None:
            n = int(n)
            if n < 0:
                raise ValueError(f"n must be non-negative, got {n}")
            if n != scores.size:
                aligned = np.zeros(n, dtype=np.float64)
                keep = min(n, scores.size)
                aligned[:keep] = scores[:keep]
                scores = aligned
        # weights represent a non-negative allocation; clamp negatives.
        scores = np.clip(scores, 0.0, None)
        total = scores.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.zeros_like(scores)
        return scores / total
