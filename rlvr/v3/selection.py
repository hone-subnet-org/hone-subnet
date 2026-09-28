"""Which miners a validator offers a task to.

The validator, not the problem server, chooses. It offers the serving miners
in a random order, and the server must issue slots for exactly the first N of
them, N fixed by release policy. Validators (permit holders) and the owner
UID 0 are never offered. The order is kept until a round completes, so a
rejected lease never earns the server a fresh draw.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Sequence

from .api import CANDIDATE_LIMIT


def eligible_miners(
    serving: Iterable[tuple[int, str]],
    *,
    validator_permits: Sequence[bool] | None = None,
) -> list[tuple[int, str]]:
    """Serving miners minus the owner and validator permit holders."""
    # The metagraph hands this back as an array; never truth-test it as a whole.
    permits = [] if validator_permits is None else [bool(item) for item in validator_permits]
    return [
        (uid, hotkey)
        for uid, hotkey in serving
        if uid != 0 and not (uid < len(permits) and bool(permits[uid]))
    ]


def next_offer(
    previous: Sequence[tuple[int, str]] | None,
    eligible: Iterable[tuple[int, str]],
    *,
    limit: int = CANDIDATE_LIMIT,
    rng: secrets.SystemRandom | None = None,
) -> list[tuple[int, str]]:
    """The order to offer this round.

    With no previous order (the last round completed, or this is the first
    round) it is a fresh cryptographically random shuffle. Otherwise the
    previous order is kept, minus miners no longer eligible, with newly
    eligible miners shuffled in at the end, so a failed lease cannot be
    retried against a different draw.
    """
    rng = rng or secrets.SystemRandom()
    eligible_set = set(eligible)
    kept = [item for item in (previous or ()) if item in eligible_set]
    fresh = [item for item in eligible_set if item not in set(kept)]
    fresh.sort()  # deterministic input to the shuffle
    rng.shuffle(fresh)
    return (kept + fresh)[:limit]


def choose_candidates(
    serving: Iterable[tuple[int, str]],
    *,
    validator_permits: Sequence[bool] | None = None,
    limit: int = CANDIDATE_LIMIT,
    rng: secrets.SystemRandom | None = None,
) -> list[tuple[int, str]]:
    """A fresh random offer from the serving set (no previous order)."""
    return next_offer(None, eligible_miners(serving, validator_permits=validator_permits), limit=limit, rng=rng)
