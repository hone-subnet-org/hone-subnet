"""Which miners a validator offers a task to.

The validator, not the problem server, chooses. It offers the serving miners
in a random order, and the server must issue slots for exactly the first N of
them, N fixed by release policy. UIDs that are validating (a permit with
non-zero validator trust) and the subnet owner's hotkey, when the chain
names one, are never offered.
The order is kept until a round completes, so a rejected lease never earns
the server a fresh draw.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Sequence

from .api import CANDIDATE_LIMIT


def eligible_miners(
    serving: Iterable[tuple[int, str]],
    *,
    validator_permits: Sequence[bool] | None = None,
    validator_trust: Sequence[float] | None = None,
    owner_hotkey: str | None = None,
) -> list[tuple[int, str]]:
    """Serving miners minus the subnet owner and the UIDs that are validating.

    The owner is the hotkey the chain names as the subnet owner, whatever UID
    it holds; UID 0 is an ordinary miner.

    A UID is validating when it holds a validator permit and has non-zero
    validator trust, that is, it sets weights. A permit alone is not enough:
    permits go to the top stakes, so a well-staked miner holds one too. When
    the trust list is unavailable, or has no entry for a UID, the permit alone
    decides for that UID, conservatively. Trust measures weights that survive
    consensus, so a validator whose weights are all clipped has none; such a
    UID is offered a task only if it also serves a miner axon, in which case
    it is a miner as well.
    """
    # The metagraph hands these back as arrays; never truth-test them as a whole.
    permits = [] if validator_permits is None else [bool(item) for item in validator_permits]
    trust = None if validator_trust is None else [float(item) for item in validator_trust]

    def validating(uid: int) -> bool:
        if not (uid < len(permits) and permits[uid]):
            return False
        if trust is None or uid >= len(trust):
            return True  # permit with unknown trust: treat as validating
        return trust[uid] > 0.0

    return [
        (uid, hotkey)
        for uid, hotkey in serving
        if hotkey != owner_hotkey and not validating(uid)
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
    owner_hotkey: str | None = None,
) -> list[tuple[int, str]]:
    """A fresh random offer from the serving set (no previous order)."""
    eligible = eligible_miners(serving, validator_permits=validator_permits, owner_hotkey=owner_hotkey)
    return next_offer(None, eligible, limit=limit, rng=rng)
