"""Zero owner burn: plain normalized miner weights, no owner resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlvr.config import Settings, release_policy_summary
from rlvr.neurons.decentralized import _submit_local_weights
from rlvr.policy import RELEASE_POLICY


def test_policy_owner_burn_is_zero():
    assert RELEASE_POLICY.owner_burn_share == 0.0
    assert release_policy_summary().endswith("owner_burn=0")


class _Engine:
    def __init__(self, weights):
        self._weights = list(weights)
        self.histories = {0: [1.0] * 8}

    def get_weights(self, n, blocked=None):
        return list(self._weights)[:n]


def _validator(recorded):
    def set_weights(**kwargs):
        recorded["weights"] = kwargs["weights"]
        return True, ""

    def owner_lookup(*args, **kwargs):
        raise AssertionError("owner lookup is disabled")

    return SimpleNamespace(
        metagraph=SimpleNamespace(
            uids=[0, 1, 2],
            hotkeys=["hk-0", "hk-1", "hk-2"],
            coldkeys=["ck-0", "ck-1", "ck-2"],
        ),
        subtensor=SimpleNamespace(
            set_weights=set_weights,
            get_current_block=owner_lookup,
            get_subnet_owner_hotkey=owner_lookup,
            query_subtensor=owner_lookup,
        ),
        wallet=SimpleNamespace(),
    )


def test_weights_submit_without_owner_resolution():
    recorded: dict = {}

    result = _submit_local_weights(
        _validator(recorded),
        _Engine([0.0, 0.75, 0.25]),
        Settings(_env_file=None),
    )

    assert result is True
    assert recorded["weights"] == pytest.approx([0.0, 0.75, 0.25])


def test_all_zero_scores_skip_submission():
    recorded: dict = {}

    result = _submit_local_weights(
        _validator(recorded),
        _Engine([0.0, 0.0, 0.0]),
        Settings(_env_file=None),
    )

    assert result is None
    assert "weights" not in recorded
