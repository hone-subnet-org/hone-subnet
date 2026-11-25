"""Weight-submission failure reporting fallback chain.

Three tiers, in strict precedence order:

  1. the SDK's own message, when it has one — already printed, so nothing added;
  2. the structured `error`/`data`, when the message is empty;
  3. the block context, when message, error and data are ALL empty — the only
     evidence left for the one question an operator can act on: "too soon?".

These tests also pin the final failure log line and the strict on-chain
rate-limit boundary.

THE BOUNDARY. The chain admits a weight set when
    current_block - last_update >= weights_rate_limit
so elapsed == limit is ACCEPTED and elapsed == limit - 1 is REJECTED. A
diagnostic that is off by one here sends the operator to raise a cadence that
was never the cause, so it is pinned as a pure predicate rather than sniffed
out of a rendered string.

SOURCE OF "LAST UPDATE". The chain's own `metagraph.last_update[uid]` — the
same number `check_rate_limit` reads. Deliberately NOT a new in-process
`last_weight_block`: that would need fresh bookkeeping, would be unknown after
every restart, and would be a second opinion about a fact the chain already
publishes. Its one weakness is staleness — a submission made after the last
metagraph sync makes elapsed look larger, i.e. it under-claims rate-limiting
rather than over-claiming it, which is the safe direction for a diagnostic.

"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlvr.config import Settings
from rlvr.neurons.decentralized import (
    _submit_local_weights,
    _weight_failure_report,
    _weights_rate_limited,
)
from rlvr.scoring.eval_engine import EvalEngine

# Every fact carries a distinct value so no assertion can be satisfied by the
# wrong number: block 5000, last update 4913, elapsed 87, limit 100, cadence 180.
_BLOCK = 5_000
_LAST_UPDATE = 4_913
_ELAPSED = 87
_CHAIN_LIMIT = 100
_INTERVAL = 180


def result(**attrs):
    return SimpleNamespace(**attrs)


def validator(
    *,
    block=_BLOCK,
    last_update=_LAST_UPDATE,
    uid=0,
    chain_limit=_CHAIN_LIMIT,
    interval=_INTERVAL,
):
    def get_current_block():
        if isinstance(block, Exception):
            raise block
        return block

    metagraph = SimpleNamespace(hotkeys=["miner", "owner"], coldkeys=["ck-miner", "ck-owner"], uids=[77, 251])
    if last_update is not None:
        metagraph.last_update = [last_update, 0]

    return SimpleNamespace(
        subtensor=SimpleNamespace(get_current_block=get_current_block),
        metagraph=metagraph,
        uid=uid,
        chain_weights_rate_limit=chain_limit,
        weights_interval_blocks=interval,
    )


# --------------------------------------------------------------------------- #
# The strict on-chain rate-limit boundary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("elapsed", "limit", "limited"),
    [
        (99, 100, True),    # one short — the chain rejects this
        (100, 100, False),  # exactly the limit — the chain ACCEPTS this
        (101, 100, False),
        (0, 1, True),
        (1, 1, False),
        (0, 0, False),      # a zero limit rate-limits nothing
    ],
)
def test_the_chain_boundary_is_inclusive_at_the_limit(elapsed, limit, limited):
    """`current_block - last_update >= limit` is the chain's own admission rule.

    Reporting `rate_limited` at exactly the limit accuses the cadence of a
    rejection the chain did not make, and sends the operator to widen an
    interval that was already wide enough.
    """
    assert _weights_rate_limited(elapsed, limit) is limited


@pytest.mark.parametrize(
    ("elapsed", "limit"),
    [(None, 100), (87, None), (None, None), (87, 0), (-5, 100)],
)
def test_an_unknown_boundary_makes_no_accusation(elapsed, limit):
    """Missing evidence is not evidence of rate-limiting."""
    assert _weights_rate_limited(elapsed, limit) is False


# --------------------------------------------------------------------------- #
# Tier precedence
# --------------------------------------------------------------------------- #
def test_a_successful_submission_reports_nothing_extra():
    """Diagnostics are for failures; a success must not grow a paragraph."""
    assert _weight_failure_report(result(success=True, message=""), validator()) == ""


def test_an_sdk_message_is_left_to_speak_for_itself():
    """The message is already printed. Repeating it is noise, not diagnosis."""
    assert (
        _weight_failure_report(
            result(success=False, message="rejected by chain", error="Invalid"),
            validator(),
        )
        == ""
    )


def test_an_empty_message_falls_back_to_structured_detail():
    report = _weight_failure_report(
        result(success=False, message="", error="InvalidTransaction", data=""),
        validator(),
    )

    assert "InvalidTransaction" in report


def test_structured_detail_outranks_block_context():
    """Block context is the LAST resort; a stated cause beats an inference."""
    report = _weight_failure_report(
        result(success=False, message="", error="InvalidTransaction"), validator()
    )

    assert "InvalidTransaction" in report
    assert str(_LAST_UPDATE) not in report
    assert str(_BLOCK) not in report


def test_an_empty_container_is_absence_and_reaches_the_block_context():
    """The 4a boundary, carried through: `data={}` must not preempt tier three."""
    report = _weight_failure_report(
        result(success=False, message="", error=None, data={}), validator()
    )

    assert str(_LAST_UPDATE) in report


# --------------------------------------------------------------------------- #
# Tier three: the block context
# --------------------------------------------------------------------------- #
def test_a_silent_failure_reports_the_whole_block_context():
    """All five facts are required together.

    A current block without a last update cannot answer "too soon?"; elapsed
    blocks without the chain limit cannot say whether elapsed was enough; and
    the effective interval is the number the operator would actually change.
    """
    report = _weight_failure_report(result(success=False, message=""), validator())

    assert str(_BLOCK) in report
    assert str(_LAST_UPDATE) in report
    assert str(_ELAPSED) in report
    assert str(_CHAIN_LIMIT) in report
    assert str(_INTERVAL) in report


def test_the_block_context_names_a_too_soon_submission():
    """87 elapsed against a 100-block limit is the case the tier exists for."""
    report = _weight_failure_report(result(success=False, message=""), validator())

    assert "rate_limited=True" in report


def test_a_cleared_limit_is_reported_as_cleared():
    """Elapsed exactly at the limit cleared it, so the cadence is not the cause."""
    report = _weight_failure_report(
        result(success=False, message=""),
        validator(last_update=_BLOCK - _CHAIN_LIMIT),
    )

    assert "rate_limited=False" in report


def test_a_legacy_result_shape_still_reports_context():
    """Legacy tuple/bool failures carry no fields at all, so tier three is all."""
    assert str(_LAST_UPDATE) in _weight_failure_report((False, ""), validator())
    assert str(_LAST_UPDATE) in _weight_failure_report(False, validator())


def test_a_legacy_success_shape_reports_nothing():
    assert _weight_failure_report((True, ""), validator()) == ""
    assert _weight_failure_report(True, validator()) == ""


def test_the_report_stays_on_one_log_line():
    """It is appended to a single print; a multi-line SDK error must not split it."""
    report = _weight_failure_report(
        result(success=False, message="", error="line one\nline two\r\nthree"),
        validator(),
    )

    assert "\n" not in report
    assert "\r" not in report


# --------------------------------------------------------------------------- #
# The diagnostic path must never become the reason weights fail
# --------------------------------------------------------------------------- #
def test_an_unreadable_current_block_still_reports_what_is_known():
    """current_block is an RPC, and it fails exactly when things are broken."""
    report = _weight_failure_report(
        result(success=False, message=""),
        validator(block=RuntimeError("substrate connection refused")),
    )

    assert report != ""
    assert str(_INTERVAL) in report
    assert str(_LAST_UPDATE) in report
    assert "rate_limited" not in report  # elapsed is unknowable without a block


def test_a_first_submission_has_no_previous_block_and_does_not_invent_one():
    """Unknown elapsed reported as 0 reads as "submitted this block" — a wrong
    diagnosis pointing at the one cause that is definitely not it."""
    report = _weight_failure_report(
        result(success=False, message=""), validator(last_update=None)
    )

    assert report != ""
    assert str(_BLOCK) in report
    assert "elapsed=0" not in report
    assert "rate_limited" not in report


def test_an_unregistered_uid_does_not_index_the_metagraph():
    """`uid` is None until setup_bittensor() resolves it, and out-of-range after
    a deregistration; neither may raise on the submission path."""
    for broken in (None, 999, -1, "seven"):
        report = _weight_failure_report(
            result(success=False, message=""), validator(uid=broken)
        )

        assert str(_INTERVAL) in report
        assert "rate_limited" not in report


def test_a_hostile_metagraph_does_not_take_down_the_report():
    class Exploding:
        def __getitem__(self, index):
            raise RuntimeError("metagraph is mid-sync")

    v = validator()
    v.metagraph.last_update = Exploding()

    report = _weight_failure_report(result(success=False, message=""), v)

    assert str(_INTERVAL) in report


def test_an_unknown_chain_limit_still_reports_the_rest():
    report = _weight_failure_report(
        result(success=False, message=""), validator(chain_limit=None)
    )

    assert str(_BLOCK) in report
    assert str(_ELAPSED) in report
    assert str(_INTERVAL) in report
    assert "rate_limited" not in report


def test_a_validator_missing_everything_returns_a_string_not_an_exception():
    """Nothing about diagnostics is worth losing a round's emissions over."""
    assert isinstance(
        _weight_failure_report(result(success=False, message=""), object()), str
    )


# --------------------------------------------------------------------------- #
# The wiring: the failure log line itself, pinned once
# --------------------------------------------------------------------------- #
class _FakeSubtensor:
    def __init__(self, weight_result):
        self.weight_result = weight_result
        self.calls: list[dict] = []

    def set_weights(self, **kwargs):
        self.calls.append(kwargs)
        return self.weight_result

    def get_current_block(self):
        return _BLOCK

    def get_subnet_owner_hotkey(self, netuid, block=None):
        return "owner"

    def query_subtensor(self, name, params, block=None):
        return SimpleNamespace(value="Burn")


def _submit(weight_result):
    engine = EvalEngine(
        num_uids=2, window_seconds=100, max_samples=6, min_samples=4
    )
    for _ in range(4):
        engine.update({0: 1.0, 1: 0.0}, dispatched={0, 1})
    settings = Settings(
        _env_file=None, netuid=5, validator_min_weight_observations=4
    )
    subtensor = _FakeSubtensor(weight_result)
    v = validator()
    v.subtensor = subtensor
    v.wallet = object()
    return _submit_local_weights(v, engine, settings)


def test_the_silent_failure_log_line_carries_the_block_context(capsys):
    """The operator dead end this behavior closes: `ok=False msg=''` and nothing."""
    assert _submit(SimpleNamespace(success=False, message="")) is False
    line = capsys.readouterr().out

    assert "ok=False" in line
    assert str(_LAST_UPDATE) in line
    assert str(_ELAPSED) in line
    assert "rate_limited=True" in line


def test_the_silent_failure_log_line_stays_one_line(capsys):
    assert _submit(SimpleNamespace(success=False, message="")) is False
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]

    assert len([ln for ln in lines if "set local weights" in ln]) == 1


def test_a_failure_with_a_message_is_not_padded_with_block_context(capsys):
    assert _submit(SimpleNamespace(success=False, message="rejected")) is False
    line = capsys.readouterr().out

    assert "'rejected'" in line
    assert str(_LAST_UPDATE) not in line
    assert "rate_limited" not in line


def test_a_successful_submission_logs_no_diagnostics(capsys):
    assert _submit(SimpleNamespace(success=True, message="")) is True
    line = capsys.readouterr().out

    assert "ok=True" in line
    assert str(_LAST_UPDATE) not in line
    assert "rate_limited" not in line
