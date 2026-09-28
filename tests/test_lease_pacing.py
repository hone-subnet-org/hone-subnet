"""Monotonic conversion of server pacing directives.

These tests isolate deadline conversion from event-loop wiring and cadence
defaults.

"""

from __future__ import annotations

import time

import pytest

from rlvr.problemserver.client import (
    _RETRY_AFTER_CUSHION_S,
    next_lease_not_before,
)


def test_deadline_is_now_plus_the_directive_plus_a_cushion():
    deadline = next_lease_not_before(45, now=1_000.0)

    assert deadline == 1_000.0 + 45 + _RETRY_AFTER_CUSHION_S


def test_the_cushion_is_strictly_positive():
    """Retry-After is whole seconds, so the server's own boundary is coarse.

    Retrying at exactly now+N can land microseconds before the server considers
    the window elapsed, which re-trips the 429 and wastes a round for nothing.
    """
    assert _RETRY_AFTER_CUSHION_S > 0


@pytest.mark.parametrize("seconds", [0, 1, 45, 3600, 1209600, 999999999])
def test_every_valid_directive_yields_a_deadline_strictly_after_now(seconds):
    """Including zero: a server that says 0 still means 'not before now'."""
    now = 5_000.0

    assert next_lease_not_before(seconds, now=now) > now


def test_large_directives_are_carried_through_without_truncation():
    """The parser refuses to cap; the converter must not quietly reintroduce one."""
    assert next_lease_not_before(1_209_600, now=0.0) == (
        1_209_600 + _RETRY_AFTER_CUSHION_S
    )


def test_deadline_is_monotonic_not_wall_clock():
    """A wall-clock deadline breaks when NTP steps the clock mid-deferral.

    time.time() is ~1.7e9 while time.monotonic() is orders of magnitude smaller,
    so the two are trivially distinguishable — which is the point: this catches
    the mistake rather than trusting a comment that says "monotonic".
    """
    before = time.monotonic()
    deadline = next_lease_not_before(0)
    after = time.monotonic()

    assert before <= deadline - _RETRY_AFTER_CUSHION_S <= after
    assert abs(deadline - time.time()) > 1_000_000, (
        "deadline tracks the wall clock; an NTP step would move it"
    )


def test_now_defaults_to_the_current_monotonic_reading():
    first = next_lease_not_before(10)
    second = next_lease_not_before(10)

    # Monotonic never goes backwards, so a later call is never an earlier deadline.
    assert second >= first


def test_validator_bounds_a_server_directed_pause():
    from rlvr.neurons.decentralized import MAX_LEASE_DEFERRAL_S, lease_deferral_s

    assert lease_deferral_s(None) is None
    assert lease_deferral_s(45) == 45
    assert lease_deferral_s(1_209_600) == MAX_LEASE_DEFERRAL_S  # two weeks is asked again in minutes
    assert lease_deferral_s(-5) == 0
