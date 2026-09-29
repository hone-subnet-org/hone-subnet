"""Tests for the shared wire protocol and validator neuron imports.

These tests NEVER touch the network or the chain stack:
  * signing uses the HMAC fallback — FORCED via the autouse fixture below, so
    the tests are hermetic whether or not bittensor is installed in this env;
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

import rlvr.protocol as protocol_module
from rlvr.config import Settings, get_settings
from rlvr.policy import RELEASE_POLICY
from rlvr.problemserver.api import PublicChallenge
from rlvr.protocol import (
    NonceCache,
    SignedSolution,
    SolutionPayload,
    TaskRequest,
    crypto_available,
    sign_message,
    verify_signature,
)
from rlvr.types import TestCase as Case


@pytest.fixture(autouse=True)
def force_no_crypto(monkeypatch):
    """Force the HMAC fallback regardless of the installed chain stack.

    A dev/live venv legitimately has bittensor (and its Keypair) installed, in
    which case verify_signature correctly refuses HMAC entirely — but these
    tests exercise the chain-free sim path, so pin the no-crypto condition."""
    monkeypatch.setattr(protocol_module, "_HAVE_CRYPTO", False)
    monkeypatch.setattr(protocol_module, "_Keypair", None)


# --------------------------------------------------------------------------- #
# Sign / verify roundtrip (HMAC fallback)
# --------------------------------------------------------------------------- #
def test_sign_verify_roundtrip_fallback():
    body = b'{"problem_id": "p1", "code": "def f(): return 1"}'
    headers = sign_message("hotkey-alice", body, signed_for="hotkey-bob")
    assert verify_signature(headers, body) is True
    # Header layout matches epistula v2.
    assert headers["Epistula-Version"] == "2"
    assert headers["Epistula-Signed-By"] == "hotkey-alice"
    assert headers["Epistula-Signed-For"] == "hotkey-bob"
    assert headers["Epistula-Request-Signature"].startswith("0x")


def test_sign_verify_no_signed_for():
    body = b"payload"
    headers = sign_message("alice", body)
    assert headers["Epistula-Signed-For"] == ""
    assert verify_signature(headers, body) is True


def test_verify_rejects_tampered_body():
    body = b"original"
    headers = sign_message("alice", body)
    assert verify_signature(headers, b"tampered") is False


def test_verify_rejects_wrong_signer():
    """A signature minted by one hotkey must not verify when the Signed-By is swapped."""
    body = b"payload"
    headers = sign_message("alice", body)
    headers["Epistula-Signed-By"] = "mallory"
    assert verify_signature(headers, body) is False


def test_verify_rejects_tampered_signature():
    body = b"payload"
    headers = sign_message("alice", body)
    headers["Epistula-Request-Signature"] = "0xdeadbeef"
    assert verify_signature(headers, body) is False


def test_verify_rejects_stale_timestamp():
    body = b"payload"
    headers = sign_message("alice", body)
    ts = int(headers["Epistula-Timestamp"])
    # now is way ahead of the signed timestamp -> stale.
    assert verify_signature(headers, body, now_ms=ts + 10_000_000) is False


def test_verify_rejects_future_timestamp():
    body = b"payload"
    headers = sign_message("alice", body)
    ts = int(headers["Epistula-Timestamp"])
    # signed timestamp is far in the future relative to now -> rejected.
    assert verify_signature(headers, body, now_ms=ts - 10_000_000) is False


def test_verify_missing_headers_returns_false():
    assert verify_signature({}, b"x") is False


def test_verify_rejects_wrong_or_missing_protocol_version():
    body = b"payload"
    headers = sign_message("alice", body)
    headers["Epistula-Version"] = "1"
    assert verify_signature(headers, body) is False
    headers.pop("Epistula-Version")
    assert verify_signature(headers, body) is False


def test_sign_requires_bytes():
    with pytest.raises(TypeError):
        sign_message("alice", "not-bytes")  # type: ignore[arg-type]


def test_crypto_available_is_false_without_chain_stack():
    # force_no_crypto pins the fallback; crypto_available() must reflect it.
    assert crypto_available() is False


def test_sign_unwraps_wallet_keypair_with_live_crypto(monkeypatch):
    """A bittensor wallet is not a signer — sign_message must unwrap wallet.hotkey
    and use ITS .sign, not silently fall back to HMAC (which live peers reject)."""
    monkeypatch.setattr(protocol_module, "_HAVE_CRYPTO", True)

    class FakeKeypair:
        ss58_address = "5" + "F" * 47  # ss58-looking

        def sign(self, message):
            return b"\x01\x02"

    class FakeWallet:
        hotkey = FakeKeypair()

    headers = sign_message(FakeWallet(), b"payload")
    assert headers["Epistula-Signed-By"] == FakeKeypair.ss58_address
    assert headers["Epistula-Request-Signature"] == "0x0102"


def test_hmac_rejected_when_crypto_available(monkeypatch):
    """With a real crypto stack present, the forgeable HMAC path must never be
    accepted, even for opaque (non-ss58) signers."""
    body = b"payload"
    headers = sign_message("opaque-signer", body)  # HMAC (fixture: no crypto)
    monkeypatch.setattr(protocol_module, "_HAVE_CRYPTO", True)
    assert verify_signature(headers, body) is False


def test_sign_accepts_wallet_like_object():
    # A wallet-like object whose address is a short, opaque (non-ss58) id so the
    # HMAC round-trip is still honored in the no-crypto sim. (An ss58-LOOKING id
    # with no real .sign would correctly NOT verify via HMAC — see
    # test_no_hmac_downgrade_for_ss58_signer_without_crypto.)
    class FakeKeypair:
        ss58_address = "sim-hotkey-id"

    class FakeWallet:
        hotkey = FakeKeypair()

    body = b"payload"
    headers = sign_message(FakeWallet(), body)
    assert headers["Epistula-Signed-By"] == FakeKeypair.ss58_address
    assert verify_signature(headers, body) is True


# --------------------------------------------------------------------------- #
# Wire model shape
# --------------------------------------------------------------------------- #
def test_task_request_defaults_and_examples():
    req = TaskRequest(
        problem_id="p1",
        language="python",
        statement="Add two numbers.",
        entrypoint="add",
        public_examples=[Case(args=[1, 2], expected=3)],
    )
    assert req.deadline_s == 60.0
    # round-trips through JSON cleanly (what crosses the wire).
    dumped = req.model_dump_json()
    again = TaskRequest.model_validate_json(dumped)
    assert again == req


def test_task_request_rejects_hidden_or_unknown_wire_fields():
    with pytest.raises(ValueError):
        TaskRequest.model_validate(
            {
                "problem_id": "p1",
                "statement": "Return x.",
                "entrypoint": "f",
                "tests": [{"args": [1], "expected": 1}],
            }
        )


def test_solution_payload_roundtrip():
    p = SolutionPayload(problem_id="p1", code="def add(a,b): return a+b", raw_response="...")
    again = SolutionPayload.model_validate_json(p.model_dump_json())
    assert again == p


def test_solution_payload_rejects_extra_miner_fields_and_keeps_local_latency():
    # Extra miner-controlled JSON fields fail closed. The locally measured
    # latency remains attached to the resulting failed solution artifact.
    artifact = SignedSolution(
        response_body=(
            '{"problem_id":"request-id","code":"def f(): return 1",'
            '"raw_response":"ok","latency_ms":0.01}'
        ),
        latency_ms=487.5,
    )
    solution = artifact.to_solution("canonical-id")
    assert solution.code == ""
    assert "invalid signed miner response" in solution.raw_response
    assert solution.latency_ms == pytest.approx(487.5)


def test_public_challenge_rejects_accidentally_exposed_hidden_tests():
    with pytest.raises(ValueError):
        PublicChallenge.model_validate(
            {
                "challenge_id": "lease",
                "problem_id": "private-id",
                "statement": "Implement f(x).",
                "entrypoint": "f",
                "tests": [{"args": [1], "expected": 1}],
            }
        )


# --------------------------------------------------------------------------- #
# Modules import cleanly WITHOUT bittensor
# --------------------------------------------------------------------------- #
def test_bittensor_present_lazy_helper_returns_module():
    # bittensor is the installed live-testnet dependency; the lazy helper returns
    # it. (The neuron modules still IMPORT fine without it — see the param test
    # below — because the import is deferred into setup_bittensor().)
    from rlvr.neurons.base import _import_bittensor

    bt = _import_bittensor()
    assert bt is not None and hasattr(bt, "Wallet")


@pytest.mark.parametrize(
    "module",
    [
        "rlvr.protocol",
        "rlvr.neurons",
        "rlvr.neurons.base",
        "rlvr.neurons.validator",
    ],
)
def test_modules_import_without_bittensor(module):
    mod = importlib.import_module(module)
    assert mod is not None


def test_base_neuron_constructs_without_chain():
    from rlvr.neurons.base import BaseNeuron

    n = BaseNeuron(get_settings())
    assert n.wallet is None and n.subtensor is None and n.metagraph is None
    assert n.hotkey_address == ""


def test_settings_fail_closed_to_docker_and_validate_difficulty_band():
    settings = Settings(_env_file=None)
    assert settings.executor == "docker"
    assert settings.docker_memory == "256m"
    assert settings.validator_dispatch_concurrency == 256
    assert settings.solve_deadline_s == 3600.0  # the lease expiry, not this default, bounds a task
    assert settings.problem_server_url == "https://d344p1xue0u9qs.cloudfront.net"  # production V3 server
    assert settings.validator_send_concurrency == 32
    assert settings.validator_verify_concurrency == 16
    assert settings.problem_server_request_timeout_s == 60
    assert RELEASE_POLICY.score_window_max_samples == 200
    assert RELEASE_POLICY.score_window_min_samples == 4
    assert RELEASE_POLICY.min_weight_observations == 4
    with pytest.raises(ValueError, match="BAND_LOW"):
        Settings(_env_file=None, band_low=0.8, band_high=0.2)
    with pytest.raises(ValueError):
        Settings(_env_file=None, docker_image="--privileged")


def test_base_neuron_setup_raises_clearly_when_bittensor_missing(monkeypatch):
    # Simulate bittensor being absent (sys.modules[...] = None makes `import
    # bittensor` raise ImportError); the lazy helper must surface a clear error.
    import sys

    from rlvr.neurons.base import _import_bittensor

    monkeypatch.setitem(sys.modules, "bittensor", None)
    with pytest.raises(RuntimeError, match="bittensor is not installed"):
        _import_bittensor()


def test_should_run_block_interval():
    from rlvr.neurons.base import BaseNeuron

    n = BaseNeuron(get_settings())
    assert n.should_run(0, 10) is True  # first check always fires
    assert n.should_run(0, 10) is False  # same block, already serviced
    assert n.should_run(5, 10) is False  # too soon
    assert n.should_run(10, 10) is True
    assert n.should_run(10, 10) is False
    # A long round can skip past an exact boundary; elapsed-interval semantics
    # must still fire on the next observed block.
    assert n.should_run(27, 10) is True
    assert n.should_run(31, 10) is False
    # interval<=0 always fires.
    assert n.should_run(7, 0) is True


# --------------------------------------------------------------------------- #
# Validator seams
# --------------------------------------------------------------------------- #
def test_validator_run_round_requires_callback():
    from rlvr.neurons.validator import ValidatorNeuron

    v = ValidatorNeuron(get_settings())
    import asyncio

    with pytest.raises(RuntimeError, match="No round_callback"):
        asyncio.run(v.run_round())


def test_validator_injected_round_callback():
    from rlvr.neurons.validator import ValidatorNeuron

    async def fake_round(validator):
        assert isinstance(validator, ValidatorNeuron)
        return {0: 1.0, 1: 0.0}

    v = ValidatorNeuron(get_settings())
    v.set_round_callback(fake_round)
    import asyncio

    rewards = asyncio.run(v.run_round())
    assert rewards == {0: 1.0, 1: 0.0}


def test_validator_set_weights_uses_injected_setter():
    from rlvr.neurons.validator import ValidatorNeuron

    captured = {}

    def setter(validator):
        captured["validator"] = validator

    v = ValidatorNeuron(get_settings(), weight_setter=setter)
    v.submit_weights()
    assert captured["validator"] is v


def test_validator_submit_weights_requires_callback():
    from rlvr.neurons.validator import ValidatorNeuron

    v = ValidatorNeuron(get_settings())
    with pytest.raises(RuntimeError, match="No weight_setter"):
        v.submit_weights()


def test_validator_run_offloads_weight_submission(monkeypatch):
    from rlvr.neurons.validator import ValidatorNeuron

    calls = []

    async def fake_to_thread(function, *args, **kwargs):
        calls.append(function)
        return function(*args, **kwargs)

    async def round_callback(validator):
        validator.stop()
        return {0: 1.0}

    validator = ValidatorNeuron(
        get_settings(),
        round_callback=round_callback,
        weight_setter=lambda _validator: calls.append("weights"),
    )
    validator.subtensor = object()
    validator.current_block = lambda: 0
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    asyncio.run(validator.run())

    assert calls == [validator.submit_weights, "weights"]


# --------------------------------------------------------------------------- #
# Recipient binding (Epistula-Signed-For)
# --------------------------------------------------------------------------- #
def test_verify_recipient_binding_accepts_matching_signed_for():
    body = b"payload"
    headers = sign_message("alice", body, signed_for="bob")
    assert verify_signature(headers, body, expected_signed_for="bob") is True


def test_verify_recipient_binding_rejects_mismatch():
    """A request signed FOR hotkey A must be rejected when peer B verifies it."""
    body = b"payload"
    headers = sign_message("alice", body, signed_for="hotkey-A")
    # Peer B insists the request was bound to it.
    assert verify_signature(headers, body, expected_signed_for="hotkey-B") is False


def test_verify_recipient_binding_rejects_empty_signed_for():
    body = b"payload"
    headers = sign_message("alice", body)  # signed_for == ""
    assert verify_signature(headers, body, expected_signed_for="hotkey-B") is False


# --------------------------------------------------------------------------- #
# No-HMAC-downgrade (fail closed for ss58-looking signers)
# --------------------------------------------------------------------------- #
def test_no_hmac_downgrade_for_ss58_signer_without_crypto():
    """An attacker that HMAC-signs but claims an ss58 Signed-By must NOT verify
    via the HMAC path: ss58-looking signers require real crypto (absent here),
    so verification fails closed."""
    assert crypto_available() is False
    ss58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    body = b"payload"
    # Forge an HMAC signature keyed by the ss58 id (what an attacker controls).
    headers = sign_message(ss58, body)
    # The fallback signer produced a valid HMAC for this id...
    assert headers["Epistula-Signed-By"] == ss58
    # ...but because the id LOOKS like ss58 and no crypto stack is present, the
    # verifier refuses the HMAC fallback (fail closed).
    assert verify_signature(headers, body) is False


def test_hmac_still_works_for_opaque_sim_signer():
    """Non-ss58 (opaque sim) ids keep round-tripping via HMAC when crypto absent."""
    assert crypto_available() is False
    body = b"payload"
    headers = sign_message("validator-hotkey", body)  # short, non-ss58 id
    assert verify_signature(headers, body) is True


# --------------------------------------------------------------------------- #
# Replay / nonce cache
# --------------------------------------------------------------------------- #
def test_nonce_cache_rejects_replayed_uuid():
    cache = NonceCache(window_ms=8000)
    assert cache.check_and_add("uuid-1", now_ms=1000) is True
    assert cache.check_and_add("uuid-1", now_ms=1500) is False  # replay
    assert cache.check_and_add("uuid-2", now_ms=1500) is True  # fresh


def test_nonce_cache_evicts_outside_window():
    cache = NonceCache(window_ms=1000)
    assert cache.check_and_add("uuid-1", now_ms=1000) is True
    # Far outside the window: the old entry is pruned, so the uuid is fresh again.
    assert cache.check_and_add("uuid-1", now_ms=10_000) is True


def test_nonce_cache_rejects_empty_uuid():
    cache = NonceCache()
    assert cache.check_and_add("", now_ms=1000) is False


def test_nonce_cache_fails_closed_at_capacity_without_evicting_fresh_entries():
    cache = NonceCache(window_ms=8000, max_entries=2)
    assert cache.check_and_add("one", now_ms=1000)
    assert cache.check_and_add("two", now_ms=1000)
    assert not cache.check_and_add("three", now_ms=1000)
    assert len(cache._seen) == 2
    # Once the signature/replay window has elapsed, stale entries are pruned
    # and new traffic is admitted again.
    assert cache.check_and_add("three", now_ms=10_000)


def test_previous_release_server_url_migrates_to_the_current_default():
    from rlvr.config import LEGACY_PROBLEM_SERVER_URLS

    legacy = next(iter(LEGACY_PROBLEM_SERVER_URLS))
    settings = Settings(_env_file=None, problem_server_url=legacy + "/")
    assert settings.problem_server_url == "https://d344p1xue0u9qs.cloudfront.net"
    assert settings.problem_server_url_migrated_from == legacy + "/"
    plain = Settings(_env_file=None, problem_server_url="https://private.example")
    assert plain.problem_server_url == "https://private.example" and not plain.problem_server_url_migrated_from
