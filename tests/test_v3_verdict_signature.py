"""A validator numbers and signs each round, and keeps it until the server has it."""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time

import pytest
from pydantic import ValidationError

from rlvr import protocol
from rlvr.v3.api import RoundVerdict, SignedRound, serialize_signed_round
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.ledger import (
    IDLE_S,
    RETRY_MAX_S,
    RETRY_MIN_S,
    RoundLedger,
    next_backoff,
    pass_interval,
)
from rlvr.v3.reasons import MinerReason, Stage
from rlvr.v3.round import MinerEvaluation, _report_signed_round
from rlvr.v3.verdicts import ROUND_DOMAIN, round_message, sign_round, verify_round

TASK = "a" * 64


@pytest.fixture(autouse=True)
def offline_signatures(monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)


def verdict(uid, passed, latency=None):
    return RoundVerdict(uid=uid, hotkey=f"hk-{uid}", passed=passed, response_latency_ms=latency)


# --------------------------------------------------------------------------- #
# the signed bytes
# --------------------------------------------------------------------------- #
def test_the_signed_message_is_the_domain_tag_plus_canonical_json_sorted_by_uid():
    message = round_message("chal-1", TASK, 7, [verdict(9, False, 250), verdict(2, True, None)])
    expected = (
        b'{"challenge_id":"chal-1","round_seq":7,"task_id":"' + TASK.encode() + b'","verdicts":['
        b'{"hotkey":"hk-2","passed":true,"response_latency_ms":null,"uid":2},'
        b'{"hotkey":"hk-9","passed":false,"response_latency_ms":250,"uid":9}]}'
    )
    assert message == ROUND_DOMAIN + expected
    # the reference the server team checks against: Python's own canonical form
    assert message[len(ROUND_DOMAIN):] == json.dumps(json.loads(expected), sort_keys=True, separators=(",", ":")).encode()


SERVER_VECTOR_HEX = (
    "686f6e652d76332d726f756e642d76313a7b226368616c6c656e67655f6964223a223066336339613765356232643463313861366531663039623364376332613534222c22726f756e645f736571223a372c227461736b5f6964223a2239623962396239623962396239623962396239623962396239623962396239623962396239623962396239623962396239623962396239623962396239623962222c227665726469637473223a5b7b22686f746b6579223a2235464c53696743394847524b566842394669456f3459336b6f50734e6d426d4c4a62705867326d703168586353353959222c22706173736564223a66616c73652c22726573706f6e73655f6c6174656e63795f6d73223a6e756c6c2c22756964223a307d2c7b22686f746b6579223a22354441416e726a375648547a6e6e32415742656d4d757942775a577336464e466a64795658556559756d335054584679222c22706173736564223a747275652c22726573706f6e73655f6c6174656e63795f6d73223a313230302c22756964223a357d2c7b22686f746b6579223a223546486e655734367847586773356d5569766555347362547947427a6d73745573705a43393255686a4a4d3639347479222c22706173736564223a747275652c22726573706f6e73655f6c6174656e63795f6d73223a38333231312c22756964223a31327d5d7d"
)


def test_the_server_teams_vector_matches_byte_for_byte():
    # Built by the problem-server team from the public development key //Alice,
    # verdicts deliberately out of uid order, one null latency, one failure.
    verdicts = [
        RoundVerdict(uid=12, hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", passed=True, response_latency_ms=83211),
        RoundVerdict(uid=0, hotkey="5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y", passed=False, response_latency_ms=None),
        RoundVerdict(uid=5, hotkey="5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy", passed=True, response_latency_ms=1200),
    ]
    message = round_message("0f3c9a7e5b2d4c18a6e1f09b3d7c2a54", "9b" * 32, 7, verdicts)
    assert message.hex() == SERVER_VECTOR_HEX and len(message) == 508


@pytest.mark.skipif(not protocol.crypto_available(), reason="no live crypto stack")
def test_the_server_teams_signature_verifies_here(monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", True)
    assert protocol.verify_detached(
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        bytes.fromhex(SERVER_VECTOR_HEX),
        "0x54efdc3d2b2f04c03d6d2990edd797d1a2fcf201ca7db6376ded7ec468165c6866e47c8f1244b6fc8bd3cf3eae2d25403bdb37278719251c11fda8c82774288f",
    )


def test_only_the_four_fields_are_signed():
    class Richer:
        uid, hotkey, passed, response_latency_ms = 2, "hk-2", False, 5
        grading_duration_ms, reason_code = 999, "check_failed"

    assert round_message("chal-1", TASK, 1, [verdict(2, False, 5)]) == round_message("chal-1", TASK, 1, [Richer()])


def test_messages_are_signed_over_exact_types_only():
    class Loose:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    good = {"uid": 2, "hotkey": "hk-2", "passed": True, "response_latency_ms": None}
    for bad in ({"passed": 1}, {"uid": 2.0}, {"uid": "2"}, {"hotkey": 2}, {"response_latency_ms": 1.5}, {"response_latency_ms": "5"}):
        with pytest.raises(TypeError):
            round_message("chal-1", TASK, 1, [Loose(**(good | bad))])
    with pytest.raises(TypeError):
        round_message("chal-1", TASK, "1", [Loose(**good)])
    with pytest.raises(ValueError):
        round_message("chal-1", TASK, 1, [Loose(**good), Loose(**good)])  # one uid twice


def test_signature_verifies_for_the_signer_and_for_nothing_else():
    verdicts = [verdict(2, True, 10), verdict(3, False, 20)]
    signature = sign_round("validator", "chal-1", TASK, 5, verdicts)
    assert signature.startswith("0x")
    assert verify_round("validator", signature, "chal-1", TASK, 5, verdicts)
    assert not verify_round("other-validator", signature, "chal-1", TASK, 5, verdicts)
    assert not verify_round("validator", signature, "chal-1", TASK, 6, verdicts)  # another sequence number
    assert not verify_round("validator", signature, "chal-2", TASK, 5, verdicts)  # another round
    assert not verify_round("validator", signature, "chal-1", TASK, 5, [verdict(2, True, 10), verdict(3, True, 20)])  # a verdict flipped
    assert not verify_round("validator", signature, "chal-1", TASK, 5, [verdict(2, True, 11), verdict(3, False, 20)])  # a latency changed
    assert not verify_round("validator", signature, "chal-1", TASK, 5, verdicts[:1])  # a verdict dropped
    assert not verify_round("validator", "0xdead", "chal-1", TASK, 5, verdicts)
    assert not verify_round("validator", signature, "chal-1", TASK, "5", verdicts)  # bad types never verify


def test_a_real_key_is_never_signed_or_verified_without_real_crypto():
    ss58 = "5GZ2KuT2TtLbYTtsMcgAtazo6KQ4bc57ykZgyQv9oit3y7iq"
    message = round_message("chal-1", TASK, 1, [verdict(2, True)])
    assert not protocol.verify_detached(ss58, message, protocol._hmac_sign_bytes(ss58, message))
    with pytest.raises(RuntimeError):
        protocol.sign_detached(ss58, message)


def test_detached_signing_covers_raw_bytes_in_the_fallback():
    message = b"hone-v3-round-v1:\xff\x00"
    signature = protocol.sign_detached("validator", message)
    assert protocol.verify_detached("validator", message, signature)
    assert not protocol.verify_detached("validator", message[:-1], signature)


@pytest.mark.skipif(not protocol.crypto_available(), reason="no live crypto stack")
def test_live_keypair_signs_and_any_holder_of_the_address_verifies(monkeypatch):
    from bittensor_wallet import Keypair

    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", True)
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = type("Wallet", (), {"hotkey": keypair})()
    verdicts = [verdict(2, True, 10)]
    signature = sign_round(wallet, "chal-1", TASK, 1, verdicts)
    assert len(signature) == 2 + 128 and verify_round(keypair.ss58_address, signature, "chal-1", TASK, 1, verdicts)
    assert not verify_round(keypair.ss58_address, signature, "chal-1", TASK, 2, verdicts)
    other = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    assert not verify_round(other.ss58_address, signature, "chal-1", TASK, 1, verdicts)
    with pytest.raises(RuntimeError):
        protocol.sign_detached("opaque-id", b"x")


@pytest.mark.skipif(not protocol.crypto_available(), reason="no live crypto stack")
def test_a_signature_over_the_wrapped_message_is_refused(monkeypatch):
    """Keypair.verify accepts a signature made over "<Bytes>" + message +
    "</Bytes>" as well; the ledger accepts only one over the exact bytes."""
    from bittensor_wallet import Keypair

    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", True)
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    message = round_message("chal-1", TASK, 1, [verdict(2, True)])
    wrapped = "0x" + keypair.sign(b"<Bytes>" + message + b"</Bytes>").hex()
    assert keypair.verify(message, bytes.fromhex(wrapped[2:]))  # the library is lenient
    assert not protocol.verify_detached(keypair.ss58_address, message, wrapped)  # we are not
    exact = protocol.sign_detached(keypair, message)
    assert protocol.verify_detached(keypair.ss58_address, message, exact)


# --------------------------------------------------------------------------- #
# the wire
# --------------------------------------------------------------------------- #
def report(**fields):
    base = {"protocol_version": 3, "challenge_id": "chal-1", "task_id": TASK, "verdicts": [verdict(2, True)], "round_seq": 4, "signature": "0xab12"}
    return SignedRound(**(base | fields))


def test_a_signed_round_is_numbered_signed_sorted_and_names_each_miner_once():
    assert report().round_seq == 4
    for bad in ({"round_seq": 0}, {"round_seq": None}, {"signature": None}, {"signature": "0xabc"}, {"signature": "ab12"}, {"verdicts": []}):
        with pytest.raises(ValidationError):
            report(**bad)
    with pytest.raises(ValidationError):
        report(verdicts=[verdict(3, True), verdict(2, True)])  # out of uid order
    with pytest.raises(ValidationError):
        report(verdicts=[verdict(2, True), RoundVerdict(uid=3, hotkey="hk-2", passed=True)])  # one hotkey twice
    with pytest.raises(ValidationError):
        report(verdicts=[RoundVerdict(uid=2, hotkey="hk-2", passed=True, response_latency_ms=3_600_001)])


def test_the_wire_carries_exactly_the_signed_fields_plus_the_envelope():
    body = json.loads(serialize_signed_round(report(verdicts=[verdict(2, True, None), verdict(5, False, 7)])))
    assert body == {
        "protocol_version": 3, "challenge_id": "chal-1", "task_id": TASK, "round_seq": 4, "signature": "0xab12",
        "verdicts": [
            {"uid": 2, "hotkey": "hk-2", "passed": True, "response_latency_ms": None},
            {"uid": 5, "hotkey": "hk-5", "passed": False, "response_latency_ms": 7},
        ],
    }


# --------------------------------------------------------------------------- #
# the ledger: number, sign, keep, resend
# --------------------------------------------------------------------------- #
class Server:
    def __init__(self, *statuses):
        self.statuses = list(statuses)
        self.bodies = []

    async def signed_round_status(self, body):
        self.bodies.append(bytes(body))
        return self.statuses.pop(0) if self.statuses else 200

    @property
    def received(self):
        return [SignedRound.model_validate_json(body) for body in self.bodies]


def test_the_outbox_is_created_level_by_level_wherever_it_is_pointed(tmp_path):
    deep = tmp_path / "data" / "nested" / "feedback-outbox"
    ledger = RoundLedger("validator", deep)
    assert not deep.exists()  # nothing is created by construction, so startup cannot fail on it
    assert ledger.pending() == [] and deep.is_dir()  # the first use creates it, level by level
    assert ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10).seq == 1
    assert (deep / "seq").read_text() == "1" and len(ledger.pending()) == 1


def test_the_sequence_number_counts_up_and_survives_a_restart(tmp_path):
    first = RoundLedger("validator", tmp_path / "outbox")
    assert first.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10).seq == 1
    assert first.prepare("chal-2", TASK, [verdict(2, True)], expires_at=10).seq == 2
    again = RoundLedger("validator", tmp_path / "outbox")
    assert again.prepare("chal-3", TASK, [verdict(2, True)], expires_at=10).seq == 3


def test_a_crash_between_the_record_and_the_counter_repeats_no_number(tmp_path, monkeypatch):
    import rlvr.v3.ledger as ledger_module

    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    real = ledger_module._write_atomic

    def crash_before_counter(path, data):
        if path.name == "seq":
            raise OSError("power lost")
        real(path, data)

    monkeypatch.setattr(ledger_module, "_write_atomic", crash_before_counter)
    with pytest.raises(OSError):
        ledger.prepare("chal-2", TASK, [verdict(2, True)], expires_at=10**10)
    monkeypatch.setattr(ledger_module, "_write_atomic", real)
    assert (tmp_path / "outbox" / "seq").read_text() == "1"  # the counter never caught up
    assert [p.name for p in ledger.pending()] == ["round-000000000001.json", "round-000000000002.json"]
    # after the restart the server takes both kept rounds BEFORE the next round is prepared:
    # deleting round 2 must not let its number be used again
    recovered = RoundLedger("validator", tmp_path / "outbox")
    assert asyncio.run(recovered.retry_pending(Server(200, 200))) is False
    assert recovered.pending() == [] and (tmp_path / "outbox" / "seq").read_text() == "2"
    assert recovered.prepare("chal-3", TASK, [verdict(2, True)], expires_at=10**10).seq == 3


def test_a_failed_signature_burns_no_number_and_leaves_no_record(tmp_path, monkeypatch):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    monkeypatch.setattr("rlvr.v3.ledger.sign_round", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
    with pytest.raises(RuntimeError):
        ledger.prepare("chal-2", TASK, [verdict(2, True)], expires_at=10**10)
    assert ledger.last_seq() == 1 and len(ledger.pending()) == 1


def test_a_prepared_round_is_signed_and_resent_byte_for_byte_until_the_server_has_it(tmp_path):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    kept = ledger.prepare("chal-1", TASK, [verdict(3, False, 20), verdict(2, True, 10)], expires_at=10**10)
    sent = SignedRound.model_validate_json(kept.body)
    assert sent.round_seq == 1 and verify_round("validator", sent.signature, "chal-1", TASK, 1, sent.verdicts)
    assert [item.uid for item in sent.verdicts] == [2, 3]  # sorted for signing
    assert [path.name for path in ledger.pending()] == ["round-000000000001.json"]
    server = Server(503, None, 200)
    assert asyncio.run(ledger.deliver(server, kept)) == "retry"
    assert asyncio.run(ledger.retry_pending(server)) is True  # None: nothing answered, kept
    assert asyncio.run(ledger.retry_pending(server)) is False  # 200: delivered
    assert ledger.pending() == [] and server.bodies == [kept.body] * 3


@pytest.mark.parametrize("status,outcome", [(200, "delivered"), (409, "delivered"), (400, "dropped"), (410, "dropped"), (503, "retry"), (429, "retry"), (None, "retry")])
def test_delivery_outcomes_decide_whether_the_round_is_kept(tmp_path, status, outcome):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    kept = ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    server = Server(status)
    assert asyncio.run(ledger.deliver(server, kept)) == outcome
    assert (ledger.pending() == []) is (outcome != "retry")
    assert server.bodies == [kept.body]  # one attempt, never a second form of the round


def test_an_expired_round_is_dropped_without_a_send(tmp_path, capsys):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=1_000)
    server = Server()
    assert asyncio.run(ledger.retry_pending(server, now=1_000)) is False
    assert server.bodies == [] and ledger.pending() == []
    assert "expired" in capsys.readouterr().out


def test_bad_records_and_failing_sends_never_stop_the_other_rounds(tmp_path):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    ledger.prepare("chal-2", TASK, [verdict(2, True)], expires_at=10**10)
    outbox = tmp_path / "outbox"
    good = (outbox / "round-000000000001.json").read_text()
    (outbox / "round-000000000001.json").write_text(good.replace('"expires_at": 10000000000', '"expires_at": "soon"'))
    (outbox / "round-000000000000.json").mkdir()  # a directory wearing a record's name cannot be unlinked

    class Flaky(Server):
        async def signed_round_status(self, body):
            if len(self.bodies) == 0:
                self.bodies.append(bytes(body))
                raise RuntimeError("socket exploded")
            return await super().signed_round_status(body)

    server = Flaky(200)
    assert asyncio.run(ledger.retry_pending(server)) is True  # the exploding send is kept for next pass
    assert [p.name for p in ledger.pending()] == ["round-000000000000.json", "round-000000000002.json"]
    assert asyncio.run(ledger.retry_pending(server)) is True  # the stray directory stays, the round went
    assert [p.name for p in ledger.pending()] == ["round-000000000000.json"]
    assert (outbox / "seq").read_text() == "2"  # dropping and delivering never lets a number come back


def test_numbering_and_removal_from_two_threads_never_repeat_or_lower_a_number(tmp_path):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    issued, errors = [], []
    deadline = time.monotonic() + 30

    def prepare_many():
        try:
            for i in range(60):
                issued.append(ledger.prepare(f"chal-{i}", TASK, [verdict(2, True)], expires_at=10**10).seq)
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    def forget_as_they_come():
        try:
            seen = 0
            while seen < 60 and time.monotonic() < deadline:
                for path in ledger.pending():
                    ledger._forget(int(path.name[6:-5]))
                    seen += 1
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=prepare_many), threading.Thread(target=forget_as_they_come)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == [] and sorted(issued) == list(range(1, 61))
    assert ledger.pending() == [] and (tmp_path / "outbox" / "seq").read_text() == "60"
    assert ledger.prepare("chal-last", TASK, [verdict(2, True)], expires_at=10**10).seq == 61


def test_backoff_doubles_to_a_cap_while_pending_and_resets_when_nothing_is_kept():
    backoff = RETRY_MIN_S
    seen = []
    for _ in range(10):
        backoff = next_backoff(backoff, True)
        seen.append(backoff)
    assert seen == [10, 20, 40, 80, 160, 320, 640, 900, 900, 900]
    assert next_backoff(RETRY_MAX_S, False) == RETRY_MIN_S
    # idle polling never feeds the backoff: after an idle stretch a fresh failure waits the minimum
    assert pass_interval(RETRY_MIN_S, False) == IDLE_S
    assert pass_interval(RETRY_MIN_S, True) == RETRY_MIN_S
    assert pass_interval(RETRY_MAX_S, True) == RETRY_MAX_S


def test_an_expired_round_is_dropped_at_the_moment_of_sending(tmp_path, capsys):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    kept = ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=1_000)
    server = Server()
    assert asyncio.run(ledger.deliver(server, kept, now=1_001)) == "dropped"
    assert server.bodies == [] and ledger.pending() == [] and "expired" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the round reports every miner the lease named
# --------------------------------------------------------------------------- #
def evaluation(uid, status, *, latency=5, code=None, stage=Stage.CHECK):
    result = EvaluationResult(status, "" if status == "passed" else "x", (), None, code, stage)
    return MinerEvaluation(uid, f"hk-{uid}", latency, result, 3)


async def report_and_wait(*args, **kwargs):
    """Report a round, then let its background first send finish."""
    kept = await _report_signed_round(*args, **kwargs)
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    await asyncio.gather(*pending)
    return kept


def test_the_signed_round_names_every_pool_miner_with_the_right_latency(tmp_path):
    pool = [(3, "hk-3"), (1, "hk-1"), (4, "hk-4"), (2, "hk-2"), (6, "hk-6"), (5, "hk-5"), (1, "hk-1")]  # unordered, one repeat
    evaluations = [
        evaluation(1, "passed", latency=120),
        evaluation(2, "failed", latency=340, code=MinerReason.CHECK_FAILED),
        evaluation(4, "rejected", latency=50, code=MinerReason.ARTIFACT_INVALID, stage=Stage.COMMIT),  # the server refused it
        evaluation(5, "rejected", latency=800, code=MinerReason.PATCH_REJECTED, stage=Stage.PATCH),  # we refused it after download
        evaluation(6, "failed", latency=4_000_000, code=MinerReason.TIMEOUT),  # a latency past the wire cap
    ]
    ledger = RoundLedger("validator", tmp_path / "outbox")
    server = Server(503)
    assert asyncio.run(report_and_wait(server, ledger, "chal-1", TASK, pool, evaluations, expires_at=10**10))
    sent = server.received[0]
    assert sent.round_seq == 1 and verify_round("validator", sent.signature, "chal-1", TASK, 1, sent.verdicts)
    assert [(item.uid, item.passed, item.response_latency_ms) for item in sent.verdicts] == [
        (1, True, 120),
        (2, False, 340),
        (3, False, None),  # silent
        (4, False, None),  # rejected at commit: no graded answer
        (5, False, 800),  # rejected after download: it answered, the latency stands
        (6, False, None),  # out of range is sent as unknown, never refused
    ]
    assert len(ledger.pending()) == 1  # refused once: kept for the resender


def test_a_round_where_nobody_was_graded_is_still_reported(tmp_path):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    server = Server(200)
    pool = [(1, "hk-1"), (2, "hk-2")]
    rejected = [evaluation(1, "rejected", code=MinerReason.ARTIFACT_INVALID, stage=Stage.COMMIT)]
    assert asyncio.run(report_and_wait(server, ledger, "chal-1", TASK, pool, rejected, expires_at=10**10))
    assert [(item.uid, item.passed, item.response_latency_ms) for item in server.received[0].verdicts] == [(1, False, None), (2, False, None)]
    assert ledger.pending() == []


def test_the_round_is_kept_before_the_first_send_and_the_send_does_not_hold_the_round(tmp_path):
    class Slow(Server):
        def __init__(self):
            super().__init__(200)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def signed_round_status(self, body):
            self.started.set()
            await self.release.wait()
            return await super().signed_round_status(body)

    async def scenario():
        ledger = RoundLedger("validator", tmp_path / "outbox")
        server = Slow()
        kept = await _report_signed_round(server, ledger, "chal-1", TASK, [(1, "hk-1")], [evaluation(1, "passed")], expires_at=10**10)
        assert kept is True and len(ledger.pending()) == 1  # returned while the send is still on its way
        await server.started.wait()
        assert await ledger.retry_pending(server) is True  # the resender leaves a round already in flight alone
        assert len(server.bodies) == 0
        server.release.set()
        await asyncio.gather(*[task for task in asyncio.all_tasks() if task is not asyncio.current_task()])
        assert len(server.bodies) == 1 and ledger.pending() == []

    asyncio.run(scenario())


def test_a_ledger_needs_a_wallet(tmp_path):
    with pytest.raises(ValueError):
        RoundLedger(None, tmp_path / "outbox")


def test_a_kept_record_is_checked_like_the_server_would_before_it_is_resent(tmp_path, capsys):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    path = ledger.pending()[0]
    tampered = json.loads(path.read_text())
    tampered["body"] = tampered["body"].replace('"passed":true', '"passed":false')
    path.write_text(json.dumps(tampered))
    server = Server()
    assert asyncio.run(ledger.retry_pending(server)) is False
    assert server.bodies == [] and ledger.pending() == []  # dropped, never sent
    assert "dropping kept round" in capsys.readouterr().out
    # a record whose number disagrees with its name is dropped the same way
    again = ledger.prepare("chal-2", TASK, [verdict(2, True)], expires_at=10**10)
    (tmp_path / "outbox" / "round-000000000009.json").write_text((tmp_path / "outbox" / f"round-{again.seq:012d}.json").read_text())
    assert asyncio.run(ledger.retry_pending(server)) is False
    assert len(server.bodies) == 1 and ledger.pending() == []
    assert ledger.last_seq() == 9  # the stray number is spent, never reused


def test_stray_files_are_ignored_and_interrupted_writes_are_swept(tmp_path):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    outbox = tmp_path / "outbox"
    (outbox / "round-bad.json").write_text("{}")
    (outbox / ".round-000000000002.json.123.abcd.tmp").write_text("half")
    (outbox / ".seq.123.abcd.tmp").write_text("half")
    assert [p.name for p in ledger.pending()] == ["round-000000000001.json"]
    assert not list(outbox.glob(".*.tmp")) and (outbox / "round-bad.json").exists()
    assert ledger.prepare("chal-2", TASK, [verdict(2, True)], expires_at=10**10).seq == 2


def test_two_ledgers_on_one_directory_never_share_a_number(tmp_path):
    first = RoundLedger("validator", tmp_path / "outbox")
    second = RoundLedger("validator", tmp_path / "outbox")
    issued = []

    def work(ledger, label):
        for i in range(25):
            issued.append(ledger.prepare(f"{label}-{i}", TASK, [verdict(2, True)], expires_at=10**10).seq)

    threads = [threading.Thread(target=work, args=(first, "a")), threading.Thread(target=work, args=(second, "b"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(issued) == list(range(1, 51))


def test_lock_failures_leave_no_descriptor_and_no_held_lock(tmp_path, monkeypatch):
    import fcntl

    ledger = RoundLedger("validator", tmp_path / "outbox")
    real = fcntl.flock
    calls = {"n": 0}

    def failing(handle, operation):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("no locks available")
        return real(handle, operation)

    monkeypatch.setattr(fcntl, "flock", failing)
    with pytest.raises(OSError):
        ledger.pending()
    assert ledger._handle is None and ledger._depth == 0
    assert ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10).seq == 1  # the next operation works


def test_numbers_beyond_twelve_digits_are_still_recognised(tmp_path):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.pending()
    (tmp_path / "outbox" / "seq").write_text("999999999999")
    kept = ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    assert kept.seq == 10**12
    assert [p.name for p in ledger.pending()] == ["round-1000000000000.json"]
    assert ledger.last_seq() == 10**12


def test_close_cancels_a_first_send_still_running_and_keeps_the_record(tmp_path):
    class Stalled(Server):
        async def signed_round_status(self, body):
            self.bodies.append(bytes(body))
            await asyncio.sleep(3600)
            return 200

    async def scenario():
        ledger = RoundLedger("validator", tmp_path / "outbox")
        server = Stalled()
        await _report_signed_round(server, ledger, "chal-1", TASK, [(1, "hk-1")], [evaluation(1, "passed")], expires_at=10**10)
        await asyncio.sleep(0)  # let the first send start
        assert len(ledger._sends) == 1 and len(server.bodies) == 1
        await ledger.close()
        assert ledger._sends == set() and len(ledger.pending()) == 1  # kept for next time

    asyncio.run(scenario())


def test_a_read_error_keeps_the_record_for_the_next_pass(tmp_path, monkeypatch):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    real = pathlib.Path.read_bytes
    failures = {"n": 1}

    def flaky(self):
        if self.name.startswith("round-") and failures["n"]:
            failures["n"] -= 1
            raise OSError(5, "Input/output error")
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", flaky)
    server = Server(200)
    assert asyncio.run(ledger.retry_pending(server)) is True  # not sent, not dropped
    assert server.bodies == [] and len(ledger.pending()) == 1
    assert asyncio.run(ledger.retry_pending(server)) is False  # delivered on the next pass
    assert len(server.bodies) == 1 and ledger.pending() == []


def test_a_record_that_is_not_text_is_dropped_not_kept_forever(tmp_path, capsys):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    ledger.pending()[0].write_bytes(b"\xff\xfe not a record")
    server = Server()
    assert asyncio.run(ledger.retry_pending(server)) is False
    assert server.bodies == [] and ledger.pending() == []
    assert "UnicodeDecodeError" in capsys.readouterr().out


def test_a_record_of_an_unknown_format_is_dropped_with_a_warning(tmp_path, capsys):
    ledger = RoundLedger("validator", tmp_path / "outbox")
    ledger.prepare("chal-1", TASK, [verdict(2, True)], expires_at=10**10)
    path = ledger.pending()[0]
    record = json.loads(path.read_text())
    assert record["format"] == 1
    record["format"] = 2
    path.write_text(json.dumps(record))
    server = Server()
    assert asyncio.run(ledger.retry_pending(server)) is False
    assert server.bodies == [] and ledger.pending() == [] and "record format" in capsys.readouterr().out
