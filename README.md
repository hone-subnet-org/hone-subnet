# RLVR subnet

This repository contains the validator for RLVR on Bittensor Finney, NETUID 5.
Validators lease a public coding challenge, rotation-deal it to half of the
serving miners,
commit the signed responses, reveal the hidden tests, grade every response in a
local Docker sandbox, and submit the resulting weights on chain.

## Launch emission policy

This release hard-codes the owner burn share at 0%. Validators submit the
ordinary normalized miner score vector without reserving weight for the subnet
owner. An all-zero vector is not submitted. Changing this value requires a
validator release; it is not an environment override.

## Run a validator

Requirements:

- Linux or macOS with Python 3.10–3.12
- Docker with the daemon running
- a registered validator hotkey on Finney NETUID 5
- a system clock synchronized with NTP

The launch configuration can run up to 16 one-CPU, 256 MiB grading containers
at once. A 16-vCPU, 16-GiB host is recommended; hosts with fewer CPU cores will
grade each sampled half more slowly but will not reduce the sample size.
Use a stable broadband connection: the production server rejects a commit
whose request body takes more than 120 seconds to upload.
The validator records its first four completed challenges before submitting an
on-chain weight vector; paced or unavailable rounds do not advance that gate.
The SN5 defaults are 38 blocks (about 7.5 minutes) between challenge attempts
and 180 blocks between weight attempts. The problem service remains authoritative:
when an attempt is too early or the shared global lease slot is busy, its
`Retry-After` response defers leasing without blocking weight scheduling. At
startup the validator reads the chain weight rate limit and raises the effective
weight interval to at least that limit plus 20 blocks.

After cloning this repository, run the bootstrap with the names of your
existing Bittensor wallet and hotkey from the repository root:

```bash
./setup_validator.sh --wallet-name YOUR_WALLET --wallet-hotkey YOUR_HOTKEY
./start_validator.sh
```

The setup is safe to rerun. It creates `.venv` and `.env`, installs the pinned
chain dependencies, pulls the immutable multi-architecture sandbox image, runs
a real Python/GNU `timeout` container smoke test, and checks the production
problem server and local clock. It does not create, read, or modify wallet keys.

If you omit the wallet arguments, edit only `WALLET_NAME` and `WALLET_HOTKEY`
in `.env`, then run `./start_validator.sh`. The production URL, Finney network,
NETUID 5, half-pool rotation, and sandbox image are already configured.

Validator dispatch, scoring, payment, and cadence are release policy rather
than operator settings. Pulling a release and restarting adopts that policy;
legacy policy entries in an existing `.env` are ignored and named in a startup
warning. Machine-specific resource limits and operator identity remain normal
environment settings.

Run the tests with:

```bash
. .venv/bin/activate
pip install -e '.[chain,dev]'
pytest -q
```

Miner developers can check solution formatting and basic behavior against five
examples in [`examples/sample_challenges`](examples/sample_challenges/README.md).

Validator hosts can measure local sandbox throughput with
[`scripts/benchmark_grading.py`](scripts/benchmark_grading.py); see
[`docs/GRADING_BENCHMARK.md`](docs/GRADING_BENCHMARK.md).

Local rollout shards are automatically limited to 2 GiB. Do not delete
`data/validator_scores.json`; it contains the validator's scoring history.

## Protocol

```text
private problem server
        |
        | public challenge
        v
validator ---- signed request ----> rotating half of serving miners
        |<--- signed responses -----|
        |
        | commit exact response set
        | retrieve hidden cases
        v
local Docker sandbox
        |
        v
verified rewards -> recent score window -> on-chain weights
```

The commit-before-reveal sequence prevents hidden evaluation cases from being
included in miner requests. The server never grades miner code or controls
validator weights. Each validator evaluates the signed responses locally.
A rejected lease consumes no problem. Once a challenge is leased, ordinary
failed miner calls—including timeouts and invalid responses—remain in the exact
committed submission list with their recorded error; they are not silently
dropped before reveal.

## Demo miner and development

A small GLM-5.2-backed demo miner is included only as a protocol reference. It
requires its own `GLM_API_KEY`; validators do not need a model-provider key.
See [`docs/DEMO_MINER.md`](docs/DEMO_MINER.md).

The subprocess executor and test-network defaults are for development only.
See [`docs/TESTNET_RUNBOOK.md`](docs/TESTNET_RUNBOOK.md) and
[`docs/DESIGN.md`](docs/DESIGN.md) for the testnet workflow and trust model.

## Repository contents

- `rlvr/problemserver/`: versioned challenge lease, commit, reveal, and
  feedback contracts plus the authenticated client.
- `rlvr/neurons/`: validator lifecycle and signed miner transport.
- `rlvr/execution/`: fail-closed Docker sandbox.
- `rlvr/scoring/`: local verification, reward allocation, and score state.
- `rlvr/protocol.py`: validator/miner wire models and signatures.

Problem construction is not part of this repository.
