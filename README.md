# RLVR subnet

This repository contains the public validator for Bittensor Finney NETUID 5.
Validators lease V3 repository or terminal tasks, send the server-assigned task
to miners, commit the signed responses, retrieve the verifier, grade each miner
in an isolated Docker workspace, and submit locally calculated weights.

## Run a validator

Requirements:

- Linux with Python 3.10–3.12
- Docker with the daemon running
- a registered validator hotkey on Finney NETUID 5
- a system clock synchronized with NTP

From the repository root:

```bash
./setup_validator.sh --wallet-name YOUR_WALLET --wallet-hotkey YOUR_HOTKEY
./start_validator.sh
```

Setup creates `.venv` and `.env`, installs dependencies, pulls and checks the
release-pinned V3 sandbox image, and verifies the problem service and local
clock. If wallet arguments are omitted, set only `WALLET_NAME` and
`WALLET_HOTKEY` in `.env`.

Dispatch, grading, scoring, cadence, resource limits, sandbox image, and owner
burn are fixed in release policy. Operators do not configure them in `.env`.
The owner burn share is 0%.

The validator stores its scoring window in `data/validator_scores.json`.
Preserve that file across restarts.

## Protocol

```text
problem server -> workspace and public task -> validator -> selected miners
problem server <- exact signed responses ---- validator <- artifact references
problem server -> verifier after commit ----- validator
                                              |
                                              v
                                   isolated per-miner grading
                                              |
                                              v
                                     local scores and weights
```

Miners return either a Git unified diff or a Bash script plus a canonical
trajectory. The verifier is unavailable until the response set is durably
committed. Candidate code cannot access verifier files, expected outputs, other
miners' workspaces, the network, or the trusted result record. Infrastructure
or protocol failures abandon the round without changing miner scores.

The complete V3 contract is documented in [`docs/V3_TASKS.md`](docs/V3_TASKS.md).

## Demo miner

The included demo miner is a protocol reference backed by Amazon Bedrock Chat
Completions. It requires its own Bedrock API key; validators do not need one.
See [`docs/DEMO_MINER.md`](docs/DEMO_MINER.md).

## Development

```bash
. .venv/bin/activate
pip install -e '.[chain,dev]'
pytest -q
```

V3 sandbox throughput can be measured with
[`scripts/benchmark_v3_supervisor.py`](scripts/benchmark_v3_supervisor.py).
Problem construction and private task provenance are not part of this
repository.

## Repository contents

- `rlvr/v3/`: V3 wire models, artifact handling, grading, and orchestration.
- `rlvr/neurons/`: validator lifecycle, signed miner transport, and demo miner.
- `rlvr/scoring/`: local score history and weight calculation.
- `docker/polyglot-sandbox/`: reproducible V3 grading image.
