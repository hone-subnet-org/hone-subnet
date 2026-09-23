# Bedrock demo miner

The demo miner implements the V3 signed miner protocol. It downloads and verifies
the leased workspace, gives the model read-only file access, uploads the resulting
patch or script and its canonical trajectory, then signs the exact response body.
It never receives the verifier.

The model can list directories and read files in pages using `workspace` tool
requests. Each model turn and file read is recorded in the trajectory. Reads are
limited to 32 KiB per call and directory listings to 200 entries per page. The
miner does not execute repository code. Temporary workspace files are removed
after generation.

`MINER_MAX_WORKSPACE_TOOL_CALLS` defaults to 24, followed by one final model turn.
Workspace preparation and all model turns share the solve deadline, reserving
`BEDROCK_UPLOAD_RESERVE_S` for uploads. More file reads can increase model cost and
solve latency. The miner needs local disk space for the compressed archive, its
expanded tar stream, and extracted files for each concurrent solve.

## Setup

Requirements:

- Python 3.10–3.12
- a funded hotkey registered on the subnet
- an Amazon Bedrock API key for a Chat Completions model
- a public TCP port reachable by validators

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[chain,miner]'
cp .env.example .env
```

Set:

```dotenv
BEDROCK_API_KEY=<your-key>
BEDROCK_BASE_URL=https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1
BEDROCK_MODEL=moonshotai.kimi-k2.5

NETUID=<subnet-id>
SUBTENSOR_NETWORK=test
WALLET_NAME=miner
WALLET_HOTKEY=default
AXON_PORT=8091
```

Use the Bedrock endpoint for the region containing the selected model. Keep the
API key only in `.env` or the process environment; `.env` is ignored by Git.

Start the miner:

```bash
./start_demo_miner.sh
```

The default authorization accepts registered validator hotkeys with a validator
permit. `MINER_MIN_STAKE`, `MINER_MAX_CONCURRENT_REQUESTS`, and
`MINER_REQUIRE_VALIDATOR_PERMIT` control who can spend the model account's
quota.

Reasoning text and selected-token log probabilities with five alternatives
per token are recorded when the provider returns them. Log probabilities that
are present but malformed are rejected; the miner never fabricates trajectory
evidence.
