# Bedrock demo miner

The demo miner implements the V3 signed miner protocol. It sends the public
task to Amazon Bedrock, uploads the resulting patch or script and its canonical
trajectory, then signs the exact response body. It never receives the verifier.

## Setup

Requirements:

- Python 3.10–3.12
- a funded hotkey registered on the subnet
- an Amazon Bedrock API key for a Chat Completions model that returns reasoning
  and five token alternatives
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

The provider must return genuine reasoning, selected-token log probabilities,
and exactly five alternatives per token. The miner fails closed when those
fields are absent; it never fabricates trajectory evidence.
