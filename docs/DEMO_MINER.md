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

`MINER_MAX_WORKSPACE_TOOL_CALLS` defaults to 48. A call over the budget is not run; the model is asked for its submission instead.
Workspace preparation and all model turns share the solve deadline, reserving
`BEDROCK_UPLOAD_RESERVE_S` for uploads. More file reads can increase model cost and
solve latency. The miner needs local disk space for the compressed archive, its
expanded tar stream, and extracted files for each concurrent solve.


## Try a real task locally

Two tasks the problem server issued once and retired live under
`tests/fixtures/`, one of each kind. They let you check your whole setup
before mining:

- `v3-yamlcpp-e6225d04`, a repository task: yaml-cpp with a private feature,
  two planted defects, six hidden checks and a reference fix. The answer is a
  unified diff.
- `v3-terminal-urlooker-651460ec`, a terminal task: a pinned Go checkout plus
  incident material, six hidden checks and a reference script. The answer is
  a bash script that runs in the workspace and leaves the required results.

```
python scripts/try_task.py extract /path/to/work      # workspace + instruction
# produce a unified diff against that workspace with your own miner
python scripts/try_task.py grade /path/to/fix.diff    # graded in the pinned sandbox

# the terminal task
python scripts/try_task.py --fixture tests/fixtures/v3-terminal-urlooker-651460ec extract /path/to/work
python scripts/try_task.py --fixture tests/fixtures/v3-terminal-urlooker-651460ec grade /path/to/solve.sh
```

Grading needs Docker and the pinned sandbox image, pulled with
`docker pull "$(python -c 'from rlvr.policy import RELEASE_POLICY; print(RELEASE_POLICY.v3_image)')"`.
The output is the validator's verdict: status, reason code, every check's
outcome, and for a failed check the same display a failure notice carries.
`v3-yamlcpp-e6225d04/reference.diff` passes all six checks; applying only its
first file fails at the fourth. `v3-terminal-urlooker-651460ec/reference.sh`
passes all six; an empty script fails at the first, and the reference script
followed by a change to the recovery inputs fails at the sixth.

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
BEDROCK_MODEL=us.moonshotai.kimi-k3

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
