# V3 test fixture: terminal task on urlooker (task 651460ec)

One real `terminal_script_v1` task, already served on mainnet and retired, packaged as a test
fixture. Miners solved it widely: every miner that returned a valid answer passed.

## Files

| File | What it is |
|---|---|
| `identity.json` | the task identity exactly as leased (hashes to `task_id.txt`) |
| `task_id.txt` | `651460ece039f4add65fc96ca62c053ff071b143d5fb8575758a591f987ad469` |
| `artifact_refs.json` | sha256 and sizes of the two archives |
| `workspace.tar.zst` | the terminal environment a miner receives (2.6 MiB, 15 MB expanded): the urlooker checkout plus a `recovery/` directory with the incident material |
| `verifier.tar.zst` | the hidden checks: `manifest.json`, 6 scripts in `checks/`, expected output in `gold/` |
| `reference.sh` | a correct Bash submission |

Profile `repo-polyglot-v1`, verifier policy `command-gold-digest-v1`, result tree `.`.

## Expected results

- **Empty script:** fails (the first check fails).
- **`reference.sh`:** passes all 6 checks.
- **On mainnet:** all miners with a valid answer passed; about 5 distinct solutions.

## Task

The instruction asks for an offline, reviewable alarm ledger rebuilt from a retained outbox,
with lost-reply retries and a mid-capture routing change to handle. The script runs in the
sandbox with no network (2 CPU, 4 GiB) and must produce the delivery tree the checks inspect.

## Provenance and license

- Upstream: 710leo/urlooker (see its LICENSE in the workspace).
- The added `recovery/` material, the checks and the reference script may be released under
  this repository's license (MIT).
- This task has been served and retired; it is not in the task pool.
