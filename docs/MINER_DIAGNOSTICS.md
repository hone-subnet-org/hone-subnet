# Local evaluation diagnostics

Validators write one round outcome and one evaluation record for each assigned
miner to `v3_evaluations.jsonl`, beside the configured score-state file. Each
line is a JSON object. These records are local to the
validator; they are never sent anywhere. Separately, and on by default, a
validator sends each failed miner a short notice about its own failure, straight
to that miner. See "Failure notices sent to miners" below.

```bash
tail -f data/v3_evaluations.jsonl
```

Set `VALIDATOR_DIAGNOSTICS_FILE` to choose another path, or set it to an empty
string to disable recording. The default retains the active file and three
backups, each up to 8 MiB. Retention is by bytes, not by time or round count.
New files use mode `0600`. A storage or logging failure produces a warning
and does not change scores or interrupt the validator.

Every record identifies the validator hotkey, challenge ID, and task ID.
Miner records also identify the miner hotkey and the UID assigned for that
challenge. Ownership is identified by hotkey; UIDs may later be reassigned.
The stable `record_id` is independent of the timestamp. A consumer combining
records or replayed logs should deduplicate by that ID and retain the latest
record. There is no persistent local deduplication index.
Writes are best effort: a storage failure can leave only some records for a
round. A missing miner record is not evidence of success, failure, or score credit.

The log uses `schema_version: 1`. Consumers should tolerate additional reason
codes and skip record types they do not recognize. Each serialized line is at
most 4 KiB; to meet that cap the optional `failed_check` is dropped first, then
the optional reason. An interrupted write can leave a malformed line anywhere in
the file, since the next write ends it and carries on; readers should skip any
line that does not parse. Run only one validator writer for each log path.

## Reading an outcome

- `round_status`: `completed`, or `abandoned` when the round did not produce
  scoreable results. A failed lease has no assigned task and creates no records.
- `status`: the miner's evaluation was `passed`, `failed`, `rejected`, or
  `abandoned`; `not_evaluated` means it did not reach evaluation.
- `reason_code` and `stage`: identify the result and where it occurred. The
  bounded `reason` adds an explanation, such as a static patch rejection.
- `dispatch_reason_code`: records a local dispatch problem independently of
  the server's artifact verdict. For example, `not_serving` can accompany
  a server verdict of `artifact_invalid`.
- `checks_passed`, `checks_executed`, `checks_total`, `checks_skipped`: count
  passed outcomes, passed or failed outcomes, all manifest checks, and remaining
  checks. Totals are `null` until the verifier manifest is loaded. These count
  recorded outcomes, not container launches: an infrastructure interruption
  before an outcome is recorded can leave that check marked skipped.
  Early stopping is unchanged.
- `grading_duration_ms` and `response_latency_ms`: describe grading and response
  time when available. They are `null` for miners without an evaluation record.
- `failed_check`: the six-line description of the failing check that was
  available for this evaluation, when one could be rendered; otherwise `null`. It
  states what the check required, never what the miner printed. It is not proof
  the miner received it: details may be switched off, delivery is best effort,
  and a partial evaluation from an abandoned round is recorded here too.
- `score_effect`: `unchanged` for an abandoned round, or `not_reported` for a
  completed round. These are evaluation records, not proof of a credited reward
  or persisted observation for an individual miner. Registration eligibility
  and scoring remain separate decisions.

A miner can have `status: passed` with `round_status: abandoned`: its checks
passed, but a later infrastructure failure prevented the whole round from
updating scores. Partial outcomes are retained only for diagnostics. Records
are written after the callback has attempted normal score persistence.

## Miner and submission reason codes

| Code | Meaning |
| --- | --- |
| `not_serving` | No matching serving miner client was available. |
| `dispatch_failed` | The solver raised during dispatch. |
| `response_unavailable` | No validated signed response was returned. This can include transport, signature, or response-format failure; it does not prove miner fault. |
| `slot_mismatch` | The server reported an upload-slot mismatch. |
| `artifact_invalid` | The server reported an invalid submission artifact. |
| `trajectory_invalid` | The server reported an invalid trajectory. |
| `patch_static_rejected` | Public static patch validation rejected the bytes, format, or mode. |
| `patch_apply_failed` | Git rejected the patch or could not apply it after its check. |
| `patch_rejected` | A patch rejection without a more specific local code. |
| `script_rejected` | Public script validation rejected its bytes or size. |
| `working_directory_invalid` | Candidate changes removed or invalidated a required directory. |
| `setup_failed` | Setup exited unsuccessfully. |
| `check_failed` | A check's output or exit status did not match its expectation. |
| `timeout` | Candidate execution or an inspection reached its time limit. |
| `memory_limit` | A container was killed for exceeding its memory limit. |
| `output_limit` | A container exceeded a stdout or stderr limit. |
| `result_tree_invalid` | The result contains an unsafe path/entry or exceeds tree limits. |

Dispatch codes describe local observations, not a new fault-attribution policy.
For example, a local signing or transport-start failure can currently result
in no signed response. The diagnostic code does not change how that outcome
is handled by the existing round and server logic.

## Round and infrastructure reason codes

| Code | Meaning |
| --- | --- |
| `lease_unavailable` | No usable challenge was leased; there is no per-miner log record. |
| `unsupported_profile`, `unsupported_verifier_policy`, `unsupported_task` | The leased contract is unsupported. |
| `insufficient_storage` | A free-space check failed. |
| `quorum_not_met` | Too few signed responses were available to commit. |
| `commit_failed`, `reveal_mismatch` | Commit/reveal failed or did not match the lease. |
| `grading_expired` | The grading window expired. |
| `submission_format_mismatch`, `submission_grant_mismatch` | A grant disagreed with the task or signed response. |
| `submission_download_failed` | A committed submission could not be retrieved or verified. |
| `workspace_invalid` | A required baseline workspace directory was unavailable. |
| `workspace_download_failed`, `workspace_extraction_failed`, `workspace_materialization_failed` | Workspace preparation failed at the named stage. |
| `patch_tool_failed`, `setup_unavailable`, `script_unavailable`, `verifier_unavailable` | The validator could not run the required tool or container. |
| `verifier_download_failed`, `verifier_invalid` | The verifier could not be retrieved, extracted, or loaded. |
| `cleanup_failed` | Managed workspace cleanup failed. |
| `validator_error` | Another validator failure occurred; `stage` identifies the operation. |

The feedback a validator sends the problem server is unchanged: it still carries
only uid, hotkey, pass or fail and a grading duration, byte for byte as before,
and the old VALIDATOR_FAILURE_EXPLANATIONS setting no longer exists.

Round outcome and miner evaluation records do not include signed URLs, response
headers, miner response bodies, the miner's own check output, or verifier code.
A miner evaluation's `failed_check` does state the expected output for the
failing check, exactly as the miner is sent it. The log does not record what the
miner's program printed or where its output first differed; an operator who
needs that has to reproduce the check. Nothing in this log is sent to the
problem server or to a miner.

## Failure notices sent to miners

When a graded round finishes, the validator sends one small signed message to each
miner whose submission failed or was rejected, straight to that miner's axon. There
is no problem server involved and nothing is stored centrally. Two settings control
it, both on by default:

- `VALIDATOR_FAILURE_NOTICES` sends the notices at all.
- `VALIDATOR_FAILED_CHECK_DETAILS` adds the failing check's command and expected
  output. With it off, a miner gets the reason code alone.

A notice is `POST {miner axon}/v3/failure`, signed the same way a task is, and
carries the protocol version, the fixed message type `failure_notice_v1`, the
challenge and task ids, the recipient's own uid and hotkey, and a `failure` object
holding `version`, `reason_code` and an optional `failed_check` display.

The reason codes are the miner and submission codes tabled above. A round or
infrastructure cause, or a missing code, is reported as the generic
`evaluation_failed`. The `failed_check` display is only ever attached to
`check_failed` at the check stage, and it is the same six-line text described
below. It never contains the miner's own output, the check id, its position, the
number of checks, or any verifier code.

Delivery is deliberately cheap and forgettable. One attempt per miner, no retries
and no redirects, at most 8 in flight, 2 seconds for one exchange and 5 seconds for
the whole batch, at most 1024 recipients. An old miner without the route, an
offline miner, or a slow one simply gets nothing. Nothing here can change a grade, a
score or a weight, and the round is already finished when it runs. A notice is sent
only for a completed round, only to a registration the round actually assigned and
graded, and only when the score file was written successfully: an operator who
disables score persistence receives no notices either. The cost is bounded but not
zero, so a round can take up to five seconds longer when many miners are
unreachable.

### What a miner does with one

The reference miner adds the route and prints what it receives, for example:

```
[demo-miner] feedback: check_failed challenge chal-8f21 task 9c4f000000000000
[demo-miner] feedback:   Command (argv): ["/usr/bin/python3","main.py"]
[demo-miner] feedback:   Working directory: "/work"
[demo-miner] feedback:   Stdin: ""
[demo-miner] feedback:   Required exit code: 0
[demo-miner] feedback:   Required stdout: "red-fox\n"
[demo-miner] feedback:   Required stderr: not checked
```

Nothing is written to disk. Operators who want history should capture the miner's
stdout, which most process managers do; retention is then whatever that environment
keeps.

A miner implementing its own receiver should do what the reference does. Refuse
anything it cannot verify: without a wallet identity or a metagraph view it answers
403 rather than trusting the sender. Check the signature, that the message was
signed for this miner, and that the signer is a validator under the same policy the
miner already applies to tasks, which honours its own stake and permit settings.
Refuse a replayed request. Refuse a notice addressed to another miner. Then check
that the notice matches a task this miner actually answered for that validator, with
the same uid and hotkey: a valid signature proves who sent it, not that they graded
you. That memory holds 256 tasks for two hours and is lost on restart, so a notice
about an older task is refused. A repeat for the same task is accepted and ignored.
The reply is a fixed `{"accepted": true}`, which is an acknowledgement and no more.

Treat the text as untrusted. The reference miner escapes everything that is not
printable ASCII rather than deleting it, prints the reason before any identifier so
a long id cannot hide it, never truncates the display, and drops it entirely with a
short note if it is too large to print. Output runs off the event loop, at most four
at once, and excess is refused immediately rather than queued, so notices cannot
interfere with solving.

### What the display can cover

The display describes the FIRST check that failed, in the order the manifest
already runs them. It is not the shortest or a minimized case, and finding it
costs no extra grading runs. Later checks never ran and are never named.

Only a narrow shape is rendered at all:

- an invocation check whose command is exactly an allowlisted Python executable,
  `/usr/bin/python3` or `/usr/local/bin/python3`, followed by one `.py` script
  path that resolves inside `/work`;
- no interpreter flags, no script arguments, no shell, no inline code;
- the stdin and expected streams must decode as UTF-8, and the whole display must
  fit 2048 bytes measured as an escaped JSON string.

Anything else gets the reason code and no display: inspection checks that run the
verifier's own checker, suites, other interpreters, compiled languages, build
steps, and any check whose data is too large or not UTF-8. Compiler and build
output are never forwarded, and an existing setup or execution failure is never
relabelled as a compilation failure.

A notice is at most 8192 bytes on the wire, and a miner should refuse anything
larger.

The display states what the check required. It is not a reproduction recipe:
setup and any earlier checks run first in the same workspace, so running the
shown command alone can behave differently.

### Before enabling the display

`VALIDATOR_FAILED_CHECK_DETAILS` shows the expected answer for the failing case.
That is safe only while a task is never dispatched more than once, which is a
guarantee from task generation and not something this repository enforces. Turn the
setting off BEFORE any change that allows a task to be reused. Turning it off later
cannot unsay what was already shown. Reason-only notices carry no such assumption.

## Scoring interpretation

An eligible passing miner gets a contribution between the release speed floor
and 1, relative to the fastest positive response time among eligible passers.
The current floor is 0.95 and the speed half-life is 180 seconds. Failed or
rejected evaluations contribute zero. Abandoned rounds add no observations.

The score window retains up to 200 recorded observations, with a startup
denominator of at least four. This is not necessarily 200 wall-clock rounds:
serving miners not sampled for a task do not receive an observation, while
previously observed nonserving miners receive zero on completed rounds when
nonresponder decay is enabled. A hotkey change resets the old registration's
history. The release policy controls these values, independently of diagnostics.
