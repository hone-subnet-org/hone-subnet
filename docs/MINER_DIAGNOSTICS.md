# Local evaluation diagnostics

Validators write one round outcome and one evaluation record for each assigned
miner to `v3_evaluations.jsonl`, beside the configured score-state file. When a
miner's check runs and fails, a failure receipt may follow that miner's
evaluation record. Each line is a JSON object. These records are local to the validator;
they do not change feedback sent to the problem server.

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
most 4 KiB; an optional reason is omitted if necessary to meet that cap, and a
failure receipt that cannot fit is skipped. Readers should skip a malformed final line
after an interrupted write. The next write repairs an incomplete active-file
tail. If external corruption leaves more than 4 KiB without a newline at the
end, writes fail until the operator moves the damaged file aside.
Run only one validator writer for each log path.

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

Round outcome and miner evaluation records do not include signed URLs, response
headers, miner response bodies, check output, expected answers, or verifier
code. Failure receipts, described below, do include the failing check's command,
input, and expected and actual output excerpts. Sending receipts to miners
requires a separate server-coordinated contract.

## Failure receipts

A `failure_receipt` record describes the first check that ran and failed.
Grading stops there, so later checks never run and are not identified. A
receipt can be written when the check's exit code, stdout, or stderr did not
match, or when the check reached a time, memory, or output limit under the
existing grading rules. `output_source` identifies whose output it shows.
Setup, script, patch, result-tree, and working-directory failures have no
receipt, and neither do validator faults.

The record carries the round and miner identity fields and a `receipt` object:

- `check_index`, `checks_total`, `check_id`: the failing check's position and name.
- `kind` and `output_source`: an `invocation` check runs the candidate, so its
  output is `candidate`. An `inspection` check runs the verifier's checker
  against the result, so its output is `checker`.
- `argv`, `cwd`: the command and its container working directory. Long values
  are shortened and marked by `argv_truncated` or `cwd_truncated`.
- `limit`: `timeout`, `memory_limit`, or `output_limit`; `null` for a mismatch.
- `mismatched`: which of `exit_code`, `stdout`, and `stderr` differed. It is
  empty when `limit` is set.
- `exit_code`: the expected and actual exit codes. The expected code is `null`
  when `limit` is set.
- `stdin`: the start of an invocation check's input; `null` for inspections.
- `stdout`, `stderr`: `first_difference` is the first differing byte offset, or
  `null` when the streams matched or were not compared. `expected` and `actual`
  are excerpts around that offset. Unchecked stderr and limit failures show the
  end of the captured output, with no expected excerpt.

Each excerpt has `offset`, base64 `data_b64`, `captured_bytes`, `total_bytes`,
`capture_truncated`, and `truncated`. The bytes are exact, including line
endings, trailing spaces, and invalid UTF-8. `capture_truncated` means the
validator retained output only up to the check's output limit and discarded the
rest: the real output was longer, and `total_bytes` is `null`. `truncated` is
true if captured bytes were omitted from the excerpt or `capture_truncated` is
true.

A receipt is at most 3 KiB. Excerpts are shortened in a fixed order to fit, and
a receipt that still does not fit is omitted. A receipt that cannot be
converted, serialized, or fit on one log line is skipped without affecting
other records, and the write reports a warning. A storage failure can still
stop the remaining writes for a round, as described above.

Output after a timeout or memory limit is only what was captured before the
process stopped. Running the shown command by itself may not reproduce the
failure, because setup and earlier checks run first in the same workspace.
Validators already hold the verifier, so receipts show operators nothing new.

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
