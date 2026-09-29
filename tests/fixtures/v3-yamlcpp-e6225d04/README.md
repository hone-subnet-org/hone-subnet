# V3 test fixture: yaml-cpp, two planted defects (task e6225d04)

One real `repository_patch_v1` task from the problem server, packaged as a test fixture. It has
been leased, solved by the reference miner and graded end to end by the V3 validator.

## Files

| File | What it is |
|---|---|
| `identity.json` | the task identity exactly as leased (hashes to `task_id.txt`) |
| `task_id.txt` | `e6225d044a252b86849bb14dc2f42ee399351dd3e9ded900712e4cdf1d6fe5c3` |
| `artifact_refs.json` | sha256 and sizes of the two archives |
| `workspace.tar.zst` | the workspace a miner receives (sha256 `bdb4aafc…34f52`, 873 KiB, 7.3 MB expanded) |
| `verifier.tar.zst` | the hidden checks: `manifest.json`, 6 scripts in `checks/`, expected output in `gold/` (sha256 `597b92f6…4acd`) |
| `reference.diff` | a correct fix (unified diff, applies to the workspace root) |

Profile `repo-polyglot-v1`, verifier policy `command-gold-digest-v1`, language C++.

## Expected results

- **Unpatched workspace:** fails (the first check fails).
- **`reference.diff` applied:** passes all 6 checks, stable across runs.
- **Reference miner (Kimi K3):** results vary run to run on this task,
  e.g. one run failed `00-d1-alias-binary-pick` (checks 2-6 skipped), another passed 6/6.
  Solve about 2-4 minutes; grading about 8 s.

## Task

The instruction is deliberately vague ("We've had reports that EventArchive sometimes gives
incorrect results..."). Two defects are planted in a private `EventArchive` feature added to
yaml-cpp: `d1` (checks 00-02) and `d2` (checks 03-05). The reference diff fixes both.

## Suggested tests

1. **Fast (no Docker):** `identity.json` hashes to the task ID; both archives match
   `artifact_refs.json` (sha256, sizes); archives pass the validator's archive limits; the
   manifest parses.
2. **Integration (Docker, pinned sandbox image):** grade an empty submission (expect fail on
   check 00), grade `reference.diff` (expect pass, 6/6), grade a diff that fixes only `d1`
   (expect fail on check 03).

## Provenance and license

- Upstream: jbeder/yaml-cpp at `f7320141120f720aecc4c32be25586e7da9eb978` (MIT; the license
  file is included in the workspace).
- The added `EventArchive` feature, the planted defects, the checks and the reference diff may be
  released under this repository's license (MIT).
- This task is not in the mainnet task pool.
