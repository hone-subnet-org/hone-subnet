# V3 task execution

V3 supports `repository_patch_v1` and `terminal_script_v1` tasks. It is a breaking protocol: every request and response uses protocol version 3.

The validator leases an immutable workspace and upload slots, verifies the workspace before contacting miners, and sends each server-assigned miner its task and two write-once slots. A miner uploads its submission and trajectory, then signs references to those exact objects. The validator commits the signed response set before receiving the verifier URL.

Repository submissions are UTF-8 Git unified diffs. Terminal submissions are UTF-8 Bash scripts. Each accepted submission is graded in a clean copy of the leased workspace with network access disabled. A passing submission must satisfy every verifier check; all other candidate outcomes receive zero.

Trajectories use the strict `trajectory_v1` JCS JSON schema and bind the recorded model and tool events to the uploaded submission hash.
V3 does not write the legacy local rollout shards.

The reference miner requires a provider response containing reasoning text, chosen-token log probabilities, and exactly five alternatives per token.

Artifact hashes, compressed sizes, and decompressed tar-stream sizes are checked exactly. Archives reject links, special files, duplicate paths, path traversal, and decompression beyond the advertised limit. Artifact URLs must use an HTTPS origin pinned by the release.

Candidate containers have fixed CPU, memory, process, time, output, temporary-directory, and single-file limits. The validator also checks available disk space before materializing each workspace, enforces a total-tree limit after candidate execution, and removes each miner workspace immediately after grading. The total workspace limit is advisory on ordinary Linux filesystems because portable non-root per-directory disk quotas are unavailable.

Whole-round infrastructure or protocol failures do not update scores. Miner-specific upload, patch, build, test, timeout, or output failures affect only that miner.

After a completed round, the validator reports binary verdicts and grading durations for submission grants. This feedback is diagnostic and does not affect scores or weights.

## Rollout

1. Publish the pinned execution image and storage origin in release policy.
2. Run the shared synthetic acceptance round.
3. Pause V2 issuance and wait for its leases and grading windows to expire.
4. Deploy the V3 server, validators, and miners together.
5. Verify a Python repository task before enabling additional task inventory.
