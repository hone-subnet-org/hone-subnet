# V3 task execution

V3 supports `repository_patch_v1` and `terminal_script_v1` tasks. It is a breaking protocol: every request and response uses protocol version 3.

The validator chooses which miners a task is offered to: a random subset of the serving miners, never the subnet owner's hotkey when the chain names one, and never a UID that is validating (a validator permit with non-zero validator trust), at most 256, named in the lease request. The server issues upload slots for exactly the first N miners of that list, N being fixed by the validator release policy (128), and the validator abandons a lease whose pool is anything else, so the server can neither add, skip, substitute nor resize. The signed-response quorum is likewise fixed by release policy (4) and a lease naming another value is abandoned, as is a lease that leaves miners less than the released minimum window (10 minutes). The offered order is kept until a round completes or fails its quorum, so a rejected or otherwise abandoned lease is retried with the same order, refreshed only for miners that stopped or started serving. A miner the chain says is eligible but the server keeps refusing therefore stalls that validator's rounds loudly rather than quietly reshuffling around it, which is deliberate: it is a server fault to fix, not a draw to redo. The validator then verifies the workspace before contacting miners, and sends each miner in the pool its task and two write-once slots. A miner uploads its submission and trajectory, then signs references to those exact objects. The validator commits the signed response set before receiving the verifier URL.

Repository submissions are UTF-8 Git unified diffs. Terminal submissions are UTF-8 Bash scripts. Each accepted submission is graded in a clean copy of the leased workspace with network access disabled. A passing submission must satisfy every verifier check; all other candidate outcomes receive zero.

An optional manifest setup step runs after applying a repository patch, or before running a terminal script. Terminal setup uses the result directory as its working directory. Result paths are checked again after candidate execution, including every ancestor of a nested result directory.

Trajectories use the strict `trajectory_v1` JCS JSON schema and bind the recorded model and tool events to the uploaded submission hash.
V3 does not write the legacy local rollout shards.

The reference miner records reasoning text and chosen-token log probabilities with five alternatives per token when the provider returns them. Neither is required.

Artifact hashes, compressed sizes, and decompressed tar-stream sizes are checked exactly. Archives reject links, special files, duplicate paths, path traversal, and decompression beyond the advertised limit. Artifact URLs must use an HTTPS origin pinned by the release.

Candidate containers have fixed CPU, memory, process, time, output, temporary-directory, and single-file limits. The validator also checks available disk space before materializing each workspace, enforces a total-tree limit after candidate execution, and removes each miner workspace immediately after grading. The total workspace limit is advisory on ordinary Linux filesystems because portable non-root per-directory disk quotas are unavailable.

Whole-round infrastructure or protocol failures do not update scores. Miner-specific upload, patch, build, test, timeout, or output failures affect only that miner.
This includes timeout, memory, and output limits reached by a trusted inspection while processing a candidate's result. Missing trusted executables and container control failures remain infrastructure failures. This classification adds no container runs.

After a completed round, the validator reports one verdict per submission grant: pass or fail, the grading duration, the miner's response latency (dispatch to signed response) and, for a failed verdict, the miner reason code. Separately, the validator posts the round for the shared ledger: numbered, signed by its hotkey over the challenge, task, sequence number and every miner the lease named with its verdict and latency, and kept on disk until the server has it, so other validators can pool it and a missing round shows as a gap. The feedback is diagnostic and does not affect scores or weights; the ledger does: each validator reads the others' signed rounds back, admits those whose signature verifies and whose signer is validating with at least 0.75% of stake, counts every admitted round equally with at most 50 rounds per validator per miner in a four-day window, and scores each miner on the pooled window with the same speed factor it applies locally.
Validators also retain bounded [local evaluation records](MINER_DIAGNOSTICS.md), including failure codes and abandoned-round outcomes. These records do not change the feedback wire contract.

## Rollout

1. Publish the pinned execution image and storage origin in release policy.
2. Run the shared synthetic acceptance round.
3. Pause V2 issuance and wait for its leases and grading windows to expire.
4. Deploy the V3 server, validators, and miners together.
5. Verify a Python repository task before enabling additional task inventory.
