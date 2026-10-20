# Native-speed continuation: repeated 64K client measurements

Goal remains ACTIVE, not achieved. The preceding disconnect/status-only turn
made no goal progress; this turn inspected current source, frozen evidence and
actual processes, then launched the next missing qualification measurement.
No A100 connection or inference work is part of this resumption.

Pinned candidate: llm-60 `native-request-probe-20261014/api-digest-v1`, Q8 profile
`2cd8857a454d5645e8caf5a0551213bcfbf99ee3b57622ef738f60b62f4f25ee`.
Reference case: `digest-reuse-v1-20261014/q8-real-64k-restart`.
The accepted contract is greedy identical-native-profile continuation. Hashes
remain enabled. Original G1-G5 requirements remain separate.

Actual GPU preflight: RTX PRO 6000 Blackwell, 15 MiB used, 97,235 MiB free,
0% utilization; no existing inference process. The chat server stays off.

New run root on llm-60:
`/home/dflanag3/branching-g3/native-client-resume-20261014-r3`.
tmux session `cow-native-client-r3`; runner PID 1648911; resident client PID
1648925. These were observed live after launch, not inferred from PID files.
The candidate/API was not edited. One GPU job executes at a time.

Resident runs first, then disk, one warm-up pair plus three measured A/B pairs
per mode. Each request uses a saved 65,536-step prefix, an 8,335-step question
and 2,048 output bound from the already qualified real-model fixture. The client
records TTFT, received token gaps/TPOT, logical completion and durable completion
separately. Tokens/controller/history/final frontier must match the retained
reference, and native counters must exclude prefix recomputation. Startup and
disk restoration are reported separately. Partial streamed output is not a
committed snapshot or completed measurement.

Wrapper source is retained locally under `evidence/runs/native-client-resume-r3`
and remotely at `branching-g3/native-client-resume-r3.sh`. It uses an exclusive
output-specific GPU-run lock, fresh directories, stdout/stderr and exit receipts,
and 15-second GPU/liveness observations. It does not start or restart serving.
No completed result is claimed by this launch note.

First completed measured resident pair (not the complete campaign): A-1 and
B-1 each consumed 2,048 outputs and passed reference token/controller/history/
final-frontier validation through durable acknowledgment. Received steady TPS:
51.681 / 41.521; median TPOT: 18.286 / 19.899 ms; p95 TPOT: 26.045 / 41.022 ms.
TTFT: 3.440 / 3.241 seconds, including the 8,335-step new question and resident
activation. Logical request TPS: 47.304 / 38.798; durable request TPS:
44.350 / 36.787. Startup is excluded. These are individual measurements,
not three-repetition medians or a completed resident/disk qualification.

The first warm-up pair had about 11-12 seconds between logical and durable
completion; the first measured pair had about 2.9 seconds. Repeated deterministic
completions can reuse an already present payload, so warm-up/fresh-file cost
must stay visible rather than treating deduplicated durable timing as the cost
of an arbitrary new answer. This is a measurement scope, not a proposed
relaxation of publication or integrity guarantees.

Resident phase completed PASS: client/worker PASS, native exit observed, one
warm-up pair and six measured requests (three A/B pairs), all 2,048 outputs.
Every request evaluated exactly 10,383 new suffix/output tokens and loaded zero
file bytes during the timed resident request. All retained-reference checks
passed and the parent payload audit passed. Model startup was 66.733 seconds.

Three-repetition medians, A / B:

- Received steady TPS: 51.681 / 41.521.
- TTFT: 3.390 / 3.242 seconds.
- Median TPOT: 18.286 / 19.899 ms.
- Median per-request p95 TPOT: 26.207 / 40.967 ms.
- Logical request TPS: 47.304 / 38.798.
- Durable request TPS: 44.350 / 36.787.

The runner then started disk client PID 1651820 under the same session. Its
results remain outstanding. Resident PASS does not imply disk PASS, native-speed
goal completion, HTTP serving integration, or original cumulative certification.
Compact/raw client and worker records are retained in the ignored run directory.

Observe the recorded process/session and the same `resident/client.json`,
`resident/worker.json`, events log and phase/exit receipts. Do not relaunch on an
observation timeout. The next action after terminal results is to attribute
client and durable latency before choosing another implementation change.

## Follow-up observation and report collector

Disk client PID 1651820 and worker PID 1651831 were re-observed live at
6m52s elapsed; wrapper PID 1648911 remained live. Completed A-0/B-0 warm-ups
and A-1/B-1/A-2 were durably acknowledged. Measured A-1/B-1 received steady
TPS was 52.000/41.464; TTFT was 7.933/7.988 seconds. A-2 received 52.069 TPS
with 7.544-second TTFT. These are individual results, not final medians.
Disk restore receipts report zero forward tokens through restoration. Disk
phase stderr was empty at the observation. The phase is still RUNNING.

Added `scripts/summarize_client_campaign.py` to derive compact medians with
client/worker hash binding, exact request identity matching, durable completion,
native exit and zero-forward disk-restore receipt checks. Artifact-only execution
with local Python 3.13.15 accepted the completed resident records (exit 0) and
rejected partial disk records (exit 1, require completed records/native exit).
No model computation, build or GPU test was executed on Windows. The collector
does not independently establish numerical correctness beyond the pinned worker's
executed assertions, and does not qualify original gates.

Read-only native review confirmed that cross-pool hashing is already batched
into one digest plan/launch before COW writes. The existing corrupt-last-pool
regression asserts unchanged ownership/COW/token state on failure. Removing a
possibly redundant synchronization before digest download remains an unmeasured
candidate; no frozen source/build was modified. Prior 512 MiB stage measurements
cannot substitute for a real 64K request breakdown.

Next remaining workload milestone is switching among distinct useful source-code
contexts under RAM pressure, then ordinary serving integration. Repeated A/B
latency alone does not prove those requirements. A100 remains disconnected;
llm-60 chat stays off. All changes/results remain local/private box-git.

## Terminal disk result

The same runner reached COMPLETE with exit 0. Wrapper/client/worker PIDs were
absent in the terminal process observation. Disk client and worker both report
PASS and native exit observed; the completed report collector accepted their
hash-bound records. Eight requests each produced 2,048 outputs, including
three measured A/B pairs. Each request evaluated 10,383 new tokens; every
explicit disk restore reports zero prefix-forward tokens. All worker token,
controller, history, final-frontier and parent-payload integrity checks passed.

Disk medians A/B: steady received TPS 52.000/41.466; TTFT 7.617/7.974 seconds;
median TPOT 18.148/19.923 ms; per-request p95 TPOT 26.070/41.168 ms;
logical request TPS 43.133/35.588; durable request TPS 40.666/33.933.
Startup was 70.621 seconds, excluded from request timing. Disk used warm
filesystem cache. Full raw records remain under the exact run root; compact
machine-readable evidence is `NATIVE-CLIENT-RESULTS-2026-10-14.json`.

This proves the bounded repeated client workload, not ordinary HTTP serving,
arbitrary new-answer publication latency, multiple useful codebase switching,
stochastic controllers, or cumulative gates. No inference is intentionally
left running by this campaign; chat stays off. Goal remains active.
