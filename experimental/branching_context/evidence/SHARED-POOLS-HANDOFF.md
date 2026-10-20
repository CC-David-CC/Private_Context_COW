# Shared pools: current implementation

Local checkout: `C:\Users\dflanag3\Documents\fleet\context-cow-execution-parity`.
Branch: `feat/shared-context-pools`. Private remote: `context-cow-private`.
No GitHub/upstream publication. Chat server stays off.

The user resumed work to implement shared RAM/disk pools and an explicit GPU
worker list. The earlier pause and the old P4-first priority are superseded.

## Implementation

- `pools.py`: normalized disjoint GPU groups, byte reservations, coalesced
  read-ahead, clean LRU eviction, dirty/borrowed pins and disk retention/GC.
- `coordinator.py`: one native-owner thread per selected worker, ready FIFO,
  same-branch publication exclusion, CPU-only background writer, two events,
  queued-source leases, retry intents and recovery from canonical receipts.
- `real.py`: immutable prepared transport, streaming publication and actual-file
  hash validation before branch commitment. Existing checkpoint format retained.
- `strata_backend.py`: per-child pinned environment and explicit device routing.
  Split-worker native capture/restore is rejected, not claimed implemented.

`GPU=[0]` is the llm-60 configuration. `[1,5,6]` means three independent workers.
`[[0,1,2]]` means one split worker. Mixed groups parse and schedule in CPU tests;
the native split adapter remains an integration gap.

## Executed evidence so far

82 CPU/filesystem control/recovery tests passed, no skips/xfails. Linux G0 passed
51/51. Executed IDs/source hashes are in `shared-pools-20261014/control-v5.json`.

Final frozen source on llm-60:
`/home/dflanag3/branching-g3/shared-pools-20261014/api-v5`.
GPU matrix:
`/home/dflanag3/branching-g3/shared-pools-20261014/matrix-v5`.

All four final matrix cases passed eight independent matching-reference tree
comparisons each, with zero error. Saved-prefix/controller/history audits also
passed. Read `matrix-summary.json` in the local shared-pools-20261014 evidence
directory. Observed case durations were 211.32/247.88/803.92/898.42 seconds;
these include independent reference runs and are not serving latency.

Q4 actual outputs: tiny 55 total, seven trees continued after restart; 8K 642,
six trees continued. Other trees had already ended at EOS. Q8 tiny produced 54
outputs, with seven continued trees. Q8 8K produced 902 outputs with all eight
trees continued. These are limits, not padded output lengths.

Real clean RAM evictions: Q4 tiny 10, Q4 8K 22, Q8 tiny 16, Q8 8K 29. Q4 uses a deliberately
tight 2 GiB test budget; Q8 uses 8 GiB, including conservative capture staging.
These are charged extents, not whole-process memory peaks.

## Failures retained

- First GPU harness tried to resume a controller already stopped at EOS. Failed
  campaign retained at `q4-tiny`; fixed harness reports completed turns separately.
- Initial crash witnesses lacked inherited PYTHONPATH; the runner was corrected,
  with original failure logs retained. Existing tests/assertions were unchanged.
- Red tests exposed missing queue leases, uncharged upload staging and missing
  validation of a newly written payload before CAS. All now have passing tests.
- Early grouped-routing negative test reached missing fixture environment before
  the unsupported-protocol guard; guard now rejects before device initialization.

## Resume/next work

1. The identified matrix completed successfully; producer/consumer processes
   exited. No automatic chat restart. Preserve frozen api-v5 for reproduction.
2. `summarize_pool_matrix.py` completed prefix-frontier, controller/decoder/history
   checks without inference and produced compact evidence.
3. Code and compact evidence were committed as `3e799117` and pushed to private
   box-git branch `feat/shared-context-pools`. Raw results were copied directly
   llm-60 to `/srv/data/git/private_context_cow.evidence/20261014-shared-pools`
   on box-git. All 3,050 regular files passed source SHA-256 verification; see
   `shared-pools-20261014/backup.json`. GitHub remains untouched.
4. Exercise three real codebases and useful fork/rewind/adopt traces. Current
   numeric fixtures are synthetic tokens; ordinary serving remains separate.
5. Add model-free CUDA tests on A100 GPUs 1/5/6 when its SSH identity is available.
   Transfer no Qwen weights. Compile for its architecture, or use inspected driver APIs.
6. Implement/qualify composite device-tagged split checkpoints before enabling
   the native split worker, including independent profiles/placement compatibility.

Original gates remain separate. Current greedy matching-operator results do not
resolve the existing cross-boundary mismatch, Windows `fcntl`, stochastic/grammar
GPU coverage, full crash/OOM campaigns, sanitizer/soak or general serving integration.
