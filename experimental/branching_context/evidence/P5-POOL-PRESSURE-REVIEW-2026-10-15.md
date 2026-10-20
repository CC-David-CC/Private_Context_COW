# P5 pool-pressure review — 2026-10-15

Reviewed frozen source/evidence at `ac6b8e6d`; no source, profile, server, or GPU
job was changed or launched for this review.

## What P5 already has

The required *workload* exists. The real Q4 fixture at
`multi-code-contexts-20261014/fixtures/q4_64k.json` contains three distinct
65,536-token source prefixes from actual project source, with 8,311-8,336-token
questions and a 2,048-output limit. Its recorded fixture SHA-256 is
`9fb32731440a794234eae1f622b34415d6a80c433df79f30aea6c37d0f9cb5c2`. Q4 and
Q8 full campaigns each ran six A/B trees and three rewinds against matching
fresh references, with zero error at 15 sampled full-vector frontiers. They
used native capacity 98,304 and 8,192-token prefill chunks. Actual outputs can
end early at EOS (Q4 included a 507-token tree; Q8 included 996 and 2,011), so
the limit must not be reported as achieved output length. See
`MULTI-CODE-CONTEXTS-2026-10-14.md:151-225`,
`SOURCE-CONTEXT-Q4-64K-2026-10-14.json`, and
`SOURCE-CONTEXT-Q8-64K-2026-10-14.json`.

The long Q4 run already measured one active request per GPU (`max_active_per_gpu
{"0":1}`), native capture/import/export and COW counters, and a real restart
that imported about 1.83-2.05 GB per selected checkpoint with zero saved-prefix
forward tokens. Its pool also recorded substantial durable state: at resume it
had 30,347,645,857 accounted disk bytes. GPU telemetry reached 90,747 MiB for
Q4 and 92,637 MiB for Q8, but those figures include model residency and do not
isolate context bytes. These prove full-workload execution, serial admission,
and process-separated disk restoration; they do not show pressure-driven RAM
eviction.

The missing coverage is specific: both full-source campaigns used 96 GiB RAM
and 128 GiB disk budgets. Q4's measured RAM peak was 30,890,651,180 bytes, with
zero clean evictions; its disk pool stayed below its 128 GiB limit. Thus neither
campaign forces budget-driven spill/return. Existing three-context pool
pressure passes are 8K-or-smaller synthetic token fixtures, with Q4's 2 GiB
pressure cap and Q8's 8 GiB cap. They have clean evictions, RAM-only switches,
and disk fallback, but they are not the three real code contexts at full
capacity. See `SHARED-POOLS-HANDOFF.md:36-48` and `STATUS.md:1311-1313`.

## Smallest existing-runner Q4 experiment

Use the existing `pool_campaign.py campaign` path with the real Q4 fixture,
not `pool_matrix.py` (which hard-codes the four synthetic tiny/8K cases). Keep
the frozen Q4 digest-v1 model/build and 96K profile; change only pool budgets.
The candidate config is:

```json
{
  "GPU": [0],
  "ram_bytes": 17179869184,
  "disk_bytes": 68719476736,
  "min_available_bytes": 17179869184,
  "profile": "/home/dflanag3/branching-g3/native-request-probe-20261014/q4-profile-digest-v1/profile.json",
  "require_clean_evictions": true
}
```

Before launch, verify the selected frozen profile resolves to capacity 98,304,
8,192-token chunks, FP16 KV, the Q4 weights/build, and the matching unbound
oracle command. The 16 GiB RAM cap is deliberately below the full campaign's
30.89 GB measured pool peak, but above the observed per-checkpoint import
(1.83-2.05 GB) plus the coordinator's conservative three-checkpoint-sized
transfer reservation (roughly 5.5–6.2 GB). That leaves room for one cold
transition while requiring the multi-context working set to evict. The 64 GiB
disk cap is above the completed full campaign's 30.35 GB maximum accounted
files and below the previous 128 GiB setting. It should retain the full A/B and
rewind history without treating arbitrary free disk as the pool budget.

Run it as a new output directory with the existing runner and exact fixture:

```sh
python3 experimental/branching_context/scripts/pool_campaign.py campaign \
  --config /home/dflanag3/branching-g3/<new-run>/p5-q4-96k-ram16g-disk64g.json \
  --output /home/dflanag3/branching-g3/<new-run>/q4-64k-pressure \
  --fixture /home/dflanag3/branching-g3/multi-code-contexts-20261014/fixtures/q4_64k.json \
  --generated 2048 --exercise-rewind
```

The campaign uses three roots, six A/B trees, three rewinds, separate prepare/
resume/rewind native processes, full matching references, unchanged-parent
checks, and one worker on GPU 0. The fixture overrides synthetic size defaults.
The campaign already fails its terminal check if `require_clean_evictions` is
true and there were none (`scripts/pool_campaign.py:205-293`). The pool's RAM
and disk byte limits, LRU clean eviction and publication admission are in
`src/context_time_travel/pools.py:47-58, 93-157, 251-305`; cold disk read,
RAM reservation and upload are in `src/context_time_travel/coordinator.py:579-588,
650-690`.

To substantiate the complete GPU-to-RAM-to-disk pressure path in the resulting JSON,
also inspect the per-request rows (the current runner does not gate every tier
transition): require nonzero clean evictions; a resume request with nonzero
`loaded_file_bytes`/`checkpoint_import_bytes`; native cold-restore-copy or
mapping counters; on initial tree requests, `forward_tokens` equal suffix plus
actual generated outputs and `prefill_tokens` equal the suffix; on disk-resumed
requests, `forward_tokens` equal only newly generated outputs and
`prefill_tokens == 0`; durable file bytes; and `max_active_per_gpu["0"] == 1`.
If these witnesses are absent, report that tier transition as unproved even if
the campaign's numerical result is PASS. The RAM pool is an accounted checkpoint
extent budget, not whole-process RAM. A fresh free-space observation is required
before launch. `min_available_bytes` is a host `MemAvailable` floor, not disk
headroom; the separate `disk_bytes` field is the checkpoint-pool budget. The
latest llm-60 `/` observation is only about 32.9 GB free, against approximately
30.35 GB of projected checkpoint files for this campaign. That leaves too
little margin for a responsible run, so do not launch until disk space is freed
or an approved larger filesystem is selected. The older 179,081,596,928-byte
free-space observation is stale and must not be used.

The same-process causal evidence hook is now implemented in this branch, but no
full GPU campaign has run with it. Opt-in `PoolTransitionTrace` records bounded
sequence-numbered RAM clean-eviction and disk-preload start/success/failure
events, keyed by the existing `(model_contract_id, execution_profile_id,
snapshot_id)`, with coordinator phase, process ID, and phase-instance ID. The
success record is written after disk load and RAM admission at the same lock-
protected linearization point, before the future wakes consumers. The eviction
record is written at the key/buffer removal point. Each start event's sequence
is echoed as a load ID on success/failure, so qualification requires eviction,
then that exact same-key load start, then successful completion. An in-flight
read that started before eviction cannot count, and a failed load leaves its
eviction available for a later successful retry. Events identify `freed_bytes`
as the pool's unique-payload accounting delta; they do not claim an OS RSS drop
or removal of external/native references. The campaign validator rejects
overflow, sequence gaps, mismatched process/phase/key/load ID, reversed order,
absent or failed loads, and loads without positive byte evidence. A shared-
payload alias eviction with zero freed bytes cannot qualify. This is separate
from native transfer counters and RSS telemetry.

An opt-in `--pressure-order interleaved-roots` changes prepare-phase request
submission to tree IDs
`[0,2,4,1,3,5]` while leaving the fixture's source mapping and branch histories
unchanged; default submission order remains the existing sequential order. A
config with `require_spill_reload_same_process: true` enables tracing, requires
the interleaved order, and requires a same-phase/process exact-key match before
the campaign can report PASS. Existing per-request full-reference and prefix-
work checks remain active. The focused CPU suite exercises the real pool's
eviction/load recorder and a deterministic completion-callback race, plus
validator adversarial cases. It does not prove the full GPU campaign will
produce a matching transition under the proposed 16 GiB budget.

Final verified remote llm-60 CPU evidence is under
`/home/dflanag3/spatial-artifacts/p5-spill-reload-luna-20261015/final-verified-v4/`.
The frozen source snapshot was staged at
`/home/dflanag3/branching-g3/p5-final-checkout-luna-20261015-v3/` from a
44 MB archive with SHA-256
`d7eff9b17fa1e96b62e84934ef0937cd45735604ba54143b77dd70e62aac922b`.
The verified launcher was
`/home/dflanag3/spatial-artifacts/namespace-q8-browser-ac6-20261014/run-command-v2.sh`
(SHA-256 `7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537`).
Its `command.exit` and `wrapper.exit` are both numeric `0`, and
`terminal-status.txt` is `COMPLETE`. The exact test script was transferred as
LF text and invoked through that launcher. Interpreter was Python 3.12.3 on
`llm-60` (`dflanag3-B650-GAMING-X-AX`); `regex` 2026.9.29 came from the existing
verified path `/home/dflanag3/issue-2026-20261012/deps/regex/__init__.py`.

The final suites reported 221 pool-control tests (including all 17 trace tests),
51 G0 tests, and 118 supplemental serving tests. Each reported zero failures,
errors, skips, expected failures, or unexpected successes. The pool-control JSON
records every executed test ID and has SHA-256
`f4d1db51240308efd7b8a40113c0857486a03114e81b67579f9235ac230afc70`; the
serving JSON hash is
`218ee18560204b8d61efc445284a7b3241789adefef6f54b5c3e2ed030b3b11a`, and the
G0 terminal log hash is
`1299f1bebf5cd8c8fad5d4cf831605bffbf15af92795fa140579c1cc6b2d3b55`.
Pool suite executed-ID digest is
`4c9032404a046b477fa57397380ca90cd18404e395c154fc906549df7893ab82`.
An earlier transferred staging omitted `tools/strata_tokenizer.py`, and the
next preflight initially lacked the known `regex` dependency path. Those setup
failures are preserved in `.../final-verified/` and `.../preflight-v3/`; they
are not counted as test passes. The earlier malformed launcher exit receipt is
also preserved and is not used as a numeric result. The intentional red test
import failure remains in `runs/red/unittest.log`.

Final source hashes copied into the frozen staging tree matched:

```text
Host: llm-60, dflanag3-B650-GAMING-X-AX
Python: 3.12.3
pools.py: 7c90645bf87a414e1686543a0b099e5c9648616c391e74bee754496fd9b2a78f
coordinator.py: fa4985406ed6f21e2791c2e1439cd93d337823f742dcf960440664d8a4ba0fac
pool_campaign.py: 2a946ae7f1d0502f68f01ac9e0e13c5d6231935f17b63e68d6aae7ea009c5f8f
test_pool_spill_reload_trace.py: bb9d46b8128ef6342c56f92e78200d8f085cf803cfc02ceb5a1eab588dd05f59
verify_pool_control.py: a859c79502409a65fd5b31f8430d8f04ec2c8b2f45960b539646ca79fd8a19d0
```

This ticket changes Python control-layer sources; previously qualified native
runtime receipts do not automatically identify this control source. A future
full GPU campaign needs a new frozen control-source manifest bound to its run.
No GPU, server, model inference, or full-capacity pressure campaign was run for
this ticket.

## Scope and limitation

The full-capacity pressure run remains a Q4 research proposal, not execution evidence. It avoids the old
12,288-token/2 GiB iteration-scale shortcut: native capacity remains 98,304 and
real contexts remain 65,536 + about 8.3K + up to 2,048. Q8 should follow only
after the Q4 budget run demonstrates successful pressure spill, disk reload,
one-request scheduling, and exact references. No new framework or model is
needed.
