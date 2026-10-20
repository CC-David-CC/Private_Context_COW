# Checkpoint digest candidate: Q4 validation plan

**Status: PLAN ONLY. No GPU command has been executed by this plan author.** The native owner
reports candidate stages at
`/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red`
and `/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/g0stage`,
each with 17 staged modules matching its local hashes. Scripts and helpers are
copied under `red/scripts` to ensure the probe's own `API/src` resolves to the
candidate code. The candidate manifest and import-origin witness are listed
below; root owns the separate launch preflight for current resources and GPU
lock state. The old source checkout/profile is not a substitute for these
candidate-bound artifacts.

The intended validation is one Q4 model, one worker on GPU 0, and three small
greedy A/B request cases, in order: 128/4/8, 1,024/64/16, and 8,192/512/128
(prefix/suffix/output limit). Keep model capacity at 98,304, Prefill chunks at
8,192, KV at FP16, and the existing native request and verifier operators.
Leave CPU affinity and the host runtime environment at their normal full-host
settings. These are correctness regressions for the Python checkpoint-digest
candidate, not a throughput or cache-speed qualification.

## Harness choice

Use `experimental/branching_context/scripts/pool_campaign.py campaign` directly.
This is the existing canonical path that exercises `ContextCoordinator`,
`SharedCheckpointPool`, `BranchingModel`, durable checkpoint publication, and
the selected native worker. Its campaign separates `prepare` and `resume` into
different Python/native processes, evaluates distinct A and B branches from a
shared root, captures the A midpoint after four accepted outputs, and compares
actual tokens/frontiers with fresh matching-native references. The resume
phase checks disk bytes and zero prefix-forward work; the campaign also checks
the saved parent payload remains unchanged and that no GPU has more than one
active worker.

Do not use `durable_resume.py campaign` for this candidate: that harness is
batches-only and explicitly requires zero verifier decode work. Do not use
`run_matching_restart_campaign.py`: it stops and restarts a chat server, outside
this isolated validation. `native_request_probe.py` is an independent native
request check but does not enter the canonical Store/pool path or test a
process-separated Store restart, so it is not the primary campaign.

The approved pilot uses `--roots 1 --trees 2` and omits `--fixture` and
`--exercise-rewind`. The pool campaign creates deterministic token-ID prompts
in this mode, with distinct forced suffix histories for A and B. This is real
Q4 inference through the candidate COW code, but it
does not represent the three codebase-text fixtures or the browser workflow.
No text or UI claim should be derived from these runs. Rewind behavior is also
outside this pilot; the no-fixture form does not satisfy the campaign's
fixture-specific rewind mapping requirement.

For tiny and medium, pair the campaign with `native_request_probe.py
--check-boundaries every`. This separate primitive-level probe uses its own
deterministic A/B token fixture (not the campaign's Store branch histories) and
reruns/checks every generated output frontier for each branch, as well as the
saved prompt/root frontier. Report those results as request-primitive coverage;
do not attribute those comparisons to the campaign's canonical Store branches.
The campaign itself checks actual Store branches at the four-output checkpoint
and final tip.

## Candidate profile and preflight

Before any launch, verify this immutable Linux staging closure and receipts:

- the exact candidate API/scripts and its 17-file runtime/import closure;
- a machine-readable source manifest with every staged Python file SHA-256;
- a newly derived Q4 profile bound to that closure and the selected frozen Q4
  native build, plus the exact profile-generation receipt;
- an unbound base profile for `pool_campaign.py` (its `oracle_command` must not
  already contain `--gpu`; the harness appends the configured GPU itself);
- a separate GPU-0-bound frozen profile for the corruption control if the
  candidate profile is marked `calibration_status: FROZEN`;
- a `pool-config.json` referring to the new unbound profile, with `GPU: [0]`.

Current candidate artifact identities reported by the native owner:

- API root: `/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red`;
- campaign working tree/scripts: `/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red/scripts`;
- unbound profile:
  `/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/derived-profile/profile.json`,
  SHA-256 `dcba2359c7e625ef5d8494cfcaf2394e352201bb8f2eae104a757c73b5a55c54`,
  ID `f32db26faea8de70f46246035c086cecfb4fbb350f8bafcfcdadd44d8c4781b6`;
- config:
  `/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/config.json`,
  SHA-256 `8e8a5fa8a02f1b3a35d5ce2c2ff9e7f04ce0a27ca54a736d1ace629a1bb84fa4`;
- GPU-0-bound CPU preflight profile:
  `/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/preflight-bound/profile.json`,
  SHA-256 `a0ef7d70718ef06dfbfae7ec3e5a7dbff7f55ab252471804ba3cd5ee5cf655fb`,
  ID `4ac77a63cfb8bd0d153c8cb5fb340ab79ae4d3cc23e554d39bd967d693adfdd1`;
- GPU lock: `/home/dflanag3/branching-g3/native-client-speed.gpu0.lock`;
- numeric runner: `/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh`,
  SHA-256 `7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537`.
- profile derivation helper:
  `/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/g0stage/scripts/prepare_native_request_profile.py`;
- complete module/script manifest:
  `C:\Users\dflanag3\Documents\fleet\context-cow-checkpoint-digest-luna\experimental\branching_context\evidence\CHECKPOINT-DIGEST-Q4-CANDIDATE-PREFLIGHT-2026-10-15.json`,
  from candidate evidence commit `841f1ce51eb36ceb867f900180fe18a6c7c8725d`,
  SHA-256 `5e6f14a4a7d5c3fd0be4f6fb3688b11a03933ce2f12d59cb65ee2a702076fce5`;
- remote source/import-origin witness:
  `/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/source-import-witness.json`,
  SHA-256 `282dbf2af31ea60ba5bfa68d10022b19ae75f387b65ddf9f47063f145270f6a8`;
  it records the 17 module origins/hashes under `red/src`, three script
  origins/hashes under `red/scripts`, and verifies their source hashes against
  the derived profile pins.

Do not reuse the old `q4-request-profile-v3.json`: its capacity is 12,288. The
98,304-capacity Q4 preflight receipt records `capacity=98304`,
`prefill_chunk=8192`, `kv_dtype=FP16`, `GPU=[0]`, and native build JSON SHA-256
`e1b151aa37ddc4a75b9f6e121ccd7d73412e9a545105f1ffbe8cfc6ba053898f`; it is a
native-build reference only. Its old Python/source identity cannot certify the
new candidate. Preserve the native build and model contract, but create a new
candidate source/profile receipt; do not edit an old profile's source label or
reuse an old execution/source identity as if it covered the candidate.

Before launch, verify the staged 17-file import/runtime closure against the
candidate manifest, then assert the loaded profile says capacity 98,304,
Prefill chunk 8,192, FP16 KV, `normal-verifier-decode`, and
`identical-native-operators-v1`. Verify both SUT and independent oracle resolve
to the pinned Q4 binaries/build receipt, the base oracle command has no `--gpu`,
the pool config selects only GPU 0, and profile/model contract IDs agree with
the source manifest and generated campaign fixture. Record the profile SHA-256
and its resulting
GPU-bound execution profile ID from each campaign output.

The native owner measured 29,654,499,328 free bytes (about 27.6 GiB) and a
10 GiB free-space floor before this plan was updated. This is a planning-time
observation, not a current preflight. Check live free space and projected
output sizes again before each run; preserve the 10 GiB floor. Use a fresh
output path per case, never overwrite a prior result.
The Q4 GPU lock must be free, and only one native job may own GPU 0 at a time.
Keep all run receipts outside the source checkout.

## Commands to run after staging is reviewed

The exact remote staging path and wrapper path must be supplied in the native
owner's frozen handoff; do not guess or substitute them. For clarity, set these
shell variables to the verified paths from that handoff, then use the same
single-lock, numeric-runner form for each campaign:

```bash
API=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/red
STAGE="$API"
CONFIG=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/config.json
RUN=/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015/pilot-v1
RUN_COMMAND=/home/dflanag3/branching-g3/checkpoint-digest-candidate-luna-20261015/run-command.sh
LOCK=/home/dflanag3/branching-g3/native-client-speed.gpu0.lock

flock -n "$LOCK" bash "$RUN_COMMAND" "$STAGE" "$RUN/tiny-command" \
  env PYTHONPATH="$API/src" python3 "$API/scripts/pool_campaign.py" campaign --config "$CONFIG" \
  --output "$RUN/tiny" --prefix 128 --suffix 4 --generated 8 --roots 1 --trees 2

flock -n "$LOCK" bash "$RUN_COMMAND" "$STAGE" "$RUN/tiny-probe-command" \
  env PYTHONPATH="$API/src" python3 "$API/scripts/native_request_probe.py" --profile "$RUN/tiny/profile.json" \
  --output "$RUN/tiny-native-boundaries" --prefix 128 --suffix 4 --generated 8 \
  --check-boundaries every

flock -n "$LOCK" bash "$RUN_COMMAND" "$STAGE" "$RUN/medium-command" \
  env PYTHONPATH="$API/src" python3 "$API/scripts/pool_campaign.py" campaign --config "$CONFIG" \
  --output "$RUN/medium" --prefix 1024 --suffix 64 --generated 16 --roots 1 --trees 2

flock -n "$LOCK" bash "$RUN_COMMAND" "$STAGE" "$RUN/medium-probe-command" \
  env PYTHONPATH="$API/src" python3 "$API/scripts/native_request_probe.py" --profile "$RUN/medium/profile.json" \
  --output "$RUN/medium-native-boundaries" --prefix 1024 --suffix 64 --generated 16 \
  --check-boundaries every

flock -n "$LOCK" bash "$RUN_COMMAND" "$STAGE" "$RUN/target-command" \
  env PYTHONPATH="$API/src" python3 "$API/scripts/pool_campaign.py" campaign --config "$CONFIG" \
  --output "$RUN/target" --prefix 8192 --suffix 512 --generated 128 --roots 1 --trees 2
```

Each command must produce separate `command.exit` and `wrapper.exit` receipts.
Do not continue to the next size after a nonzero exit, an incomplete campaign,
profile mismatch, reference mismatch, or disk-floor failure. Preserve the
failed output and logs. A natural EOS before the output bound is valid if the
campaign completes and the fresh reference agrees; report actual output counts,
not requested maxima.

The expected campaign output is a new `result.json`, `profile.json`,
`prepare.json`, `resume.json`, branch frontiers/history/controllers, matching
reference frontiers, and phase-native traces. Require status PASS, zero failures,
skips, or xfails; distinct A/B suffix histories (generated token IDs may
legitimately coincide); exact token and full-frontier equality at the generated
four-token checkpoint and final branch tip against matching fresh references;
distinct prepare/resume Python and native PIDs; positive checkpoint disk-load
bytes; zero forward work during root restore; resumed forward tokens equal only
the newly generated suffix; and unchanged parent payload SHA-256. The restart
must continue each branch after the same first four accepted outputs without
restarting its controller or duplicating/omitting tokens. The campaign does not
compare every intermediate frontier for any size. Tiny and medium get the
separate primitive-level every-boundary probe described above; neither report
should imply that probe exercised the Store campaign's branch histories.
Report all per-size results separately; these small prompt cases do not qualify
the 64K pool-pressure or full original G0-G5 matrix.

For each campaign, explicitly compare the saved shared-root frontier with the
first request frontier from its fresh matching reference. In `prepare.json`,
the root is `roots[0]`; its `frontier_file` is the saved root vector. The fresh
reference is `fresh-tree-0.f32`; its first request vector is the first
`4 * manifest.logits_count` bytes of
`fresh-tree-0.f32.requests.f32`. Require the root manifest's `logits_count`
to agree with that byte width, compare the bytes exactly, and record both
SHA-256 values in a machine-readable audit receipt. Do not infer this
comparison from `native_request_probe.py`: that probe uses a separate token
fixture. The campaign checks actual Store-branch tokens and frontiers at the
four-output checkpoint and final tip; tiny/medium primitive probes check every
generated boundary for their own fixture.

Run this read-only post-audit once per campaign directory and retain its JSON
receipt beside the campaign output:

```bash
python3 - "$RUN/tiny" "$RUN/tiny/root-frontier-audit.json" <<'PY'
import hashlib, json, pathlib, sys
run = pathlib.Path(sys.argv[1])
root = json.loads((run / "prepare.json").read_text())["roots"][0]
width = 4 * root["manifest"]["logits_count"]
saved = pathlib.Path(root["frontier_file"]).read_bytes()
requests = (run / "fresh-tree-0.f32.requests.f32").read_bytes()
fresh = requests[:width]
assert len(saved) == width and len(fresh) == width
assert saved == fresh, "saved root frontier differs from fresh prompt frontier"
audit = {
    "status": "PASS", "snapshot": root["snapshot"],
    "logits_count": root["manifest"]["logits_count"], "bytes": width,
    "saved_sha256": hashlib.sha256(saved).hexdigest(),
    "fresh_prompt_sha256": hashlib.sha256(fresh).hexdigest(), "equal": True,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, sort_keys=True))
PY
```

Repeat with the medium and target directories and corresponding receipt paths.
This reads the saved checkpoint payload and fresh-reference bytes; it does not
create a native process or modify the Store.

## Corruption control and candidate-specific limits

Run the candidate CPU digest-attestation tests before the GPU sequence. They
cover exact-owner and payload/frontier identity, mutation invalidation, weakref
cleanup, bounded LRU behavior, and the actual prepare-to-pool-admit-to-publish handoff.
Run existing durable corruption controls as well; do not weaken file hashing or
restore rejection to satisfy the new RAM memo.

If the staged Q4 profile has a frozen calibration receipt and the native owner
approves its use, the existing real-inference corruption subset is:

```bash
flock -n "$LOCK" bash "$RUN_COMMAND" "$STAGE" "$RUN/m08-command" \
  python3 "$API/scripts/mutation_campaign.py" --profile \
  "$STAGE/q4-gpu0-frozen-profile.json" --output "$RUN/m08" --cases M08
```

`M08` includes a non-mutated control and a corrupted-checkpoint mutant; it is a
bounded corruption probe and does not qualify the full mutation suite. The
candidate profile must be `FROZEN`, its model/build/source identities must
match the campaign, and its output path must be fresh. If that precondition is
not met, do not fabricate a frozen profile; report the GPU corruption subset as
UNRUN and retain the CPU corruption results separately.

The digest memo is host-side. These GPU runs establish end-to-end compatibility
of the candidate with real checkpoint capture, admission, persistence and
restore; they do not establish fewer GPU hashes, physical-page performance,
latency improvement, disk-hash elimination, or memory savings. Any measured
host digest reduction must come from the focused CPU instrumentation tests and
be reported separately from native/GPU timings.

## Existing source/evidence references inspected

- `experimental/branching_context/scripts/pool_campaign.py`: canonical pool
  campaign, direct CLI and reference/completion checks.
- `experimental/branching_context/scripts/native_request_probe.py`: fresh
  matching-native references and request operator semantics.
- `experimental/branching_context/scripts/durable_resume.py`: batches-only
  control; not valid for the normal-verifier profile.
- `experimental/branching_context/scripts/mutation_campaign.py`: real-inference
  M08 corruption control, conditional on a frozen profile.
- `experimental/branching_context/evidence/NAMESPACE-Q4-FULL96-PREFLIGHT-2026-10-14.json`:
  prior 98,304-capacity Q4 native build settings; not candidate source identity.
- `experimental/branching_context/evidence/P5-Q4-FULL96-RUNNING-2026-10-15.json`:
  historical full96 disk projection and output tree size; its old source profile
  cannot qualify the digest candidate.

No GPU, native process, HTTP service, profile generation, or remote filesystem
command was run for this plan.
