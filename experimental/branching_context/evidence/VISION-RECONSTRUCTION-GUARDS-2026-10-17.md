# Real-Q4 image artifact rejection and reconstruction

All five bounded cases pass through actual CLI/HTTP/COW/CUDA. Four negative
cases reject with the intended actual IntegrityError, preserve canonical history,
controller and branch head, and evaluate zero main-model tokens:

| Case | Encoder calls | Main-model forward cells | Observed guard |
|---|---:|---:|---|
| One-byte corruption in cached embedding | 0 | 0 | Embedding payload digest mismatch |
| Position digest changed, envelope checksum recomputed | 0 | 0 | Checkpoint manifest incompatible |
| Rotary cursor changed, envelope checksum recomputed | 0 | 0 | Checkpoint manifest incompatible |
| Regenerated embedding differs by one finite float | 1 | 0 | Embedding payload digest mismatch |

The fourth case runs the actual GPU encoder, then deliberately changes one output
float in a declared test-only wrapper. It proves the immutable-input guard rejects
a differing regeneration. It does not show a defect in the normal encoder.
No mismatched embedding is published. Native and production Python code are unchanged.

The successful control starts a separate inference process after all derived
checkpoint/embedding copies have been removed from its private Store. It re-encodes
once, reports366 paid prefix cells and consumes36 output tokens, totalling402
model-forward cells. Its saved-prefix frontier hash and all36 tokens match the
uninterrupted reference. The retained source/controller stay unchanged. This is
explicit reconstruction, not strict zero-prefix-work restoration.

A separate executed read-only measurement compares the final saved frontiers:
all248,320 float32 logits are byte-identical, maximum absolute error0. Positions,
rotary cursor122, consumed count402, native input identity and complete execution
schedule also agree. Both raw frontier vectors and the comparison program are
archived. This is one matching-arrangement saved-frontier comparison, not the
separate full-history numerical oracle or every generated-token frontier.

Five new harness methods plus25 serving regressions pass30/30, G0 passes51/51;
zero final failures/errors/skips. The first campaign already passed all5 behavior
checks but labelled total402 as prefix work in its successful-case summary. A new
accounting regression and explicit total/prefix/generated split corrected that;
the complete real-GPU campaign was repeated successfully on sourcev2. Initial
missing-implementation REDs and sourcev1 outputs remain retained unchanged.

Each arm used a fresh native process, so a resident cache could not bypass the
damaged disk artifact. All six server/native pairs exited0 and are absent. The
normal producer and successful-control thoughts close naturally at33 within512.
Source/root payloads remained unchanged across every case. Profile1fcece65,
Q4/FP16 KV, GPU0, capacity98,304, prefill chunk512, image300 rows/width2560.
Detailed measured native bytes, lifecycles, timings and fingerprints are in JSON.
No whole-model RAM/VRAM peak or throughput benchmark is claimed.

Private raw backup:
box-git:/srv/data/git/private_context_cow.evidence/20261017-vision-reconstruction-guards-v1/evidence.tar.gz
42,805,278 bytes, SHA2568dfed9382bf86c495aabea432b48a5e61517dc3ecf41b0442884a82da604b5e0.
It contains1,652 regular members;1,651 data files were individually hash-verified,
with manifest member coverage checked. Optional numeric payloads are inventoried
but excluded; canonical metadata, source inputs and measured frontier vectors remain.

## Reproduce

Run scripts/run_vision_reconstruction_probe.py with --runtime, --branchwrite,
--profile, --dependency-root, --vision-config, --fixtures and a new --output.
The archived run-vision-reconstruction-probe-v2.py records exact paths and the
profile environment. Use the established GPU flock and external run-command.sh.
Frozen sourcev2/manifest.json pins both harness/test inputs. Do not reuse an
existing output directory or alter the pinned producer/profile. The explicit
regeneration-mutant marker must never be enabled in production.

After completion, compare saved vectors using archived
compare-vision-reconstructed-frontiers-v1.py --campaign CAMPAIGN --output NEW_DIR.
The checked command is retained in vision-reconstruction-frontiers-run-v1.
This comparison requires optional state payloads before their recorded retirement;
the retained raw vector pair can subsequently be compared without native state.

This advances VC08 artifact/position rejection and regeneration mismatch coverage.
VC08 remains incomplete: publication-stage crashes, allocation failures, pressure,
and deeper cancellation still need their campaigns. VC09 Q4 8K/restart/performance,
remaining VC06/07 limitations and original G1-G5 qualification remain open. Chat
stays off, Q8 stays deferred, and all new work stays on Linux/private LAN Git.
