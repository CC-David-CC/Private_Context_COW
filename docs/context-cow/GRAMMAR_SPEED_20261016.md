# Request-owned grammar controller measurement

The constrained path spent most of its time copying and validating a
248,320-token controller policy for each output. This ticket keeps that policy
private to one request and exports a detached, fully validated controller once
at the consumed request boundary. Controller format, sampler, native kernels,
checkpoint format, sealed-byte hash policy and execution schedule are unchanged.

`GrammarEngine.restore` still validates identities, canonical history, parser
progress and decoder state. Public `consume(controller, ...)` still rejects
stale state and returns detached state. The private loop uses `_consume_selected`;
every transition still checks token bytes, grammar legality, EOS and control
tokens. Sessions reject foreign-thread access before mutation. Controller exports
are detached. Cancelled/failed ingestion cannot publish privately advanced state.

## Executed comparison

llm-60, RTX PRO 6000 Blackwell GPU0, Qwen3.8-Flash-Next UD-Q4_K_XL, FP16 KV,
capacity98,304, prefill8192 and normal-verifier generated-token operator. One
active GPU request. Both profiles freshly computed the same7,231-token code
prefix. Each state had one excluded warmup, three measured repetitions and one
separately instrumented repetition. Outputs include EOS:13/29/11/10. No tools ran.

| State | Baseline generation TPS | Request-owned generation TPS | Complete request, before / after |
|---|---:|---:|---:|
| READ | 12.96 | 30.33 | 1.228 / 0.661 s |
| EDIT | 11.49 | 23.14 | 2.751 / 1.500 s |
| TEST | 13.09 | 30.45 | 1.062 / 0.596 s |
| FINISH | 13.06 | 30.35 | 0.989 / 0.562 s |

Generation TPS times selection through native consumption; it excludes setup,
final capture and publication. Complete request includes those operations,
but excludes CLI/HTTP and durable disk save. Short actions do not establish
long-output, ordinary-serving or Q8 throughput.

Instrumented READ controller consumption fell559.34ms/13tokens (43.03ms each)
to0.577ms/13tokens (0.044ms each). Complete validation remains at boundaries:
three calls per action, versus output_count+2. READ selection247.1ms/13tokens,
native consumption120.1ms/13tokens, logit reads48.5ms/13tokens. EDIT selection
858.2ms/29tokens. Host selection now dominates per-token cost. Inclusive wall
timers overlap and must not be summed. No GPU-event attribution is claimed.

All20 actions per profile matched the earlier qualified same-operator reference
in tokens, controllers and final full vectors:40 comparisons, error0. Every
action evaluated exactly its outputs and preserved P. The new-profile restart
campaign also passed: save after four READ tokens, exit Python/native, restore
in new processes, complete READ and run all four actions from P. Both strict
restores had zero forward tokens; five final-vector comparisons had error0.
Benchmark/restart native processes and restart Python children exited0 and were
absent on readback. This samples final frontiers, not an every-token oracle.

Final combined CPU/filesystem/HTTP/CLI suite304/304; G051/51; no failures/skips/
xfails. Seven new tests cover public/private controller equality, no per-token
full copy/validation, detached exports, thread ownership, stale state, invalid
tokens and UTF-8 partial restore with an independent sibling. Six baseline
missing-method errors are retained as the test-first red run. No tolerance,
reference equation or registered gate test changed. Read-only review found no
confirmed defect; its suggested owned-path UTF-8 regression was added and passed.

## Fingerprints and retained evidence

Base`f9e979f4e132eef550c2a27124e0d6db9f96e26d`; private branch
`work/grammar-speed-20261016` in `context-cow-grammar-speed`.
Profile`6178300d8adb160df429335e43e954039c8dcb527b63ee72d42d2df72f911417`;
profile SHA`cf5d6039718476d1c58aabf65d691d2290b6b93a632b52f3f299e1da93401559`.
Unchanged native SHA`4fb6f7978d6af1e8a11d448e760799aa3207f4953dc78a75d76361af15121d61`.
Model aggregate`32759e8867f3e52a7fa559761d003950dd4449e7e970a6de1550bc11c02b8827`.

Full pins, test IDs, counters, timings and process receipts:
`experimental/branching_context/evidence/GRAMMAR-SPEED-QUALIFICATION-2026-10-16.json`.
Raw root:`/home/dflanag3/spatial-artifacts/grammar-speed-20261016-v1/`.
`baseline-v1`, `candidate-v1`, `restart-v1` are terminal PASS;
`combined-cpu-v3.json` is the final304 suite. `root-green-v2` was an unexecuted,
test-only staging intermediate. CPU source and GPU runtime API hashes agree.

To recover space, production DiskCheckpointPool evicted only old grammar runs
`gpu-q4-v1`/`gpu-q4-v2` derived states:923,894,265bytes. Every canonical file hash
was unchanged. Their manifests, vectors and failures were already backed up;
the archive hash was reverified. Derived .state files were deliberately not
archived. Final old v3 and current restart checkpoints and earlier three codebase
roots remain on llm-60. Exact removed-file hashes are in the cleanup receipt.

FOCUSED-PASS only. Original G1-G5 unqualified; no new independent full-history
oracle or native-speed parity. Global peak RAM/VRAM was not sampled; retained
native memory counters are not whole-system peaks. Q8, Min-P, nonzero RNG,
broader controller settings and serving grammar remain unqualified. Chat off;
private only. Next: reduce host selection cost while preserving finite-logit
rejection, exact legal argmax and lowest-ID ties; measure narrow/dense masks.
