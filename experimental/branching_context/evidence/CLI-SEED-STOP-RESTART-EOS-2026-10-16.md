# Restart continuation EOS accounting

The CLI seed/stop probe now treats a restarted continuation's actual consumed output as the bound. It requires a nonempty canonical output span no longer than the requested budget, exact RNG and accepted-token deltas, and either (a) a full-budget active result without EOS or (b) an EOS-terminated stopped result. The earlier continuation remains required to consume its full budget and leave an active controller so the restart tests a resumable state. Budget validation no longer assumes grammar characters equal model tokens.

The preserved GPU v3 attempt reached `restart-continue` successfully, but the probe then failed its active-status postcondition. Actual operation record: `/home/dflanag3/spatial-artifacts/seed-stop-combined-20261016-v1/gpu-cli-store-v3/pool-operations/sampled-restart-continue-462b27f76bc7.json`. It records 10 consumed/forward tokens against max_tokens 16, terminal token 248046 (the profile EOS), `stopped=true`, and `grammar_complete=true`. Its answer/controller snapshot and receipt are present, but the failed probe result did not publish a `restart_continue` summary; do not treat v3 as a workflow pass.

Test-first evidence for removal of the character/token-count assumption:

- RED: `/home/dflanag3/spatial-artifacts/seed-stop-eos-postcondition-20261016-v1/runs/red-budget-v1`, command/wrapper `1/1`; `validate_sampled_budget(8, 64)` was rejected by the prior heuristic.
- GREEN focused: `/home/dflanag3/spatial-artifacts/seed-stop-eos-postcondition-20261016-v1/runs/green-focused-v1`, 10 tests passed, command/wrapper `0/0`.
- G0: unchanged runtime `/home/dflanag3/spatial-artifacts/seed-stop-combined-20261016-v1/combined-cpu-v16/source`, run `/home/dflanag3/spatial-artifacts/seed-stop-eos-postcondition-20261016-v1/runs/g0-v16`, 51 tests passed, 0 required real-backend cases, command/wrapper `0/0`.

Immutable harness-only stage: `/home/dflanag3/spatial-artifacts/seed-stop-eos-postcondition-20261016-v1/frozen-v1` (the two files are mode `0444`). Script SHA256 `7d492654bea93abc42a6fbb4343f2d74f61ffa911744dd968d102c7c457589ff`; test SHA256 `23ea3e0c0ad95ee201ef399cac7320948c18efcd15bd559321b9bfd67ae8de99`. Frozen profile v4 and production runtime were not changed. No GPU qualification is claimed by the CPU tests.

## Follow-up: grammar completion evidence

The restart postcondition now reads `grammar_complete` from the completed operation's actual per-client CLI journal answer and records that observed boolean. An EOS-terminated continuation is accepted only when its persisted controller is stopped and `grammar_complete` is exactly true. A full-budget continuation that remains active may have either completion flag: the grammar text can be complete while its EOS token is still pending. The grammar constant is named in characters to avoid implying it predicts model-token count.

- RED: `runs/red-grammar-complete-v2`, 12 tests entered, 3 errors against the preserved old script because it had neither the `grammar_complete` parameter nor `_operation_answer`; command/wrapper `1/1`.
- Intermediate GREEN attempt: `runs/green-grammar-complete-v2`, 12 tests, 1 failure caused by an overstrict test expectation for a short active grammar-complete result; command/wrapper `1/1`. The source stage is retained unchanged.
- GREEN: `runs/green-grammar-complete-v3`, 12 tests passed, command/wrapper `0/0`, terminal `COMPLETE`. Frozen source: `runs/green-grammar-complete-v3/source`.
- Final stage SHA256: script `86f96e0d9e5aa52c0bbb4bcff6b5ce941b4f4dc464fc196ff29d265a85c67ba5`; test `5fc5a63edfbb2e223368784b313dd3d93a2f0c8ee0ee1f3e93f58486cb911114`. Both hashes were rechecked after the run.

These controls validate harness interpretation only. The preserved GPU v3 campaign remains `FAIL` at its prior postcondition; the parent may rerun the corrected harness separately. This worker did not launch GPU or inference work.
