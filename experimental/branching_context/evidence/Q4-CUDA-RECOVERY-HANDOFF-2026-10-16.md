# Q4 CUDA OOM recovery harness handoff

This isolated harness validates the recovery lifecycle after a real Q4 CUDA VMM allocation failure. It does not modify the native engine, serving API, profile, or registered tests, and it does not claim numerical-oracle coverage.

The runner requires the native error to reach the client unchanged, `cuMemCreate` driver error 2 with a positive OOM counter delta, and one replacement native child under the same HTTP process. It checks that the old child exited before replacement, the replacement becomes healthy with exactly one restart, and the same-service continuation plus a separately forked post-restart continuation each produce eight tokens with zero prefix prefill. Each continuation must have one durable completion bound to the answer snapshot and actual branch head revision/tip. It also checks clean child/service exits and hashes the source checkpoint again after both continuations.

CPU evidence on llm-60, using the approved `run-command.sh` wrapper and pinned Python 3.12 environment:

- Frozen stage: `/home/dflanag3/spatial-artifacts/q4-cuda-recovery-harness-20261016-v1/cpu-final-v2/source`
- `run_q4_cuda_recovery.py`: `eb7d679996236c803ccaa7b3d73beb3ad637addc4bae142e9c33d99621a561e2`
- `test_q4_cuda_recovery.py`: `424418234884dba78b814d60da7d27b75e96b5d7bc224953f1fc7ea1b6e8f3c5`
- Test receipt: `/home/dflanag3/spatial-artifacts/q4-cuda-recovery-harness-20261016-v1/cpu-final-v2/run-green`; 13 tests passed, command exit 0, wrapper exit 0.
- A subsequent reviewer found the valid stream carries a `durable` event before `complete`, with identical receipts. The first revised test run (`cpu-final-v3/run-red`) failed as expected against the old validator (14 tests, 5 errors); the positive fixture modeled the actual durable/complete pair, and negatives cover missing, duplicate, reordered, or mismatched durable receipts.
- Latest frozen stage: `/home/dflanag3/spatial-artifacts/q4-cuda-recovery-harness-20261016-v1/cpu-final-v4/source`
- Latest `run_q4_cuda_recovery.py`: `7cdab6b1e7671524929363fc7abfe70679a7af726b1d808e9077e03a7c830f1a`
- Latest `test_q4_cuda_recovery.py`: `e4beb48aa3547758db834685c9f104914b4dec913defff4462d02d6c7c46ab84`
- Latest test receipt: `/home/dflanag3/spatial-artifacts/q4-cuda-recovery-harness-20261016-v1/cpu-final-v4/run-green`; 15 tests passed, zero skips, command exit 0, wrapper exit 0.
- Actual profile/native no-model preflight receipt: `/home/dflanag3/spatial-artifacts/q4-cuda-recovery-harness-20261016-v1/cpu-final-v4/run-preflight`; `PREPARED_NOT_RUN`, command exit 0, wrapper exit 0. It used the pinned tiny input store/snapshot `96e083885945e0cf0517aa1709a1d80db3029bf04bcc4028cc8cb55a4a74e0d8` and verified profile `a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755`, native binary `4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451`, HTTP source `9f6839a359e05cbb47e0850046c6f32f00e7468265dff80fedc3d72d9f720f31`, and pressure helper `a3d2690d104f192ca0a25ee224db4b3f7fcef45d0cad4de4a491c755937cd61b`.
- The `cpu-final-v4/source` tree was read-only during both test and preflight. Runner/test SHA-256 values matched before and after both commands. The earlier zero-OOM-delta RED is retained in its separate stage. No GPU run was performed by this worker.

The Q4 runtime/profile and native binary remain root-owned pinned inputs. This CPU suite validates evidence guards; GPU qualification, fresh numerical comparison, and runtime behavior remain unrun by this worker and require the root-owned campaign and independent audit.

## Token-only fixture correction

The first root-owned tiny GPU attempt stopped during setup because its token-only source branch has no presentation namespace; the prior runner asked `/cow/workspace/history` and then attempted namespace-bound forks. The runner now cross-checks `/cow/head` against the canonical Store under a read lease and uses the plain `/cow/fork` request with only `{name, snapshot}` for its two branches. It rejects a mismatched fork name or tip. This preserves the token-only fixture and does not add presentation metadata.

- Frozen stage: `/home/dflanag3/spatial-artifacts/q4-cuda-recovery-harness-20261016-v1/cpu-token-only-v2/source`
- `run_q4_cuda_recovery.py`: `074317b5a2214dcdf71664837e64a67fd90f404ecc5a893ab399c02c71f32e21`
- `test_q4_cuda_recovery.py`: `38f17aa33b74008169e26cc447795875c861853b2953e47e9c524ba67bda4527`
- CPU evidence: `cpu-token-only-v2/run-tests`; 18 tests passed, zero skips, command exit 0, wrapper exit 0. The added positive control asserts the exact token-only request body and returned branch/tip; a wrong returned tip is rejected.
- Actual pinned-store preflight: `cpu-token-only-v2/run-preflight`; `PREPARED_NOT_RUN`, command exit 0, wrapper exit 0. It loaded the real token-only `CliA` branch and confirmed the pinned snapshot under the Store read lease, without starting the model or service.
- This preflight does not exercise the HTTP fork route itself. The root-owned next GPU run must still confirm the service accepts the exact plain-fork body.
