# Q4 canonical-reference ENOSPC harness handoff

Root's subsequent final v6 run is FOCUSED-PASS: CPU 20, G0 51, model-free
real-ext4 filler control, actual Q4 reference-writer ENOSPC, no-work retry and
eight-output continuation. See CANONICAL-ENOSPC-Q4-PRIMARY/QUALIFICATION for
executed evidence and limitations. The paragraphs below retain worker-stage
history; pending root qualification there is historical, not current.

The latest CPU-only candidate is frozen at `/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1/final-green-v6/source`.

| File | SHA-256 |
| --- | --- |
| `experimental/branching_context/scripts/run_q4_canonical_enospc.py` | `59ea6586862da15b93b28dbee5de19cdcdadfaa3aea1c7b77be42ac71aa3c0fe` |
| `experimental/branching_context/scripts/run_q4_publication_fault.py` (required helper sibling) | `d4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695` |
| `experimental/branching_context/tests/implementation/test_q4_canonical_enospc.py` | `c01add5e7467a21d8c10e75d1cfebdd5623d6f62662012632c2ada4e67bd99fe` |

The final approved Linux runner completed 20 tests with command and wrapper exits `0/0`; receipts are under `/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1/final-green-v6/run/cpu-tests-v1`. The new adaptive filler test proves an initial large-write ENOSPC with 4096 available bytes is followed by smaller writes that consume the remaining space before zero-available is accepted. Other tests cover a request finishing immediately after release, rejecting an already-finished request before release, filler cleanup after file `fsync` EIO and configured-cap failure, refusal to unlink a pre-existing filler, and stopping a child left by a readiness exception after process creation.

The implementation checks the future immediately before writing the release signal, after the filler receipt has been persisted and read back. The filler cleanup encloses opening, writing, flushing, closing, and validation; cleanup only unlinks a path whose device/inode match the file created by this invocation. Cleanup errors are attached to the original failure. Service failure cleanup checks whether a child process exists, including when readiness startup raised. After ENOSPC, the filler fsyncs and checks available bytes; if ext4 still reports space, it retries with 4 KiB then 1-byte writes, bounded by the configured maximum, and accepts success only after actual ENOSPC and zero `f_bavail`.

Preserve these earlier artifacts as separate attempts: `harness-cpu-v3/run/cpu-tests-v5` recorded a prior 13-test pass; `final-green-v1/run/cpu-tests-v2` recorded a 17-test pass before the release-race correction; `final-green-v2/run/cpu-tests-v1` failed from a test helper-name typo; `final-green-v3` lacked the sibling `run_q4_publication_fault.py` dependency and the root-owned mount preflight stopped before model startup (`ModuleNotFoundError`, command/wrapper `1/1`, cleanup succeeded); `final-green-v5-red/run/cpu-red-v2` is the intended RED on the old filler, which failed with 4096 available bytes after the initial ENOSPC; and `final-green-v5/run/cpu-tests-v1` is a test-fixture expected-length mismatch. The first real GPU harness attempt (`gpu-q4-v1/campaign`) safely stopped before release because ENOSPC from a 1 MiB write left 417792 available bytes; it did not exercise the branch writer. Its command/wrapper failed, and the root audit confirmed source and branch unchanged, HTTP/native exited 0, and mount cleanup succeeded. An earlier `harness-cpu-v3/source` copy was overwritten between CPU runs, so its original bytes are not available as a stable provenance record. The final v6 source stage was copied after fixes, made read-only, and hash-checked before and after its 20-test run; it contains no `.pyc` files.

No GPU or mount launch was performed by this worker. The evidence qualifies only the harness CPU controls. The actual bounded-ext4 kernel ENOSPC test, same-operation retry, and following request remain for root-owned preflight and launch.
