# Fresh native multi-turn oracle

This is an isolated oracle experiment, not a production serving or cache implementation. The fresh executable performs normal Prefill for the rendered root and first user as two separate requests, generates the first reply with the ordinary verifier until actual EOS, validates all 2,042 token IDs against the fixture, Prefills only the 59-token next-user suffix, then continues ordinary verifier generation. It does not use `BranchSession`, `ConversationCache`, checkpoint restore, or captured first-turn tokens as second-turn input.

The isolated checkout is branch `work/luna-fresh-multiturn-oracle`, based on `7b8364de67fc6082f3a7fec8fd732cb03d1c3f3c`. Owned source files are `scripts/build_native_multiturn_oracle.py`, `scripts/run_native_multiturn_oracle.py`, `scripts/test_native_multiturn_oracle.py`, and `scripts/queue_native_multiturn_oracle.sh`. Compact receipts are in this directory. The pinned source and binary trees were read-only.

## Validation

The remote CPU contract suite ran on llm-60 with:

```sh
cd /home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/scripts
python3 test_native_multiturn_oracle.py
```

It executed 13 tests: 13 passed, 0 failed, 0 skipped. IDs and source hash are in `cpu-tests.json`. Test coverage includes actual EOS versus output-bound rejection, exact root/first-user request segmentation, generated fresh-Prefill/normal-decode path shape, profile prompt option handling, expected bare first-frontier path, and the measured no-suffix-work EOS guard.

The native v5 build ran through the Spatial `run-command.sh` wrapper in tmux. The build command was:

```sh
bash /home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/scripts/run-command.sh \
  /home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/scripts \
  /home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/build-v5-command \
  python3 /home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/scripts/build_native_multiturn_oracle.py \
  /home/dflanag3/branching-g3/native-request-probe-20261014/source-digest-v1 \
  /home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/multiturn-build-v5
```

`build-command.exit` and `build-wrapper.exit` are both 0. The executed fresh binary is `/home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/multiturn-build-v5/strata-batched-fresh`, SHA256 `9026f76da18ee7245984e2ba9adfc0abcc458908fac4074cc20097b24b7729a7`. Generated `fresh_prefill.cpp` is SHA256 `b2b12cbf6dd6cb03d296c9158d3ea7e45db0150e175f5d434f20abe27ace91cc`. Pinned source hashes, build flags, link recipe and compile commands are in `build-v5.json`. The build receipt’s `runtime=UNRUN` field was written at build time; runtime evidence is separate in `full-run.json`.

The GPU sequence used tmux `luna-multiturn-sequence-v4` and held `/home/dflanag3/branching-g3/native-client-speed.gpu0.lock` for both tests. The queue script contains the exact two-stage invocation and launches each case through `run-command.sh`; it should be copied to a new attempt with unique output directory names before reuse because the original run directories are retained.

The first case was a 128-token prompt with `--max-new 1`. It exited with the intended code 3 and logged `first_generated_tokens=1 first_prefill_work_tokens=128 prefill_work_since_first_prompt=0`; no second-turn sidecars were written. The runner recorded `PASS_EXPECTED_EOS_GUARD_REJECTION` in `tiny-eos-rejection.json`.

The full run receipt is `full-run.json`, from `/home/dflanag3/spatial-artifacts/native-multiturn-luna-20261014/full-q8-two-turn-v2/output/run.json`. Native process exit and wrapper exit are both 0. It consumed the distinct Prefill requests `[65579, 8258]`; the first request completed in nine chunks, then the first turn generated exactly 2,042 tokens ending at EOS 248046 and matched the fixture token-for-token. It then consumed 59 suffix tokens at position 75,879 in one native Prefill chunk and generated the expected next-turn 128 tokens.

Measured native counters: first Prefill 73,837 tokens/11 chunks; suffix Prefill 59 tokens/1 chunk; total Prefill 73,896/12; normal forward tokens 2,170; total forward tokens 76,066. `prefill-requests.json`, `prefill-chunks.json`, and `multiturn.json` contain the actual schedules, frontier/controller data, and counts. The raw frontier traces remain only on llm-60 under the run directory. The parent independently audited the full traces: all first and second generated IDs, both complete controllers, root frontier and both final frontiers matched exactly; counters and pinned profile/fixture/binary hashes were verified.

## Preserved unsuccessful attempts

An initial test-first CPU RED was observed before the builder module existed (`ModuleNotFoundError`); the final suite is the green result. Build candidates v1–v4 were build-only and were never executed; only v5 was used for the successful native run. The first queued GPU sequence used nested flock acquisition and was stopped before native execution. A subsequent preflight failed because the pinned profile has no `--tokens` option; its traceback and exit receipt are preserved as `failure-profile-option.stderr` and `failure-profile-option.command.exit`. A retry then reused an existing run directory and exited 73 before launching the binary; its log is `failure-run-dir-collision.log`. These failed preflights did not perform inference. The final attempt uses one flock acquisition and the profile’s actual `--tokens-file` handling.

No model weights, profile, oracle fixture, or raw frontier trace were copied into this checkout. No public push or publication was performed.
