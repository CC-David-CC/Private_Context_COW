# Native grammar selector builder handoff — 2026-10-16

## Scope and result

This isolated builder adds a native `select_mask EXPECTED_CONSUMED MASK_FILE` command to the receipt-pinned generated branch driver. It reuses the session's current host frontier (`session.logits()`), performs no forward/decode/session mutation, validates the complete mask and every logit, and reports a selected token under `finite-mask-argmax-v1`. The build receipt reports `BUILT_NOT_EXECUTED`; no model/GPU or numerical campaign was run.

## Source and CPU receipts

Worktree: `C:\Users\dflanag3\Documents\fleet\context-cow-grammar-selector-build`, branch `work/grammar-selector-build-20261016`, base `b139911e6fbe28de4553d3a000f156f700faa6fb`.

The test-first placeholder stage `red-v2` ran four methods: the three builder/reference tests errored on the unimplemented transformation/reference functions, while the pure C++ selector-helper test passed. This is a behavioral implementation RED; the earlier `red-v1` import/header setup failure is retained separately and is not cited as the RED result.

Final source stage on llm-60 is read-only at `/home/dflanag3/spatial-artifacts/grammar-select-20261016-v1/native-builder-v1/final-v2/source`. Using `/home/dflanag3/src/Strata/.venv/bin/python -B` with bytecode writing disabled, the approved runner recorded test command/wrapper exits **0/0** and `test_grammar_selector_native.py` completed **4 tests, 0 failures, 0 skips**. Final local and staged SHA-256 values match:

| File | SHA-256 |
|---|---|
| `native/grammar_select.h` | `354b5da4b17546833fa430a787c1d88415ae0971aee6aefc33c3d1af08f2f9dc` |
| `scripts/build_grammar_selector_probe.py` | `1cc7ddb77c990096b6663f26aff1be236f4798450c8d7de4a998edbcb9121391` |
| `tests/implementation/test_grammar_selector_native.py` | `f896493c31816119e6e3fbd7bb5c4aab3a9ab3c20432ec3021a982a903505588` |

The pure C++ helper tests exercise lowest-ID ties including signed zeros, masked-out NaN rejection, empty support, invalid mask width, nonzero trailing bits, consumed mismatch, and seeded randomized scores/masks compared with an independent Python reference.

## Build receipt

Input receipt: `/home/dflanag3/spatial-artifacts/grammar-first-20261016-v1/native-receipt-build-v1/output-v4/build.json`, SHA-256 `25d4d87214128f394528f46408174fd918b64e9310b89e7c6024599eed37aead`.

Generated output: `/home/dflanag3/spatial-artifacts/grammar-select-20261016-v1/native-builder-v1/final-v2/output-runner/output/build.json`, SHA-256 `b1b1e59e6529c0d3c7e94a9a8edd02810a73e18028b4f4118de5e9f6de1e5ef2`. The approved runner recorded build command/wrapper exits **0/0**; its receipt records compile and link subcommands both exit 0. The build compiled one generated driver translation unit and linked it in place of the prior driver object; the other recorded objects and libraries were checked against the input receipt and reused. Compiler warnings in the generated driver are existing warnings in the pinned driver (signed `tellg()` comparison and unused function parameters); the selector helper added no warnings.

- Generated driver SHA-256: `673824cfb5c8c406e26b33f833ae09be34b9debafae98bda932e392f13cc594e`
- Generated driver object SHA-256: `341460940798a16d80fad4cf045523c209f3a5cffa8e72141c37d45a40547b31`
- Selector ELF SHA-256: `2b58e8e872e46705805bf119bf7c7b382c85b63114b03f7f00f61f2452b225dc` (70,598,440 bytes)
- ELF status: `BUILT_NOT_EXECUTED`; numerical qualification is `UNRUN`.

Outer runner receipts are under `/home/dflanag3/spatial-artifacts/grammar-select-20261016-v1/native-builder-v1/final-v2/receipts/{tests,build}/` (`command.exit` and `wrapper.exit` are `0` in both runs). The test stdout/stderr and exact argv are retained there; the per-build compile/link argv and output hashes are also recorded in `build.json`.

The earlier `built-v2` compile failure from a malformed stats anchor is preserved. It did not change inputs and was corrected before the successful final build. The finite-value guard reads IEEE-754 binary32 exponent bits, so NaN/Inf rejection does not depend on floating-point finite-math optimization assumptions. No GPU launch, profile change, native engine change, or commit/push was made.
