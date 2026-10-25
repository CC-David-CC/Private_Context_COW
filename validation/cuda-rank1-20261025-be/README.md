# Credited CUDA singleton fix: local qualification checkpoint

The imported fix is **edenfunf**'s [llama.cpp PR #30142](https://github.com/ggml-org/llama.cpp/pull/30142),
commit `fdb4a0553f6ecb00450f04cc8da5b2b38dd9146d`. This project preserves the
original author in the mail patch. These local observations are separate from
the author's reported tests.

On an RTX4090, our unchanged CUDA control reproduced the size-one outer-product
cuBLAS leading-dimension crash. The patched backend passed 103/103 CPU-reference
OUT_PROD cases, including both operand views and all three GEMM dispatch paths.
See [operator checks](operator-checks.json).

On the shared A100 server, PCC's isolated patched native build completed an
actual MiniCPM5-2B rank-one AdamW update: 27 presentations, 590 finite gradient
tensors, training loss approximately 4.264 to 4.145, and a saved adapter and
optimizer state. See [native update](native-update.json) for exact identities
and values. This is training mechanics, not a held-out improvement claim.

The first independent reload command mistakenly retained optimizer_state_output
in its evaluation-only SGD request and was refused. The corrected probe reloaded
the existing saved adapter without another update and passed: absolute loss
error 1.2583202746441202e-7 against the unchanged 2e-4 bound. See the
[independent reload receipt](independent-reload.json). The earlier
[native-update-only snapshot](native-update.json) retains its original status. Its original driver also omitted the physical
training GPU UUID, so that metadata gap is explicit. The corrected reload
records its selected physical GPU.

The broader upstream matrix checker still reports numerical failures: 31 with
default precision and 7 with TF32 disabled, with the same thresholds. Repeated
operator runs can draw different inputs. Do not infer a precise cause or a
patch regression from those totals alone. Unsupported/skipped configurations
are included in the tool's totals; these are not all completed gradient checks.

The original saved rank-one update has now completed all 864 frozen COW
assessment cases: 60 correct and 418 naturally finished, versus 59 and 415 for
the original model. That is one additional correct answer, with no original
correct answer lost; completion gains/losses were 8/5. This small development
observation establishes neither broad mastery nor a release.

A second native update resumed the exact first adapter and AdamW state, reached
54 practice presentations, retained 590 finite gradient tensors, and passed
independent reload at 1.9868e-7 against the unchanged 2e-4 bound. See the
[continuation snapshot](continuation.json) and
[S³ rank report](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/LORA-RANK-EXTENSION.md).
The snapshot was captured before assessment 2 completed. This validates the
recorded continuation, not bitwise equivalence against an uninterrupted two-update
run. Baked-model equivalence and broader numerical qualification remain pending.
No HIP/SYCL result is claimed. Original running-study libraries and recorded
failed evidence are preserved.
