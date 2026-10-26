# Rank-one merge audit: preserving the singleton dimension

MiniCPM's completed rank-one adapter merged successfully through the existing
native CPU exporter. The independent FP64 auditor then refused its geometry:
GGUF had omitted the A matrix's trailing dimension of size one, so NumPy saw a
vector rather than a two-dimensional matrix. This was an audit-reader refusal,
not a failed native update, a new CUDA error or evidence that rank one cannot work.

The reader now restores `(1, input_width)` only when the base matrix's input width
and B's explicit `(output_width, 1)` shape uniquely confirm that interpretation.
It still refuses mismatched widths/ranks, missing B, ambiguous vector B and
higher-dimensional inputs. Existing explicit two-dimensional pairs keep their
original geometry. The error bound was not relaxed.

Four CPU test methods cover explicit/collapsed rank one, rank two with independent
hand-computed products, and eight malformed/ambiguous shape combinations. Run:

```sh
python -m unittest tools.test_lora_merge_geometry -v
```

The repaired reader then audited the **already-written** real rank-one F32 model.
It compared every saved weight with independent FP64 `W + (alpha/rank) B A`:
381 tensors, 295 adapted matrices and 2,516,756,480 scalar weights. Maximum error
was **7.277862712484762e-8**, below the unchanged **2e-7** gate. Unadapted tensor
values were exact. Original base, adapter and merged-file hashes were verified
before and after the audit. No native merge, training update or inference was
repeated to obtain this audit.

The [complete weight-audit receipt](../validation/rank1-merge-weight-audit-20261026.json)
contains per-tensor errors and immutable input identities. This qualifies saved
weight reconstruction, not generated-answer or logit equivalence. The separate
standalone F32 reference and quantization assessments still determine behavior.
All model weights remain private and no model is promoted by this repair.

The previously credited [llama.cpp PR 30142](https://github.com/ggml-org/llama.cpp/pull/30142)
addresses the CUDA singleton outer-product leading dimension. That native fix
and this Python audit-reader correction are distinct. The earlier failed audit
and original successful merge receipts were retained rather than overwritten.
