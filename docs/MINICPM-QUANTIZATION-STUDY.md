# Does quantization preserve the learned behavior?

This is an approved experiment in progress. Conversion is not behavioral
qualification. The original MiniCPM baseline used F16 saved weights with FP32
activations; the first trained candidate uses F32 saved weights. Its 59/864 to
482/864 exact-answer change was measured with the same harness and limits, but
an unchanged-original F32 storage control is still needed to isolate the
contribution of storage/kernel precision from learned weight changes.

## Native support and ownership

PCC's pinned llama.cpp dependency already contains `llama-quantize`, including
Q1_0, Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q8_0, F16 and F32. It does not need a newly
invented quantization engine. An isolated CPU-only executable has been built from
pin `972d2313bc0bf0a45f634f77d95c9fb03aeab12c`; serving builds and their qualified
libraries were left untouched. The PCC owner-operated wrapper
`python -m tools.pcc_model_quantize` records input/tool/source hashes, bounded CPU
settings, original exit/logs, output hashes, tensor names/shapes, actual storage
mix and unchanged architecture/tokenizer metadata. It refuses a lossy input
master, an incorrect source hash or an existing output directory.

The GGUF header reader now correctly identifies upstream type 41 as Q1_0, with
128 values in 18 bytes; it retains the custom Q2_0 type 42 used by the original
Strata work. This is metadata/geometry recognition, not a reimplementation of the
quantization kernels. Actual tiny native F16/F32 conversions and source/output
refusal checks pass. The quantizer may reorder tensor records; identity checks
compare unique tensor names and shapes rather than requiring unchanged order.

## Frozen comparison matrix

Compare both the original model and the first trained AnswerStop-01 checkpoint
at F32, F16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K and Q1_0. The trained master SHA256
is `e83249ef7576e6223141bca717b0d8c8442b9f53e0b14cfea775f87be4701e42`;
the original F16 master is `0ffba3682a853295566b98bc38c0ab755d6b2d91b994bc031ed727a08cfcdf25`.
Convert every format directly from its floating-point master, never from another
lossy quant. Use the same pinned converter/default policy, without an importance
matrix in this first comparison. Calibration variants, if useful later, require
separate practice-only calibration data and separately labeled experiments.

Q1_0 is the named 1.125-bit tensor format, not a promise that every model tensor or
the entire file is exactly one bit per parameter. K-format defaults can retain
higher precision in selected tensors. Record actual tensor counts/types and file
bytes, not just the friendly format name. Do not assume accuracy is monotonic in
bit width or that the original F16 score is an upper bound.

Use the same frozen 864 development cases, checker, rendered token prefixes,
512-token context/batch, greedy decoding, 64-token response cap, sampled stop IDs
and qualified numerical execution profile within each hardware comparison.
A100 execution requires its own pinned oracle/build and floating controls; do not
silently compare a new A100 runtime to the Blackwell baseline as if the profiles
were identical. Record shared contention and process-attributed VRAM, with the
context bound. Unsupported/load failures are unmeasured, not zero model scores.

Report for each row:

- Exact answers, natural completion, and paired per-skill/gate gains/losses.
- Change from its own floating master and from the other model at the SAME format.
- Learned advantage at that format, compared with the measured floating-model
  advantage. A retained-gain ratio is undefined if its reference advantage is zero.
- Exact bytes, tensor storage mix, observed process VRAM and model/token work.
- Converter/build/model/evaluator identities, actual costs and limitations.

The original F16-to-F32 control is checked value by value against all saved source
weights before inference. Equal stored values do not prove equal native kernel
behavior; that is why the behavior control remains necessary.

## Run conversion

```sh
python -m tools.pcc_model_quantize \
  --executable /path/to/qualified/llama-quantize \
  --source /models/floating-master.gguf --source-sha256 PINNED_SHA256 \
  --format Q4_K_M --threads 4 --output /new/quantized-variant
```

The output receipt says `CONVERTED_NOT_EVALUATED` until the independent S3 suite
has actually run. Named model files and their local notes remain unpublished.
The full first quantization matrix is a separate experiment alongside continuing
mixed-repair learning. Later loop checkpoints can prioritize deployment formats
without erasing the owner's requested full original-versus-trained matrix.


## Conversion checkpoint

All seven newly generated original-model formats converted successfully; original
F16 is reused as the eighth row. [File sizes, storage mixes and hashes](../experimental/native_training/validation/original-quant-conversion.json).
The F32 original has 10,072,171,488 bytes; Q8_0 2,679,710,688; Q5_K_M
1,808,061,408; Q4_K_M 1,561,318,368; Q3_K_M 1,291,420,640; Q2_K 1,039,926,240;
Q1_0 541,475,808. These are conversion results, not accuracy/VRAM results.
All 2,516,756,480 saved weight values across 381 tensors match exactly between
the original F16 and converted F32 when represented as float32.
[Value-control evidence](../experimental/native_training/validation/original-f32-value-control.json).
Trained-model conversions and the full paired behavior matrix remain outstanding.
