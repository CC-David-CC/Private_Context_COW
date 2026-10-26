# Native training for longer tool lessons

`dense-llama-action-8k-v1` is a new opt-in ordinary-Llama training profile. It
admits up to 8,192 input tokens and at most 256 selected next-token targets per
example. It uses the existing native standard backprop engine and supports full
AdamW, full Muon and AdamW LoRA. It does not introduce PyTorch, a second trainer,
or a new learning controller.

The reason for a separate profile is practical: S³'s first faithful BW tool
lessons contain 3,029–4,873 tokens of instructions, previous actions and real tool
feedback. They cannot fit the existing 512-token training profile. Increasing an
exporter's token limit alone would not make native training support that data.

## What has been checked

The native executable was built in an isolated directory and exercised on CPU
and an RTX 4090. Generated tiny ordinary-Llama weights proved:

* Actual forward and standard full backward execution at 8,192 tokens, with
  allocator graph checks enabled. This is not only a parser/admission test.
* At 513 tokens, one scalar derivative in each of 12 full-weight tensors was
  checked against independent FP64 finite differences. The CUDA maximum absolute
  difference was about 3.14e-6. Eight rank-one LoRA B directions were checked the
  same way, with maximum absolute difference about 1.40e-9.
* Independent FP64 forward checks covered every returned logit at 4, 513 and
  8,192 tokens, including a reloaded full update. The old and new profiles produced
  identical short-input logit bytes in the same executable.
* AdamW and Muon full updates and rank-one AdamW LoRA updates resumed their exact
  longer-example optimizer stream. Separate actual HTTP/lesson/native-job tests
  enabled completion recovery before dispatch, independently reloaded saved
  candidates, resolved long-context parents and refused changed-stream resume.
  Duplicate submissions returned the original operation without another update.
* Requests exceeding 8,192 tokens, model context metadata, or 256 selected
  targets were refused. The old short profile still refused 513 tokens.

The existing CUDA short-profile qualification also passed, including independent
layer/complete-model scalar derivatives, F16/F32 input, full/LoRA update/reload,
partial batches and 42-layer standard backprop. Another 64 Python admission,
optimizer-profile and completion/recovery controls passed.

These are tiny-fixture mechanics results, not real MiniCPM training, an improved
BW score or a release decision. They do not qualify larger model geometries,
multi-GPU long-context training, HIP or SYCL. The existing native trainer exposes
CPU/CUDA device routes; no new backend is added here. Real MiniCPM allocation and
the subsequent target/workflow/retention round remain separate qualification.

Rank-one CUDA qualification used the dependency fix from
[llama.cpp PR 30142](https://github.com/ggml-org/llama.cpp/pull/30142), by edenfunf.
PCC's existing build option pins and records that imported patch. This profile
does not claim authorship of the size-one outer-product fix.

## Deep-model graph planning correction

The first real MiniCPM probe exposed another limit that the one-layer 8k fixture
could not cover: graph bookkeeping still restricted models deeper than two
layers to 512 tokens. The native process refused the 3,663-token request in
memory planning, before loading the real weights. That original failure remains
preserved; no real training update or completed gradient probe was repeated.

The action profile now uses the existing bounded 262,144-node arena for ordinary
Llama's vectorized graph. The older token-times-layer estimate belongs to the
token-unrolled recurrent path. At 42 layers and the maximum 256 selected targets,
both 513-token and 1,024-token native gradients used exactly 5,878 nodes on CPU
and CUDA. Both covered all 381 gradient tensors. Node metadata was bounded at
128 MiB. Numerical tensor-buffer allocation is still measured and admitted for
the actual graph; a small node count does not make a large attention tensor free.

The old short/recurrent profiles keep their previous planning path. The
[additional qualification receipt](../validation/native-action-graph-arena-20261026.json)
records the depth/context tests. This correction does not by itself establish
that a real 42-layer model at every length up to 8k fits one GPU.

## Run the qualification

Use an authorized Linux host and a fresh output directory for each command.
Build with the existing `experimental/native_training/build.py`, its pinned
llama.cpp revision and the inspected FP32-activation dependency. For CUDA,
include `--cuda-out-prod-singletons` after applying the already documented
dependency patch in an isolated tree. Use the actual PCC GPU lease when running
on a GPU shared with other PCC work.

```sh
export PYTHONPATH="$PWD:$PWD/tools:$PWD/experimental/branching_context/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m experimental.native_training.test_llama_action_profile \
  --executable /absolute/path/pcc-native-train \
  --output /private/fresh-cpu-proof --devices CPU
python -m tools.test_native_action_jobs \
  --executable /absolute/path/pcc-native-train \
  --output /private/fresh-service-proof
```

For the CUDA fixture, select the explicitly mapped GPU, set
`GGML_CUDA_F32_ACTIVATIONS=1` and `NVIDIA_TF32_OVERRIDE=0`, and use
`--devices CUDA0`. The fixture is deliberately limited to CPU or one CUDA device.
It uses NumPy for independent reference arithmetic and no tensor/autograd runtime.

```sh
python -m unittest tools.test_llama_action_admission \
  tools.test_native_sequence_profiles tools.test_native_training \
  tools.test_optimizer_profiles tools.test_native_completion -v
```

The machine-independent [qualification receipt](../validation/native-action-profile-20261026.json)
binds source, executable, dependency, result and log hashes. Raw private paths and
operational records are retained on the qualification host rather than published.

## Use it in the existing learning pathway

Select `sequence_profile="dense-llama-action-8k-v1"` and an explicit finite
`max_tokens` no larger than 8,192 in the owner learning-service configuration.
Use the existing allocator profile, real device budgets/reserves, registered
optimizer recipes and immutable candidates. This selects a supported bound; it
does not promise that a particular model fits a particular GPU. Actual persistent
and graph admission still decide that.

Install independently verified practice through S³'s existing lesson registry.
The [S³ BW action guide](https://github.com/CC-David-CC/sss/blob/feat/learning-stack-20261026/docs/BW-ACTION-PRACTICE.md)
describes the original capture, oracle, exact append-only rendering and masking.
Neither a synthetic fixture's model ID nor a hash-shaped evidence field is
production lesson approval.

Enable the existing `completion_recovery` source closure before dispatch so the
saved-model reload check and original completion evidence are recorded. Continue
an unchanged practice stream with its exact saved optimizer state; choose fresh
moments explicitly for a new stream while retaining learned parent weights.

The old `dense-llama-short-v1` stays at 512 tokens. Existing fixed-study settings,
cases, registries and budgets are not migrated. No training-time budget is added
to this new profile. Evaluation, retention decisions, named private GGUF export
and release acceptance remain responsibilities of the composed S³/BW/PCC pathway.

## Real-model resource findings

The [real MiniCPM memory report](MINICPM-ACTION-MEMORY.md) now records a completed
3,663-token full-gradient probe with all 381 tensors finite/nonzero, plus guarded
resource refusals for the longer full/LoRA cases and an allocation-only 8k probe.
No real weights were updated. This is partial allocation/gradient qualification;
the complete longer-context training and learning-quality loop remains unfinished.
