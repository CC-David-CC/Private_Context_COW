# Wen training preparation

The original Wen GGUF remains immutable. The optional importer in
`tools/wen_training_import.py` reconstructs a separate trainable checkpoint from
its own BF16/F32 tensors. It requires the exact source SHA256 and preserves its
custom vocabulary, merges and chat template. It does not substitute upstream
Qwen weights or tokenizer files. It registers the Wen aliases only inside its
own process against the pinned GGUF importer dependencies.

Install `requirements-wen-training.txt` in a separate training environment;
ordinary COW serving does not require PyTorch or PEFT. Import admission tests
check geometry, source types and vocabulary/template identity. These checks do
not establish forward numerical agreement with the native COW backend. The
import receipt explicitly reports that numerical comparison remains pending.
No training or adapter promotion is performed by the importer.

## Agreed learning arrangement

BW collects actual text/code task attempts, observed failures and independently
verified corrective context across several skill families. Infrastructure errors
are engineering evidence, not model training cases. Training and held-out
variants remain separate. Passing tasks are retained as preservation checks.

The intended self-OPD update uses fresh student rollouts and a frozen round-start
Wen teacher given verified corrective context, with dense teacher/student
distillation on aligned response positions. A corrected-answer-only supervised
loss must not be labelled self-OPD.

The local target is inference on a 24 GB GPU host and LoRA backpropagation on a
separate 96 GB GPU host over LAN. Requests, evidence, model identities and
versioned adapter artifacts cross the network. Their VRAM is not a combined
pool. On a shared A100 host the same process roles can use separate devices.
Actual Wen memory use and the 24 GB inference context limit require measurement.

Start training with microbatch size one and accumulate gradients across a mixed
group before one optimizer update. Each example's activation graph is released
after backward; retain parameter gradients and optimizer state, not all example
graphs. Bound sequence length separately. The loss normalization must reflect
the chosen example/skill weights and valid response-token counts, including a
partially filled accumulation group. More A100 memory can allow a larger measured
microbatch without changing the intended objective. Do not assume accumulation
and a padded large batch are equivalent without testing normalization.

Inference stays pinned to its current immutable adapter during a training round.
Validate the candidate against repairs, held-out variants and preservation tests
before publishing a new adapter version. Keep the original GGUF; any later
merged model is a new versioned file. Old prefills are not regenerated at
promotion. On the next request, including one in an old chat, a newer selected
adapter requires a new compatible prefix materialization; old KV state is not
relabelled as compatible. In-flight requests remain pinned to their old version.

The optional native LoRA attachment now compiles in both dense adapter variants,
with immutable adapter-byte/scale admission controls. Loading a real trained
adapter, its numerical effect and restore behavior remain unqualified. The
trainer, version transition and promotion machinery remain pending. This document records the agreed
interfaces and memory strategy, not a claim that Wen has learned these tasks.

See [WEN-PRECISION.md](WEN-PRECISION.md) for the separate activation, TF32 and
attention/cache precision controls and measured cross-runtime differences. The
importer defaults to float32 to retain the source GGUF F32 parameters; selecting
bfloat16 also rounds those parameters.
