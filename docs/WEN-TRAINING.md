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
adapter is now measured in a bounded numerical diagnostic: one rank-8 final-MLP
down-projection update, a 1024-token prompt and 128 greedy decode tokens. Native
and PyTorch adapter-delta RMS errors are about 0.35% of the measured update RMS;
the regular and strict mixed builds pass exact adapter readback, zero-adapter and restored-prefix
checks. See the cross-runtime report for the distinct absolute errors and scope.
This diagnostic is not mixed-task BW learning or adapter promotion. The production trainer,
version transition and promotion machinery remain pending. This document records the agreed
interfaces and memory strategy, not a claim that Wen has learned these tasks.

See [WEN-PRECISION.md](WEN-PRECISION.md) for the separate activation, TF32 and
attention/cache precision controls and measured cross-runtime differences. The
importer defaults to float32 to retain the source GGUF F32 parameters; selecting
bfloat16 also rounds those parameters.

## Initial mixed self-OPD pilot

A private operational prototype now runs the proposed loop with newly authored,
owned synthetic BW action tasks. It uses the actual BW registry/schema and a
compact controller header, the original Wen template with thinking disabled,
unconstrained greedy student rollouts, and independently checked corrections.
It does not ingest protected workload answers or publish training prompts.

The baseline passes 15/19 cases: 7/9 training, 5/7 held-out and 3/3 preservation.
The four writing cases emit EOS before closing the outer JSON object, and their
paragraphs also miss the requested whitespace-delimited word counts. Production
BW grammar is a separate syntax mitigation, not a measured learning gain here.

Each of two small pilots starts from the immutable source-F32 base and takes
four AdamW updates at learning rate 0.0005. Rank-8, alpha-8 LoRA targets only the
last layer's MLP gate/up/down projections. Microbatch one accumulates gradients
across all nine training cases before each update. The teacher uses unchanged
round-start weights with corrective context; dense teacher-to-student KL is
averaged over valid response positions and then equally over examples. Held-out
and preservation cases provide no gradients or teacher corrections.

The first pilot's teacher wording incorrectly implied that a reference action,
validated in another workspace, had already executed in the current task. The
teacher replied with finish instead of the pending write. Its candidate is not
promoted. A separate correction makes the pending-state distinction explicit:
both teacher checks then produce valid write actions with exactly 37 words.
The prototype now refuses training unless the teacher demonstrates every
observed training repair. Student prompt IDs and baseline results remain exact
between the two pilots.

Both pilots finish at 15/19 under native COW evaluation. Passing skills are
preserved, but none of the four writing cases is repaired. This is a negative
task-learning result for these small pilots, not a claim about larger adapters
or longer training. Fresh rollouts differ across updates; changing per-step KL
means alone do not establish fixed-fixture convergence.

Training used one A100-SXM4-80GB, with peak PyTorch allocation 32,920,940,544 bytes
for this restricted adapter. The exported candidate was transferred to a separate
RTX PRO 6000 Blackwell host and loaded by the regular mixed COW build. All six
GGUF adapter tensors read back exactly equal to the PEFT checkpoint. Each
candidate has its own SHA256 and model/cache identity; the original Wen GGUF is
unchanged. Neither adapter is promoted. Integrated automatic training, version
transition and broader BW adapter qualification remain pending.

Separately, the unadapted model completes two real BW CLI repair/test/finish
workflows with host verification. Temporary serving uses 100 GiB RAM/disk
limits and exits zero with zero ephemeral sessions. These bounded successes do
not establish other BW task families. Training and inference processes release
their GPUs after the runs. Aggregate receipts, excluding prompts and answers,
are in `validation/wen-precision-20261022/bw-self-opd-pilot.json`.
