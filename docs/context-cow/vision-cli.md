# Image contexts through the COW CLI

This is an opt-in Q4 prototype on the pinned Linux/CUDA, single-GPU execution
profile. It prepares an image-containing context, then reuses that context for
independent questions, including questions with new images. It also supports continuing from a compatible disk
checkpoint after the inference server exits. Read the current
[vision evidence](../../experimental/branching_context/evidence/STATUS.md) for
executed results and limitations. This is not G1-G5 certification.


Retained original images are fully integrity-checked on reuse. Cached reuse
streams those checks through admitted RAM scratch instead of keeping another
full original in memory. Re-encoding admits the complete original input before
reading it; insufficient capacity returns a clear error before encoder work.
This bounds these accounted buffers, not every allocation in the process.
See VISION-ORIGINAL-MEMORY-2026-10-17.json in the evidence directory for executed
read-size, cleanup and Q4 reconstruction regressions. Greedy image GBNF now
passes bounded core and CLI/HTTP fork, continuation and restart tests.
Basic sampled image controllers also pass bounded CLI/HTTP restart tests:
temperature, Min-P, Top-K, Top-P, seed and explicit reseed, with optional GBNF.
See VISION-SAMPLING-CLI-2026-10-17.json for the exact client scope.
Image-aware DRY also passes bounded CLI/restart and real-penalty GPU checks.
See VISION-DRY-2026-10-17.json for exact results and remaining limitations.
Typical and DynaTemp, separately and together, also have bounded real-Q4
CLI/restart evidence. Image-aware DRY can now be combined with either or both
on the checked mixed-text V4 path. See VISION-ENTROPY-2026-10-17.json and
VISION-DRY-ENTROPY-2026-10-17.json for the exact fixtures and remaining limits.
These controller campaigns used their recorded profiles. Later renderer and
performance changes require their own integration checks; an earlier sampler
pass is not full qualification of every later source profile.

## Start the inference server

Use the existing qualified model/profile and compiled COW and vision executables.
Run these commands in the repository on the Linux inference host. Windows is a
client/driver; the experimental store currently requires POSIX locking.

Set `COW_PYTHON` to the Python interpreter in the server environment. Set
`COW_API_KEY` to a private key you choose. Do not commit that key.

The server administrator supplies `vision.json` with these fields. Replace the
paths and hashes with the actual qualified files; the server checks all three
pins before starting GPU work. The model-file pin must name the exact
`model_path` from the execution profile, including its absolute path.

```json
{
  "schema": 1,
  "executable": "/absolute/path/to/strata-vision",
  "projector": "/absolute/path/to/mmproj.gguf",
  "embedding_width": 2560,
  "threads": 32,
  "max_rows": 1024,
  "original_bytes_limit": 268435456,
  "pinned_sha256": {
    "/absolute/path/to/strata-vision": "REPLACE_WITH_FULL_SHA256",
    "/absolute/path/to/mmproj.gguf": "REPLACE_WITH_FULL_SHA256",
    "/absolute/path/to/model-metadata-shard.gguf": "REPLACE_WITH_FULL_SHA256"
  }
}
```

The profile must already declare `GPU=[0]`, the mixed-input resume contract and
the matching embedding width. A text-only profile cannot be turned into a vision
profile merely by supplying this configuration. The 256 MiB limit above covers
retained originals and their artifact/grant metadata, separately from the derived
RAM/disk checkpoint pools. Configure it for the workload's retained images.

```sh
export PYTHONPATH=.:experimental/branching_context/src
"$COW_PYTHON" -m serve.cow_http \
  --profile /absolute/path/to/qualified-profile.json \
  --store /absolute/path/to/cow-store \
  --text --vision-config /absolute/path/to/vision.json \
  --api-key "$COW_API_KEY" --host 127.0.0.1 --port 8099 \
  --ram-gib 16 --disk-gib 32 --checkpoint-idle-seconds 1800
```

Those pool sizes are examples for the bounded demonstration, not recommended
limits for a large deployment. Keep the model/encoder headroom and residency
settings from the qualified profile. The same worker serializes image encoding
and model computation. An upload validates bytes but does not run the encoder.

Connect a remote client through an SSH tunnel, or a separately configured HTTPS
endpoint. The CLI refuses plaintext HTTP to remote hosts. For example, run this
on the client and leave it connected:

```sh
ssh -N -L 8099:127.0.0.1:8099 llm-60
```

## Prepare a shared context and ask questions

Use the same `COW_API_KEY` in the client environment. In the commands below,
`NAMESPACE_ID` is the namespace identity returned by the first command.

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  namespace --title "Diagram questions" --operation-id diagram-namespace

"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  context --namespace-id NAMESPACE_ID --branch diagrams \
  --image ./diagram.png --text "Keep this diagram as shared context." \
  --operation-id diagram-context

"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  fork --source-branch diagrams --name second-question --operation-id diagram-fork

"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  ask --branch diagrams --text "Describe the left side." \
  --max-tokens 128 --operation-id diagram-left

"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  ask --branch second-question --text "Describe the right side." \
  --max-tokens 128 --operation-id diagram-right
```

`context` saves the beginning without starting an answer. Fork before asking if
both questions must start from that same beginning. `prepare` instead includes
the initial assistant frame; use `answer` after it. A question's output limit is
an upper bound; a model can stop earlier.

The `context`, `prepare`, `ask` and `append-user` commands accept repeated
`--image` to place multiple images before the text. For exact interleaving,
use `--parts-file parts.json` instead of `--text` and `--image`:

```json
[
  {"type": "text", "text": "First diagram: "},
  {"type": "image", "file": "diagram-a.png"},
  {"type": "text", "text": " Compare it with "},
  {"type": "image", "file": "diagram-b.jpg"},
  {"type": "text", "text": " Keep both as shared context."}
]
```

Image file paths in this document are relative to the parts file. The CLI reads
them locally, computes full SHA-256 identities and uploads the bytes. The server
receives immutable image IDs in the preparation request, never client paths or
URLs. `upload-image --namespace-id NAMESPACE_ID --file diagram.png` is available
for inspecting a standalone upload receipt.

## Add another image to a conversation

After the model has completed its answer, send the next question with a new
picture. If the previous answer reached its output limit while still active,
use `continue` first. A new user turn requires a completed answer.

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  append-user --branch diagrams --image ./updated-diagram.png \
  --text "What changed in this diagram?" --max-tokens 128 \
  --operation-id diagram-update
```

For the first question after a saved context, use `ask` with the same image
options. Use `--parts-file` instead for text/image/text order. The CLI obtains
the upload namespace from the selected branch's head; no conversation download
or second namespace argument is required. The server computes only the new
input and generated output when the prefix checkpoint is available.

The final answer's `input_preparation` receipt lists accepted image identities,
actual rows, encoder calls and preparation work. Preparation itself performs
zero language-model forward tokens; the answer's `counter_deltas` reports the
new input and generated-token work. These are different stages.

## Constrain an image branch with a grammar

The server administrator can add `--grammar-registry /absolute/path/registry.json`
to the server command. The file defines the permitted named grammars using the
[registry format](../../experimental/branching_context/docs/grammar-registry-v1.md).
Clients select a name; they do not send grammar source or server file paths.
The following example assumes the registry defines `EDIT` and `TEST`.

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  context --namespace-id NAMESPACE_ID --branch grammar-demo \
  --image ./diagram.png --text "Keep the diagram and code as context." \
  --operation-id grammar-context

"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  ask --branch grammar-demo --text "Propose a small patch." \
  --grammar-state EDIT --max-tokens 4 --operation-id grammar-edit

"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  continue --branch grammar-demo --max-tokens 32 --operation-id grammar-edit-tail

"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 --machine-result \
  append-user --branch grammar-demo --text "Propose the test action." \
  --grammar-state TEST --max-tokens 32 --operation-id grammar-test
```

`continue` resumes the saved parser without requiring the state name again.
If the output limit leaves the response active, continue it before adding a
new user turn. A new completed-turn question can select another grammar and
can include a new image. Forking a partial response copies its parser progress;
finishing one branch does not advance another branch's parser.

Restart with the same registry definitions, tokenizer and execution profile.
A saved model checkpoint can resume without its embedding file or an encoder.
Changed definitions cannot resume the old parser. Exact `retry --from-operation`
replays the committed answer. Sampled settings and seed/reseed are described
below; image text-stop controls remain outside the supported scope.

Grammar rules constrain output syntax. The host still validates the proposed
action, checks permission and decides whether to execute it. The validation
campaign generated tool-shaped JSON; it did not execute those proposals.

## Restart, retry and expiration

Wait for an acknowledged durable completion before stopping the server with
Ctrl+C. Restart with the same store, model and execution profile. You may omit
`--vision-config` for a restore-only server: its mixed renderer still loads, but
it accepts no new uploads and starts no encoder. A compatible image checkpoint
can continue even after its derived embedding files have expired.

If every numerical checkpoint is gone, ordinary questions explicitly report
that the saved prefix is unavailable and must be recomputed. Cached embeddings
can be used without an encoder. If those have expired too, restart with the same
pinned `--vision-config` so retained original images can be re-encoded. Their
normalization, embedding hashes and grids must match before model computation.
Without the required producer, the request fails and retains the old branch.

Reconstruction preserves saved inputs/controller records but uses a new batch
arrangement. It does not recover the old numerical computation. The final
`source_recomputation` record separates paid prefix cells, text tokens, image
rows and encoder work from the new question/output work. Strict checkpoint
restore remains a separate operation that fails when its checkpoint is absent.

Use a new operation ID for new work. Use `retry-request --operation-id ORIGINAL`
for the original journaled request after a lost response. A completed local retry
does not need the original image file. The journal contains the exact submitted
prompt and immutable IDs; protect it as private client data. A completed server
operation can replay its image question after a restart without its upload grant,
embedding payload or producer. It validates the supplied message against the
recorded input spans. An upload itself is
content-addressed: if its response is lost, re-uploading the same bytes refreshes
the same namespace grant. It does not create a new conversation operation.

Pending preparation can require re-upload after an unused grant expires. Re-upload
the original bytes before retrying it. Changing a local file creates a different
image identity; it cannot rebind an already journaled operation. An upload receipt
acknowledges retained input bytes, not a computed model checkpoint.

## How the client uses COW

The core Python API provides
`model.feed_inputs(snapshot, new_spans, artifacts=owner)`. It consumes only newly
supplied inputs and returns an immutable child snapshot. The owned greedy request
also accepts an `InputHistory` delta and artifact owner. Old image embeddings are
not needed when the saved numerical prefix is available. These core methods do not by
themselves move a branch or acknowledge durable publication; use the existing
revision-checked save/publish operation for that boundary. See
[the executed core-delta report](../../experimental/branching_context/evidence/VISION-INPUT-DELTA-2026-10-17.json).

The client `ask` and `append-user` image controls use worker-owned preparation
and the coordinator's typed queue/publication path. Image leases stay held
through publication. Completed retries validate full identities and cell counts;
they do not re-encode images or regenerate an answer. Pending work returns an
explicit recovery instruction rather than admitting a duplicate encoding.

## Sampling an image branch

After `prepare`, select settings for one generation. For example:

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  answer --branch diagrams --max-tokens 4 \
  --temperature 0.70 --min-p 0.05 --top-k 0 --top-p 1.0 \
  --seed 42 --operation-id sampled-diagram
```

If the saved controller remains active, `continue` preserves its RNG draw
cursor and parser progress. It can select different settings for that generation:

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  continue --branch diagrams --max-tokens 4 \
  --temperature 0.90 --min-p 0.05 --top-k 60 --top-p 0.95 \
  --operation-id warmer-diagram
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  reseed --branch diagrams --seed 271 --operation-id reseed-diagram
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  continue --branch diagrams --max-tokens 4 --operation-id reseeded-tail
```

Use `reseed` to change the active scope's random stream. It preserves consumed
inputs, grammar progress and other controller fields, and resets the draw
cursor. Forks retain independent progress. A new operation ID means new work;
replaying a completed operation returns its saved answer. Default generation
remains greedy. A stopped turn needs a new user turn rather than `continue`.

Optional `--grammar-state NAME` selects a configured server-owned grammar on
`answer`, `ask` or `append-user`; continuation restores the active grammar.
Both grammar and plain sampling require the actual pinned native sampling
capability. Tool-shaped output is a proposal for the host to validate/execute.

The two Q4 sampling campaigns execute 36 CLI commands and compare 52 full-logit
vectors exactly, ten controllers and 191 independent CPU sample choices. They
check durable request frontiers, not every streamed frontier, and do not measure
serving throughput.

### Typical and dynamic temperature

For an active response, this example combines Typical with DynaTemp for the
next generation. It keeps the branch's saved RNG and parser progress:

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  continue --branch diagrams --max-tokens 128 \
  --temperature 0.70 --min-p 0.05 --top-k 0 --top-p 1.0 \
  --typical-p 0.95 --dynatemp-range 0.35 --dynatemp-exponent 1.0 \
  --dry-multiplier 0 --operation-id combined-diagram
```

For Typical alone, set `--dynatemp-range 0`. For DynaTemp alone, set
`--typical-p 1`. A value of `--typical-p 1 --dynatemp-range 0` disables both.
The parameters stay fixed during that generation. Omitted settings on a later
`continue` inherit the saved policy. A completed response needs a new user turn.

These image settings require a worker that actually reports its supported V4
command, not only a client-side setting. DRY must be disabled when using these
image entropy filters. This restriction is checked before model work. The host
still owns grammar-state selection, semantic validation and tool execution.

Changed-policy retries also pass separate tiny and approximately 1K Q4 GPU
campaigns: 44 CLI commands, 116 exact full-logit comparisons, 18 controller
checks and 388 independent reference sample choices. For example:

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  retry --from-operation sampled-diagram --branch diagrams \
  --temperature 0.80 --min-p 0.05 --top-k 60 --top-p 0.95 \
  --seed 411 --operation-id another-diagram-answer
```

This generates from the original operation's source and inputs; it does not
append the preceding answer. It retains the original immutable image identities
without needing a new upload or local file. Cached embeddings require no
encoder. Expired embeddings are regenerated on the serialized GPU worker using
the pinned producer and checked against their original hashes and grids. With
no compatible producer or a changed embedding identity, the request fails and
keeps the old branch. Preparation receipts report actual encoder work. In the
tested runs, the retained model prefix was reused; only the question delta and
new outputs were evaluated. Normal explicit-reconstruction rules still apply
if that model checkpoint is also gone. These changed retries qualify the new
sampled mixed-input descriptors; legacy greedy typed descriptors 9/11 remain
outside changed-controls support. See
[the retry evidence](../../experimental/branching_context/evidence/VISION-SAMPLING-RETRY-2026-10-17.json).

### XTC with optional Typical and dynamic temperature

For an active image response, enable XTC for the next generation:

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  continue --branch diagrams --max-tokens 128 \
  --temperature 0.90 --min-p 0.02 --top-k 0 --top-p 1.0 \
  --xtc-probability 0.12 --xtc-threshold 0.15 \
  --dry-multiplier 0 --operation-id xtc-diagram
```

Add `--typical-p 0.95 --dynatemp-range 0.35 --dynatemp-exponent 1.0`
to combine these filters. Set `--xtc-probability 0` to disable XTC. As with
other controls, a later `continue` inherits omitted settings; `retry` with
changed settings generates from the original operation's source and inputs.

Image XTC requires the worker's checked native V5 proposal command. DRY,
Mirostat 2 and Adaptive-P combinations remain rejected. XTC can consume one
or two random draws for one output token. The saved branch controller retains
the exact cursor across fork, retry and restart; output length is not a random
draw count. Host permissions and tool execution remain separate from sampling.

The bounded Q4 XTC campaign covers XTC alone and XTC with Typical/DynaTemp,
including a prefix with 1,028 text/control tokens and 49 image rows. Its three
CLI runs pass 168 exact full-logit comparisons, 27 controller checks and 583
independent sample decisions. A separate read-only native check passes 18
choices and two additional exact frontier comparisons. These are correctness
audits at saved request boundaries, not serving-speed measurements or a promise
for every sampler combination. See
[the XTC evidence](../../experimental/branching_context/evidence/VISION-XTC-2026-10-17.json).

### Mirostat 2 feedback

Mirostat 2 keeps a feedback value for each branch. Continuing restores that
value with the branch's RNG and grammar progress. Forking copies its current
value; progressing one fork does not advance another fork's feedback.

For an active image response:

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  continue --branch diagrams --max-tokens 128 \
  --temperature 0.80 --mirostat 2 --mirostat-tau 4 --mirostat-eta 0.05 \
  --min-p 0 --top-k 0 --top-p 1 --typical-p 1 --dynatemp-range 0 \
  --xtc-probability 0 --dry-multiplier 0 --operation-id mirostat-diagram
```

This mode uses its own probability filter. The explicit neutral settings above
disable filters that may have been inherited from an earlier generation. The
worker must report the native V6 Mirostat command and versioned feedback
capability. Image DRY combinations remain rejected. Reseeding changes the RNG;
it preserves the existing feedback and grammar progress.

Tiny and 1K-text Q4 CLI campaigns pass 116 exact full-logit comparisons,
18 controller checks and 388 independent sample decisions. A separate read-only
GPU probe adds 18 feedback choices and two exact frontier comparisons. These
checks include actual settings changes, fork, cold continuation, retry and
image re-encoding. They measure saved request boundaries, not every streamed
frontier or serving speed. See
[the Mirostat evidence](../../experimental/branching_context/evidence/VISION-MIROSTAT-2026-10-17.json).

### Adaptive-P feedback and combined filters

Adaptive-P keeps a running probability estimate in the branch controller.
Continuing restores that estimate with the branch's RNG and grammar progress.
Each fork has its own copy.

For an active image response:

```sh
"$COW_PYTHON" -m tools.cow_cli --url http://127.0.0.1:8099 \
  continue --branch diagrams --max-tokens 128 \
  --temperature 0.70 --min-p 0.05 --top-k 0 --top-p 1 \
  --adaptive-target 0.55 --adaptive-decay 0.90 \
  --mirostat 0 --xtc-probability 0 --dry-multiplier 0 \
  --operation-id adaptive-diagram
```

Add `--typical-p 0.95 --dynatemp-range 0.35 --dynatemp-exponent 1`
to combine the entropy filters. Adaptive-P also supports the existing Min-P,
Top-K and Top-P controls. Mirostat 2 and active XTC cannot be combined with it.
Image DRY combinations remain rejected. The worker must report the checked
native V7 command and versioned feedback capability.

Changing the target or decay during `continue` preserves progressed feedback.
Reseeding preserves feedback and grammar progress while resetting the RNG.
Use `--adaptive-target -1` to disable Adaptive-P. Re-enabling it starts its
initial feedback state and preserves the branch's current RNG cursor.

The Q4 image campaigns cover Adaptive-P alone, the combined filters, and a
combined prefix with 1,028 text/control tokens plus 49 image rows. They pass
177 exact full-logit comparisons, 27 controller checks and 603 independent
reference choices. A separate read-only native probe passes 24 choices and
two additional exact frontier comparisons, including actual filter removal
and both target-clamp boundaries. These are saved-request correctness audits,
not every streamed frontier or serving-speed measurements. See
[the Adaptive-P evidence](../../experimental/branching_context/evidence/VISION-ADAPTIVE-2026-10-17.json).

## What has actually been checked

The [supported-case matrix](vision-supported-matrix.md) separates available
features, executed examples and work still needed. The latest Q4 thinking flow
passes first image questions, later repeated-image questions, exact request
retries and a new server process. Its 14 full-vocabulary comparisons have zero
error against checkpoint-free references or the uninterrupted control. The
final answer is constrained to `OK`; this tests state handling, not image-answer
quality. See [the executed flow](../../experimental/branching_context/evidence/THINKING-NEW-IMAGE-FLOW-Q4-2026-10-17.json).

Use `ask` for the first question on an idle saved context. Use `append-user`
after an answer has completed. Use `continue` when that answer is still active.
The first and later image questions now share the same suffix-rendering path
as exact retries. This prevents repeating the system instructions on a new
image turn. Older profile `8e7fa4f` failed that check; corrected profile
`92df1837` has the executed passing flow. They are not interchangeable.

The current orientation/order matrix also passes all eight EXIF orientations in
tiny RGB PNG and JPEG inputs, plus both orders of two images. Its36 raw-reference
prefix/final vectors match exactly and18 saved-parent byte checks remain equal.
This is geometry and state handling, not an image-answer-quality benchmark.
Alpha/non-RGB and larger-image coverage remains separate. See
[the format evidence](../../experimental/branching_context/evidence/VISION-FORMAT-MATRIX-Q4-2026-10-18.json).

## Current boundaries

The bounded Q4 client campaign executed 11 CLI commands, with first/later image
questions, a new server process and producer-free replay. Eight full-vocabulary
frontiers matched independent references exactly. Answers stopped at two tokens
against an eight-token limit. This is a correctness example, not a speed or
long-context qualification. See [the client-suffix evidence](../../experimental/branching_context/evidence/VISION-SUFFIX-CLI-2026-10-17.json).

- PNG/JPEG only: up to eight images, 16 MiB and 16 million decoded pixels per
  image, with a qualified encoder cap of 1,024 rows per image.
- New images in first and later questions require an existing mixed context.
  Adding the first image to a legacy text-only root is not supported yet.
- Greedy and temperature/Min-P/Top-K/Top-P image continuation, optional named
  GBNF, seed/reseed and bounded image-aware DRY. DRY requires a profile declaring
  the checked native mixed-text-runs-dry-u53-v1 protocol; older workers reject
  it. DRY counts real text and cannot match a sequence through an image.
- Typical and DynaTemp work separately and together on the checked native V4
  path. Combining them with DRY requires the worker's checked mixed-text V4
  family; the original V4 command alone is insufficient. Bounded CLI/restart
  and independent selection evidence exists for DRY+Typical, DRY+DynaTemp and
  all three together. DRY with XTC, Mirostat 2 or Adaptive-P remains rejected.
- XTC works on the checked V5 proposal path, including qualified combinations
  with Typical/DynaTemp and DRY disabled. Mirostat 2 uses the checked V6 feedback
  path with DRY disabled. Adaptive-P uses the checked V7 feedback path, including
  supported probability and entropy filters with DRY disabled. Image text stops
  remain rejected pending image-aware integration and tests.
- Serving reconstruction after all numerical checkpoints are evicted has bounded
  Q4 evidence. Original-read buffer admission is fixed and tested; process-wide
  peak RAM and sustained pressure remain unqualified. See
  [the original-memory report](../../experimental/branching_context/evidence/VISION-ORIGINAL-MEMORY-2026-10-17.json).
- Exactness applies to the same model/build/profile and native execution
  boundaries. Different arrangements are not promised equivalent.
- No image web-interface controls, Q8 vision qualification or split-GPU vision
  qualification are supplied by this delivery. The existing health response's
  general image capability remains off while the wider integration is unfinished.
