# Resolve the exact parent before another native update

An inference base hash does not identify a LoRA hat, and a job label does not
prove which weights an operator intended. PCC now has an optional offline
reader that resolves a complete candidate reference against an actual native
version, saved files and, when requested, its optimizer state. It reuses PCC's
existing GGUF reader, COW identity functions and optimizer-state reader.

This is an owner-side command, run on the host holding the files. It neither
loads a model for inference nor allocates a GPU, trains, approves or publishes.
Client requests cannot use this command to supply server filesystem paths.
The result says `RESOLVED`, with `inference_qualified: false` and `promoted: false`.
That establishes an artifact binding, not model quality or resource admission.

## Required identities

A full-weight parent names its exact full GGUF hash. A measured composite also
names the COW model contract and an empty adapter list. A LoRA parent names the
base GGUF, the actual adapter, scale and COW model contract. Native warm start
currently supports one adapter at scale one; it refuses another scale rather
than silently changing the model being continued.

The version file is the content-addressed JSON written by
`tools.pcc_native_training.execute_native`, under that run's `versions` directory.
It must record a completed standard-backprop native candidate. The resolver
checks the actual F32 candidate, its GGUF training-base metadata and the complete
COW identity. It will not silently dequantize a serving model. The original base
is separately pinned for LoRA; full-weight continuation uses the complete trained
weights themselves.

Only explicitly configured owned artifact roots are searched by reference, not
by scanning the filesystem. Paths must be regular files within those roots;
symlink traversal, wrong bytes, missing files and changed inputs are refused.
The command does not guess a replacement for a retired temporary checkpoint.

## Fresh moments and exact resume are different choices

Choose `--state-policy fresh` for a new lesson block. It retains learned weights
or the learned hat, and returns weight fields only. The caller must explicitly
choose the new optimizer recipe. Old moments may have been retired; the command
does not pretend to reuse them.

Choose `--state-policy resume` to continue the same ordered practice stream and
saved optimizer. Supply the exact native example array with `--examples`.
Candidate, state hash, dataset order, cursor, completed-step count and canonical
optimizer configuration must agree. A changed practice stream is refused. The
result includes the exact optimizer request fields for AdamW or the supported
Muon variant. It never silently resets moments to make a resume succeed.

Saved canonical headers contain inactive defaults as well as active settings.
Those headers are not request documents: native AdamW, for example, refuses
Muon-only input knobs. The resolver sends only the applicable settings, checks
that omitted inactive fields have their known defaults, and leaves the native
loader's full canonical-state comparison in place.

```bash
python -m tools.pcc_candidate_parent \
  --reference /owned/selected-reference.json \
  --version /owned/run/versions/CANDIDATE_SHA.json \
  --profile /owned/qualified-dense-template.json \
  --artifact-root /owned \
  --state-policy resume --examples /owned/native-examples.json \
  --output /owned/new-parent-resolution-private.json
```

For LoRA, also supply `--base-model /owned/base.gguf`. The identity template must
be the existing dense FP32 COW profile, with its adapter capability where needed.
Using it here does not qualify a build, observe a GPU or grant permission to use
that GPU. Ordinary native execution and inference admission remain mandatory.

## Use a verified milestone after cache retirement

Add `--checkpoint /owned/milestone/checkpoint-verification.json` to use a durable
copy made by PCC's existing checkpoint copier. The receipt must identify the
same candidate, optimizer and completed step. Every file used is checked against
its registered hash and byte count. Fresh continuation needs the copied weights;
resume additionally requires the copied moments. No original manifest or model
is rewritten, and files remain on their host.

Resolution receipts contain private absolute paths. They are created exclusively
with mode 0600; an existing output is refused. The short stdout summary contains
an identity and status, not the full paths. Preserve the receipt privately and
publish only reviewed aggregate evidence.

## Bind a registered job at the API boundary too

Both BW and COW CLIs now accept the optional `--expected-parent-sha256` together
with `--parent-job-id`. PCC compares that hash with the completed registered
parent before queuing, then again before its worker starts. The hash means the
full trained model in full mode, and the adapter in LoRA mode. A base hash alone
cannot pin a hat. Unknown or wrong parents cannot create a new training job.

This guard carries no paths and does not approve a lesson or a model. Omit it
and existing request fields and default execution remain unchanged. Existing
operation IDs still bind the exact original request: retry the original command
or inspect its status; do not issue a new update after an uncertain response.

An offline study candidate is not automatically a registered training-service
job. This reader does not invent that job ID or import arbitrary files into a
service. Owner-side offline execution can consume the resolved fields through
the existing admission/executor; service clients need an actual completed job.

Native/offline LoRA AdamW state is covered by this reader. The existing service's
registered adaptive-optimizer profiles are still full-mode policies; its default
LoRA client path remains SGD. File resolution does not extend that registration
policy or silently select a different optimizer.

## Evidence and limits

The CPU native controls compare uninterrupted and resolved-resume weights and
moments for full AdamW, LoRA AdamW, standard Muon, RMS/decay Muon and fast-to-refine
Muon. They also exercise fresh lesson blocks, wrong hats/contracts/data/cursors,
missing moments, verified milestone relocation, symlinks and output reuse.
These use generated tiny ordinary-Llama fixtures, not language-model scores.

The optional S³ integration uses real PCC HTTP and both CLIs for six native jobs,
while its semantic scores are explicitly controlled injections. It proves parent
binding, decision plumbing and duplicate prevention, not learning or forgetting.
The ongoing MiniCPM study supplies separate measured behavior evidence.

See [S³'s proposal-to-artifact guide](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/RESOLVED-LEARNING-PARENT.md)
and [BW's guarded client commands](https://github.com/CC-David-CC/BranchWrite/blob/feat/native-learning-path-20261024/docs/PINNED-TRAINING-PARENTS.md).


### Actual saved MiniCPM candidates

The readonly A100 exercise resolved update 16 for both full AdamW and attached
LoRA AdamW. Both temporary candidate files had already been retired. Explicit
verified milestone receipts selected the durable copies; each state identified
16 completed updates and cursor 432 on the original ordered practice stream.
The attached hat's computed COW contract matched its previously recorded scored
contract exactly. The full series retained its original weight-only reference;
no stronger measured binding was invented for that legacy evidence.

This made no native update, inference request, model transfer or GPU allocation.
The full-candidate read/hash check took 282.67 seconds; the LoRA check took 6.76
seconds. They read different artifact sizes under different cache/shared-storage
conditions, so these are not a controlled optimizer or training-speed comparison.
Cold artifact verification is a real cost for the operator loop; weakening the
hash checks is not the demonstrated solution. The same resolver logic was used
in the real-model and final CPU checks; the final CLI's separate input-freeze
check was exercised by the CPU command controls.

[26 native CPU controls](../validation/parent-binding-20261025-ay/native-controls.json)
and [actual MiniCPM reference/state bindings](../validation/parent-binding-20261025-ay/actual-model-binding.json)
retain outcomes and source hashes without private artifact paths.
