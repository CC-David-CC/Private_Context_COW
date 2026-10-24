# Supported continuous native learning: design and implementation plan

Owner direction, 2026-10-24: make the learning loop a supported pathway a person
or agent can drive, preserve useful working models, and keep iterating. This
page is a design contract and implementation checklist, not a claim that every
route below exists. No additional repository or external training framework is
needed. Model files stay local and unpublished unless the owner later changes
that instruction.

## One product boundary per responsibility

| Project | Owns | Consumes |
| --- | --- | --- |
| S3 | Skill graph, independently checked lessons, practice/review selection, test suites, paired learning/forgetting reports and proposed next blocks | Host model/evaluation identities and completed outcomes |
| PCC, built on modified Strata | Model/weight/context identities, native execution, resource admission, optimizer/checkpoint state, durable run/round receipts and named GGUF export | Admitted lessons and frozen S3 policy/results |
| BW | User/agent task workflow, tools and an ergonomic learning-loop driver | PCC capabilities/status/operations and S3 reports |

S3 remains usable without BW, PCC, NumPy, a GPU or a network. PCC does not copy
S3's skill definitions or graders. BW does not implement a second training loop
or optimizer. A technical human, BW, or another agent uses the same PCC pathway.
The operation/state boundary extends the existing LessonService and NativeRunner,
not a second workflow engine with another account of which job is running.

## Standing cycle and distinct decisions

**test → choose lessons → train → test → report → preserve a named GGUF → repeat**

Retain a substantially improved working checkpoint when a smaller set of earlier
successes regresses. Put those failure families into the next mixed block with
retention review and continuing learning. Record the original outcomes, both
gains and losses; a strict gate's historical result is not rewritten. Working
parent selection, broad qualification and public release are separate decisions.
A checkpoint can be useful for the next experiment without being described as
universally better. Never copy protected evaluation questions/answers into lessons.

The baseline comparator, previous working parent and frontier checkpoints stay
addressable. Every round reports changes against the original baseline AND its
immediate parent. Avoid silently walking backward over many small regressions.
Keep per-skill/gate best-known and current scores, cumulative losses, uncertainty,
coverage gaps, token/compute work and training lineage. Repeated development
feedback is adaptive selection, so sealed qualification remains an independent
measure; do not call the development score an unbiased final benchmark.

## Shared native engine, with an independent check

Inference and backprop should use the same model definition, tensor roles,
tokenizer/template, operators, numerical profiles, placement and weight-version
contract. Backprop differentiates that forward graph; AdamW/Muon update supported
tensors in native code. No PyTorch/autograd training path is introduced.

Today the native GGML trainer and inference runtime share pinned native backend
machinery but still have separate graph builders. This is qualified experimental
support, not completed unification. Move toward shared ordinary dense model graph
components incrementally, preserving forward/gradient parity and architecture
routing. Then extend supported architectures rather than pretending a dense
MiniCPM experiment establishes MoE or all hardware support. Retain the separately
implemented oracle as a validation tool. Sharing production math and keeping an
independent numerical oracle serve different verification purposes.

A learned weight version is immutable. Existing inference/context/KV state stays
bound to the weights that produced it; a new candidate does not reinterpret an old
cache under changed weights. Activation, optimizer, adapter and cache identities
remain distinct. Test concurrent serving/training only under explicitly admitted
resource placement. One linked native engine does not require unbounded shared
mutable weights or one inseparable process.

## Durable operation lifecycle

Use the existing PCC operation-ID, store lock, worker/lease and immutable candidate
registration machinery. Add versioned learning-run records with stable round IDs:

`created → baseline_pending → choosing_lessons → lessons_verified → training →
checkpoint_saved → evaluating → reporting → preserved → next_block`

Expose `waiting_for_resources`, `needs_operator`, `failed` and terminal budget/owner
stop states explicitly. Infrastructure uncertainty is not a model failure. Each
phase records intent before effects, native child PID/start identity, exact input
and source hashes, output receipts, actual allocations and terminal status.
Reconnection reconciles the same phase and owned child; it never repeats an
uncertain optimizer update because an HTTP/SSH observation timed out.

A different lesson block changes the optimizer dataset identity. Continuing the
WEIGHTS is allowed, but moments/cursors must not be silently resumed under changed
examples. Define and record either a qualified curriculum-transition policy or an
explicit fresh optimizer on the new block. Existing exact same-dataset completed
resume remains a separate supported case. Training is ordinary end-to-end
backprop; passing gradients between sections remains experimental.

## Human and agent contract

Extend the existing authenticated `/cow/training` capability/jobs surface. The
following loop-specific operations are PROPOSED and must not be advertised as
implemented until the acceptance checks below pass:

- Discover owner-configured model, optimizer/resource and learning-policy IDs.
- Start a run with an operation ID, model/working-parent ID, policy ID and budget.
- Read run/round status, pending lesson attestations, gains/losses, errors and costs.
- Submit independently verified lessons through the existing lesson boundary.
- Request the next block, reconcile, stop after a completed update, or preserve a
  named checkpoint using the same operation-ID rules.

Expose the same operations in `python -m tools.pcc_learning_loop` with structured
JSON output and human-readable summaries. BW then calls that contract; it does
not shell out to operator scripts with arbitrary client-supplied paths. Remote
agents select registered IDs, not executables, environment variables, GPU grants
or unchecked checker code. Owner configuration grants resources and verifiers.
The CLI module/routes above are planned, not present at this checkpoint.

## Implementation order and evidence required

1. **Round policy and next practice block (S3).** Separate working-parent decisions
   from strict preservation outcomes. Translate aggregate failure families into
   checked practice selection and mixed review, with no protected-answer export.
   Tests must cover net progress with losses, no progress, cumulative regression
   tracking, changed evaluator identity and retained coverage.
2. **Native optimizer path through PCC's existing runner.** Expose named owner
   AdamW/Muon profiles and whole-update budgets, pinned optimizer state and explicit
   fresh-state transitions for new lessons. Preserve existing SGD/default behavior.
   Test real tiny native updates, reload, exact resume and refusal cases.
3. **Durable loop API/CLI.** Compose existing lesson operations and S3 policy through
   PCC-owned run/round records. Exercise multi-round CPU fixtures, interrupted
   observation, child reconciliation, resource refusal and idempotent export. No
   mocked successful GPU result can qualify this layer.
4. **BW driver and handoff.** Add discover/start/status/next/preserve controls and
   documented workflows using the same API. Exercise a human CLI and a lower-cost
   agent following only the guide. Preserve BW's task-ledger and tool-grant scope.
5. **Real-model repeated loop.** Run mixed repair from the improved MiniCPM parent,
   evaluate the frozen suites, report both comparators, preserve named F32 GGUFs,
   and repeat within the standing resource budget. Keep the current frozen
   four-optimizer comparison unchanged and publish its independent results.
6. **Native graph consolidation and wider skill coverage.** Reduce production
   forward-definition duplication under numerical/gradient regression checks.
   Expand coding/admin/tool/research and context curricula only with real tests;
   public-source data is pinned, verified, attributed and decontaminated first.

At each step write reproducible commands, exact versions, ownership, recovery
instructions and limitations. A stub is a visible unfinished interface, not a
supported feature. Progress is checked code and measured behavior; completion of
this design is not completion of the ongoing learning goal.

## Current measured checkpoint

MiniCPM's first full AdamW control improves exact answers 59/864 to 482/864 and
natural completion 415/864 to 864/864, with 434 gains and 11 lost earlier successes.
It remains a useful working parent. The standalone F32 GGUF is retained locally
with notes. Native standard Muon and the two frozen variants are running next.
[Current S3 evidence](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/MINICPM-NATIVE-LEARNING.md).
The full supported loop/API and shared graph consolidation remain unfinished.


The [quantization-retention study](MINICPM-QUANTIZATION-STUDY.md) is an approved
part of export qualification. GGUF creation and successful loading do not prove
that a smaller deployment format preserves the learned behavior.
