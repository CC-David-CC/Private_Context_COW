# PCC model extension and serving architecture plan

Revision: 2026-11-07. Proposed implementation, not implemented capability.
Supersedes the implementation recommendations in BACKEND-STATE-MACHINE-AUDIT.md;
that document remains the source-inspection record. QWEN36-LOW-MEMORY-INTEGRATION.md
remains the Qwen qualification record and concrete model task.


## Direction: from individual integrations to a reusable model platform

Today PCC combines a specialized Flash MoE engine with separately integrated
native dense adapters. Adding Qwen3.6 currently means another architecture branch,
new loader wiring and new continuation handling. We are steering toward one
serving and learning platform: a model supplies its math, weight mapping and state
semantics; PCC supplies reusable branching, execution/resource coordination,
publication and recovery. BW and SSS consume the same contracts across models.

The immediate proving case is Qwen3.6-35B-A3B UD-Q4_K_XL, a hybrid-attention MoE
that runs on both a 6 GB Surface and larger GPUs. Standalone generation works;
PCC setup/serving, state export/import, exact continuation, branches, eviction
and reconstruction are not yet integrated or qualified for it. A standalone
engine process is not a PCC saved-context backend. Qwen3.5 dense support does
not automatically confer support on the different Qwen3.6 MoE architecture.

The desired practical outcome: run BW against this model, save its consumed
state, branch from it, continue after restart, survive cache eviction and rebuild
when explicitly requested, with honest resource and recomputation accounting.
Existing LoRA and full-training paths must keep working; this does not claim
Qwen3.6 training exists. Its gradient/adapter implementations need their own work.

### Present model support: bounded evidence, no universal fully-supported label

| Model route | Existing documented support | Important boundary |
| --- | --- | --- |
| Qwen3.8 Flash / qwen4exp | PCC serving and COW context mechanics; bounded Q4/Q8 numerical pressure checks | Lower quants/hardware and training do not inherit that qualification |
| Qwen3.5-9B Q8_0 | Dense native adapter, text/image COW, strict restores and oracle campaigns | Whole-state copies, not physical GPU prefix sharing; hardware/profile-bound |
| MiniCPM5-2B dense Llama | Explicit checked geometry, COW route, bounded attached-LoRA inference and separate native training work | Not arbitrary Llama models or every quant/device/optimizer combination |
| Wen / Wen 1.1 | Dense adapter route and bounded client/evaluation work; native update pilots documented | Naming/import and training preparation are distinct from qualified end-to-end learning |
| Qwen3.6-35B-A3B Q4 | Standalone 128-token generation on 4090, Blackwell and Surface GTX1060 | New PCC integration target; no PCC checkpoint qualification yet |

Sources: MODELS-AND-RESULTS.md, QWEN35-9B-BACKEND.md, DENSE-LORA-COW.md,
DENSE-LLAMA-ACTION-TRAINING.md, WEN-TRAINING.md and the Qwen integration record.
This table summarizes repository documentation inspected on 2026-11-07; it is
not a fresh execution of every historical qualification. No row establishes all
seven operations on all supported hardware. Publish a generated per-profile
operation/evidence matrix as the capability contract work lands.

Success means the next supported-operator dense or MoE model reuses this path,
instead of repeating Qwen-specific orchestration changes. New math still needs
implementation and qualification; the framework removes repeated plumbing.

## Outcome and measures

Build a highly capable serving engine that is easy to extend without weakening
correctness. “Most capable in the world” is an ambition, not an acceptance claim.
Measure capability coverage, model onboarding effort, correctness, throughput,
time to first token, tail latency, concurrency, startup, peak memory, disk/network
traffic and recovery cost on published hardware/configurations. Do not optimize
one tokens/s number at the expense of usable memory, reliability or extensibility.

For a model using supported operators/state semantics, adding it should require
one model definition, weight/tokenizer/template mapping, a qualification fixture
and registration. It should require no model-name edits to scheduler, checkpoint
store, request/publication state machines, BW or SSS. A genuinely new architecture
adds operator/state implementations behind the same contracts. Record actual
files touched and model-specific branches added; use those as regression measures.
Do not promise arbitrary models work from metadata alone.

## Preserve semantics, improve implementation boundaries

Keep immutable input/history, controller state, revisioned branch references,
model/profile identities, permissions and atomic publication as the existing
source of truth. Keep strict zero-recompute restore distinct from reconstruction.
Preserve external contracts and currently qualified numerical behavior. Internal
classes, duplicated factories and inheritance hierarchies may change. Existing
coupling is not a compatibility guarantee. Version any required persisted-format
change explicitly; never weaken identity checks to make a new backend fit.

The current branching specification covers fixed-weight inference. Training uses
its own existing candidate/optimizer contracts. Supporting training is not a
license to mutate the model underlying an immutable inference snapshot.

## Separate six responsibilities

1. Model definition: tensor names/shapes, operator graph, attention/recurrent/other
   state semantics, routing, positional semantics, tokenizer/template identity,
   multimodal inputs and adapter target mapping. Dense and MoE are properties,
   not mutually exclusive implementation hierarchies.
2. Execution implementation: kernels, devices, tensor/activation/cache formats,
   numerical schedule, compiled ABI and distributed execution where implemented.
   Compose reusable primitives; keep model-specific math out of orchestration.
3. Continuation state: typed state components, consumed frontier, export/import,
   copying/sharing rules, physical layout, compatibility and work accounting.
   Include recurrent state and controller-related boundaries, not only KV tensors.
4. Placement and resource policy: weights/experts, live state, checkpoints,
   workspace, training tensors, gradients and optimizer memory. VRAM, RAM and disk
   are distinct budgets. Expert streaming, checkpoint spill and active-state
   offload are different operations; never imply one enables the others.
5. Training operations: differentiable operators, parameter selection, frozen-base
   LoRA application/training, full updates, precision and optimizer continuation.
   Reuse the model definition where valid; do not force training and inference
   through an identical kernel or serialized-state interface.
6. Serving/control plane: admission, queues, batching, scheduling, branch operations,
   cancellation, publication, authentication, recovery and observability. Operate
   on validated operation contracts, not architecture-name conditionals.

Start with a small in-repository registry and shared factory. Do not introduce a
new package manager, dynamic plugin ecosystem, generic graph compiler or parallel
history store merely to demonstrate these boundaries.

## Capability contracts are constrained operations

Independently describe inference, LoRA inference, LoRA training, full training,
optimizer resume, checkpoint restore and reconstruction. Also describe branching,
continuation, export, input modalities, legal-token selection, cancellation,
batching, physical prefix sharing and supported placement operations as relevant.
None inherits support from another. Use shared contract types, not one boolean
per feature and not an unbounded collection of arbitrary flags.

Each operation declaration includes:
- implementation/protocol version and compatible state codec;
- supported model/tensor formats, device topology and numerical profiles;
- operation-specific limits, adapter targets/rank/scale and optimizer combinations;
- prerequisites, memory/work estimates and cleanup/cancellation boundary;
- qualification status: implemented-unqualified, qualified for an exact envelope,
  unavailable, or disabled by policy, with a useful reason and next action;
- evidence references bound to source/build/model/profile and qualification scope.

Static declarations, native startup probes and executed qualification are separate
facts. A startup handshake proves protocol availability, not numerical correctness.
Unknown operations fail closed. Missing declarations must not masquerade as
unsupported implementation: report what is unknown and how to qualify it.

Admission resolves (model, runtime, operation, inputs, adapters, device, budget) to
a concrete checked plan before expensive allocation. When exact estimates require
loading, use a bounded staged probe with cleanup. Report which constraint failed
and available compatible alternatives. Never silently change precision, schedule,
model or strict-restore semantics. An explicitly selected replay path remains
available when qualified; it reports recomputation rather than claiming restore.

## Make existing state machines explicit without serializing everything

Use separate state dimensions and transition guards, not a single giant enum:
- worker health/lifecycle and lease ownership;
- admitted/executing/settled requests, cancellation and pause at consumed boundaries;
- durable publication, retry receipts and committed-result recovery;
- logical snapshot retention and independently tracked materializations in each tier;
- expert cache entry lifetime and asynchronous-transfer leases;
- training jobs, completed-step optimizer state, candidate verification and promotion.

A snapshot can simultaneously have GPU, RAM and disk materializations. Export can
produce a durable copy while a resident copy remains usable. Consequently the old
linear absent -> resident -> exporting -> durable sketch is not the state model.
Write transition tables identifying authority, locks/leases, side effects, commit
point, legal concurrency, cancellation result, rollback/cleanup and restart action.
Trace identifiers connect these machines without coupling their lifetimes.
Only poison a worker when native state validity is uncertain; input rejection
before mutation must not destroy a healthy model. Preserve safe concurrency.

Training writes a separate immutable candidate/adapter. Resume binds optimizer,
base/adapter, dataset, cursor, schedule and configuration. Publication/activation
is explicit, preserves pinned in-flight work and creates a new identity. Test
cache incompatibility after updates. Quantized inference support must never imply
quantized full training or gradient support.

## Delivery slices and merge gates

A. Characterize existing behavior with existing model profiles. Identify all
factory/architecture branches and current contracts before replacement. Capture
startup, admission and lifecycle regressions; retain known unsupported cases.

B. Introduce shared registration, construction and constrained operation admission.
Prove known profiles retain identities/behavior and unknown backend names fail
consistently through CLI and HTTP. Add table-driven compiler/toolkit selection
(GCC 14/15 where supported); test actual GCC14/CUDA12.9 Pascal and GCC15/newer CUDA.

C. Migrate one existing dense backend and one existing MoE backend without changing
their math. Add lifecycle guards incrementally, preserve leases/receipts and run
applicable G0-G5 checks. Do not label unexecuted gates passed. Avoid making every
future optimization a prerequisite for the first useful merge.

D. Integrate Qwen3.6 using those contracts and bounded expert placement. Test the
normal PCC server and clients on llm-49 and Surface16, not only standalone chat.
Use independent fresh computation as oracle; compare logits/controllers, branch
isolation, continuation and strict restores within the declared numerical envelope.
Hybrid/all-resident parity is currently unqualified and outputs differed in the
smoke checks; investigate rather than treating readable text as equivalence.
Qualify LoRA/training only after actual corresponding implementations/tests exist;
retain existing qualified training on other models throughout migration.

E. Expand conformance: process restart, cancellation/failure injection at allocation,
consume/export/publication boundaries, wrong leases, poisoned-worker recovery,
duplicate retries, optimizer mismatch, adapter mismatch, corrupted state, eviction,
reconstruction, multi-client admission and memory pressure. Add generated legal/
illegal transition sequences and targeted native numerical tests. Simulations
supplement, not replace, hardware tests. Account for all allocations and cleanup.

F. Run BW/SSS unchanged first: BW real tool loop and branch reuse; SSS evaluation,
LoRA/full-training and optimizer-resume regression flows where currently supported.
SSS imports PCC helpers as well as using service contracts. Change clients only
for a demonstrated contract requirement; document and test any migration.

For each coherent slice, review source, compatibility and evidence and merge only
the passing slice into main. Use a clean main checkout to build and repeat relevant
checks. Keep Qwen and runtime capability qualifications explicit; main may contain
opt-in unqualified operations only if they cannot be advertised/admitted as
qualified by default. No automatic promotion based on branch smoke results.

## Future capability roadmap, not current promises

Use the same boundaries for continuous batching, chunked prefill, speculative
execution, paged/shared state, distributed placement, multimodal serving and
training/serving scheduling. Prioritize measured bottlenecks; add each with its own
operation, resource, continuation and cancellation contract. Preserve existing
correctness while allowing faster execution profiles with explicit qualification.
