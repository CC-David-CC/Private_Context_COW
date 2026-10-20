# Context COW guide for people and agents

Private prototype guide. Updated October 17, 2026.

JSON update: named JSON Schema definitions now sit beside GBNF in the existing
grammar registry. Required fields, allowed tool names and argument types have
bounded real-Q4 CLI/restart evidence. The controlled timing comparison showed
no meaningful overhead relative to GBNF. See [JSON and JSON Schema](JSON-SCHEMA.md)
for configuration, measured scope and remaining qualification limits. Host
permissions and semantic checks still decide whether an action may execute.

Image sampler update: the Q4 CLI/HTTP path has a bounded real-GPU pass for DRY
with Typical and DynaTemp together, including fork, reseed, retry, disk restart
and verified re-encoding after embedding eviction. This requires the new pinned
mixed V4 native build; old builds reject the combination explicitly. Use the
existing `--dry-multiplier`, `--typical-p` and `--dynatemp-range` generation
options. Each branch retains its own controller progress. DRY with XTC,
Mirostat or Adaptive-P is not yet supported on image histories. Separate-filter and combined 1K checks also passed; target 8K qualification
remains open. This is ready for bounded client integration;
it does not establish full image qualification or original G1-G5 certification.
See the current evidence status and VISION-DRY-ENTROPY-TINY-2026-10-17.json.
### A future contribution: DRY + XTC with pictures

**Plain English:** DRY helps the model avoid repeating itself. XTC sometimes
passes over the obvious answer so the model can try a different one. Together,
they can help when you want several ideas, explanations or possible answers
about the same text and pictures. They do not make an answer true. David does
not need this picture-aware combination yet, so that connection is unfinished.
The pieces already exist. If this would help your work, a contribution and PR
with tests would be welcome when the project opens for contributions.

**Technical:** DRY and XTC are individually exercised on image histories, and
the ordinary text-only sampler pipeline already supports their combination.
The image-aware COW profile still rejects DRY + XTC explicitly, including
text-only requests made through that profile. Combining them can diversify
context-conditioned generations when exploring learned associations, alternative
explanations or candidate structured extractions; it does not inspect model
weights directly or establish semantic correctness. An isolated native V5
handler fixture passes six CPU tests, but it is not an enabled engine feature.
The remaining connector must derive bounded repetition history and barriers
from canonical mixed inputs, bind the actual native command capability, preserve
XTC's one/two-draw RNG receipts, and prove fork/resume/retry/restart behavior on
the GPU against an independent reference. Reuse the existing frozen selectors
and controller representation. Do not remove the rejection guard to expose an
untested combination. David deferred this integration to prioritize his current
workloads; a focused community PR can build on the existing pieces.

Context COW lets several lines of work keep their own saved model contexts while
sharing a GPU. The immediate target is several codebase trees with one inference
request running at a time. Read the first half to understand and use that
workflow. The second half is a handoff for an agent implementing, integrating or
testing it. Client integration proposals are written in English, without code or
pseudocode.

## Part one Human guide

### What the system saves you from doing again

Suppose you give a model a large codebase, then ask eight different questions
about it. The expensive beginning is mostly the same. Context COW aims to compute
that beginning once, preserve it, and start each question from that saved point.
Each question still costs computation for its new input and its generated answer.

The useful change is that an answer does not have to become part of every later
question. Keep a clean codebase base. Fork a branch for a performance question,
another for a correctness question, and another for an alternative design. Each
can continue independently. The same principle applies to a useful intermediate
discussion: fork there when the next investigation should inherit that discussion.

The proposed 100K-token codebase plus several-K-token questions is the intended
larger workload. It is not the size of the latest full client qualification.
The latest three-codebase tests used roughly 7K to 8K tokens per prepared base,
different questions of roughly 512 tokens, and 128-token answer segments.

### The three things called saved state

| Thing | Plain meaning | What changes |
|---|---|---|
| Immutable snapshot | An exact saved point: consumed input history and continuation controller | Existing snapshots stay fixed; further work creates a new snapshot |
| Branch reference | A named bookmark selecting a snapshot | The bookmark moves after an accepted operation |
| Computational checkpoint | Saved native model state and the next-token frontier attached to a snapshot | It can move between storage tiers or be discarded under retention rules |

The controller includes the decisions needed to continue a turn, such as stopping
and output-decoder state. The current real fast-path evidence is for greedy
generation. It does not establish that every sampler or grammar state works.

A transcript tells us what the model received. A numerical checkpoint preserves
work already performed on that input. Keeping the transcript alone does not avoid
prefill. Keeping numerical bytes without the matching history, controller and
model profile does not identify a safe continuation.

A snapshot also does not contain the whole agent's world. A Git checkout, an
edited file, an open terminal and a tool's external effects need their own
management. Retrying from a saved point does not undo a command that changed a file.

### Several trees sharing one worker

Imagine three people investigating three projects. Each project has a prepared
base and a few question branches. They may all have work waiting, but the GPU
serves one request at a time. A waiting branch keeps its identity and saved state;
it does not need its own permanently occupied GPU.

For example, the service can answer a question on the renderer project, switch to
the database project, then return to an unfinished renderer answer. Its job is to
restore the selected branch, consume only the new work, and publish the new saved
point on that branch. The database discussion must never leak into the renderer
branch.

This is more useful than forcing all three projects into one continually growing
chat. It also lets a person compare two answers from the same starting point.
Branching does not automatically improve reasoning or choose the best answer;
the person or agent still evaluates the results.

### Where GPU RAM system RAM and disk fit

The active computation lives on the GPU. System RAM holds reusable checkpoint
material shared by waiting work. Disk holds durable copies so a context can
survive a process restart and so RAM can be reclaimed.

The pool manager accounts for admitted checkpoint bytes and temporary working
reservations. A context currently being used is pinned against eviction. A RAM
entry is eligible for ordinary eviction when it is unpinned and has a durable
copy. A failed save is a separate case: its unpinned bytes are also eligible for
eviction, and the caller receives an error. Retrying publication can use those
exact bytes only while they remain available; it never silently recomputes them.
When a requested context is only on disk, the
system must read, validate and load it before computation can use it.

Durable publication still costs time when creating a new save point. Maintaining
a disk copy moves that cost earlier; it does not make the cost disappear. The
system cannot promise zero waiting if all victims are pinned, disk is full or
storage is slower than the work arriving.

The disk collector can remove inactive numerical copies, including a branch's
beginning or current tip, when it needs space. It keeps the saved conversation
history and branch bookmarks. Copies currently being read or saved stay pinned.
Fork and Retry here can still move bookmarks without loading the model state.

If another valid cached copy remains, the next request can reuse it. If every
numerical copy is gone, the request rebuilds the saved history and reports that
extra computation. This does not recover the old computational state: under
the prototype's arrangement-dependent execution, a rebuilt answer can differ.
Strict restore is different: it always fails when its required bytes are missing.
The updated eviction/rebuild path has bounded GPU evidence; it is not a complete
crash, disk-full or long-running qualification.

The shared-pool budget is not the machine's total memory use. Model weights,
native allocations, Python, filesystem cache and temporary copies also use
memory. A checkpoint's file size is not its peak capture-memory requirement.

The current HTTP service uses GPU 0. The broader design distinguishes several
independent GPU workers from one request split across multiple GPUs. Selecting
three independent workers should allow up to three simultaneous requests. A
split worker using three cards is still one request occupying those three cards.
Real multi-GPU model serving is not qualified by the current single-GPU results.

### A practical terminal workflow

The terminal client connects to a COW service running the model on Linux. Your
terminal can be on Windows. An SSH tunnel can carry requests to the server's
loopback address. Installing the client is separate from installing the model,
native engine and machine-specific execution profile.

Use this sequence when the client package has been qualified:

1. Create a named workspace or namespace for a project. A namespace groups saved
   presentation and branches; it is not a separate model or an inference worker.
2. Load the stable codebase context into a new base branch. Retain that base
   rather than asking every question directly on it.
3. Fork two question branches from the base. Ask a different question on each.
   The unchanged codebase computation can be reused, while each question gets its
   own suffix and answer.
4. Continue a branch when you want more of its current answer. A token limit may
   end the request while the answer's controller remains active.
5. Once the model has stopped its turn, append another user question to that
   discussion. For a completely independent question, fork the clean base again.
6. Return to a saved point to try another approach. **Fork** adds another named
   branch. **Retry here** moves the current branch bookmark back; its old ending
   remains in saved history. It does not generate until you send or continue.
   The existing CLI command and API operation are still named `rewind`.
7. Keep the durable receipt for an operation whose outcome matters. After a
   connection failure, inspect authoritative state before submitting more work.

The command walkthrough belongs in the companion CLI instructions. The CLI has
passed bounded Q4 and Q8 real-service fork, question, continuation, rewind and
restart tests; it is an early private draft. This guide does not make
an OpenCode, Hermes or Codex integration available by changing a URL.

### Streaming and durable completion

Text can appear before its state has been durably published. That text is
provisional. A successful durable receipt means the selected tokens have been
consumed and the associated saved point has been published.

A durably saved partial answer can still have an active controller. These are
different questions: has this request committed, and has the model finished its
answer? Continue an active saved answer; do not pretend a token cap was an
end-of-turn token.

If the connection drops, publication may have won the race or cancellation may
have happened first. A missing response alone does not tell you which. Use the
operation identity and branch state. Starting the same logical action under a
new identity can accidentally create another action.

### When a change requires new computation

Reuse applies to an unchanged consumed prefix. Changing an earlier token means
that the affected later computation normally has to be redone. A small change
near the end is cheaper than replacing the beginning of a large context.

There is no feature here that repairs arbitrary changed earlier input by rotary
position math. The prototype is not using a promise of equivalent meaning to
treat different token histories as the same computation. It also does not promise
identical logits across arbitrary prefill partitions: exactness claims belong
to a pinned execution profile and the tested arrangement.

For code changes, keep the old base as the old revision and prepare a new base
for the new revision. The client may append an explicit patch as new information
when that is the intended prompt, but that is a different history from replacing
the old file. Keep that distinction visible.

### What has worked and how fast

These are bounded real-model results on llm-60's RTX PRO 6000 Blackwell, FP16 KV,
with pinned Q4 and Q8 configurations. The pressure runs deliberately used a 9GiB
RAM pool to force eviction; normal service defaults are 96GiB RAM and 128GiB disk.

| Executed check | Result and scope |
|---|---|
| Three code contexts and different A and B questions | Q4 and Q8 each matched 768 generated token IDs and 12 saved full-logit vectors against independent references; maximum measured error zero |
| Forced RAM eviction and disk return | Both configurations observed six positive-byte evictions and one disk reload; the return generated 128 new tokens with zero prefix prefill |
| New process restart and live disconnect | Q4 and Q8 each passed clean restart, cancellation before publication, retry checks and further continuation; the lifecycle harness did not compare logits |
| Three clients sharing one worker | Q4 passed three concurrent submissions from different trees, 128 outputs each, observed queueing and a maximum of one active request on GPU 0; source state stayed unchanged |

For six distinct short questions per model, median generation-only rates were
74.63 output tokens per second for Q4 and 62.29 for Q8. Including question prefill,
capture and durable publication, median output tokens divided by whole request
time were 33.20 and 28.35. These measures answer different questions. The latter
is useful for a short-answer request; it is not steady decoding speed.

The measured disk-return requests took 4.606 seconds for Q4 and 6.018 for Q8 to
produce and durably publish 128 new tokens. Startup and initial base preparation
are excluded. These are individual observations, not a repeated benchmark or a
promise for a 100K context. The saved latency figures distinguish these costs.

### What an early tester should expect

The immediate draft is a private terminal client against the tested service.
A friend installing the engine on a different machine still needs a compatible
build, model files and a newly qualified execution profile. Machine-specific
profiles cannot simply be copied across GPUs or model preparations.

The broader G1 through G5 certification remains incomplete. Extended fault and
pressure campaigns, general samplers, long serving workloads and ordinary
coding-client adapters remain work. A separate ordinary-serving cold-versus-cache
disagreement is unresolved; the bounded COW results do not settle that issue.

An early trial is most useful when the tester brings a few real projects, keeps
their base versions stable, asks independent questions on branches and reports
latency together with operation and snapshot identities. Merely repeating the
same tiny prompt provides less information about the intended workload.

## Part two Agent guide

### Preserve the objective and the authority boundaries

The objective is native-speed Context COW for one active request, serving
several useful context trees with RAM and disk backing. Q4 is the iteration
configuration; Q8 is the target. Keep integrity checks and durable publication.
Do not replace the objective with a fast transcript cache or a passing test-only
runner.

Canonical history is parent-linked immutable consumed input plus controller
state. Branch references are movable selectors. Checkpoints and execution
receipts describe derived computation; they must not become a second canonical
history. Presentation text is a view of accepted token history, not a source for
re-encoding generated tokens during resume.

The user accepts a bounded prototype whose exact continuation is tied to its
execution profile. Preserve the original mathematical contract and registered
gate tests. Record cross-partition mismatches rather than editing equations,
assertions or tolerances to accept them.

### Find the project and its evidence

The complete current Windows checkout is
`C:\Users\dflanag3\Documents\fleet\context-cow-model-free-fleet`, branch
`feat/model-free-fleet`. Its name predates the serving work. The private source
remote is `context-cow-private`, pointing to
`box-git:/srv/data/git/private_context_cow.git`. Large raw evidence lives separately
under `/srv/data/git/private_context_cow.evidence` on box-git. A source clone does
not download those archives.

Read repository instructions, the branching specification and
`experimental/branching_context/evidence/CURRENT-HANDOFF.md` before changing
anything. Follow its evidence links to actual terminal results. Historical
“running” notes and old PIDs are not current process state.

| Location in the checkout | Responsibility |
|---|---|
| `experimental/branching_context/src/context_time_travel/` | Canonical store, controllers, coordinator, checkpoint pools and backend adapters |
| `src/` and `include/` | Modified native Strata engine and COW implementation |
| `serve/cow_http.py` and adjacent COW modules | Dedicated COW service, text admission and canonical workspace projection |
| `experimental/branching_context/scripts/` | Audit and campaign runners, not the end-user CLI |
| `tools/cow_cli.py` and `docs/context-cow/CLI.md` | Private terminal client and walkthrough, bounded Q4 and Q8 real-service qualification |
| `experimental/branching_context/evidence/` | Compact results, decisions, source fingerprints and restart handoff |

Use llm-60 GPU 0 for this campaign, with one inference job at a time. Keep the
ordinary chat server off and work private. Use `python` on Windows; verify the
interpreter on Linux independently. Tokenizer and service work on llm-60 uses
the established Strata virtual environment. Do not schedule from a lock file
alone: inspect the actual process and the external runner's exit receipts.

### What each client must learn

The client needs a durable association between its project or task and the COW
namespace, branch, saved snapshot, revision and execution profile. These handles
select server state; they do not duplicate that state. Keep the association
separate from a friendly branch name, which can change.

An integration must distinguish four actions: append new information to a
discussion, continue an unfinished answer, fork a saved point, and move a branch
back. A generic send-full-transcript call cannot express all four reliably.
Reconstructing the same-looking transcript is especially unsafe for pending
UTF-8 output, tool framing and exact token boundaries.

Prepare stable project material once, then attach each independent investigation
to an explicitly selected base. Keep changing questions, tool results, timestamps
and branch-specific instructions after that base. Changing stable instructions,
tool definitions or source files must produce an appropriate new identity and
computation boundary. Similar text is not sufficient proof of shared state.

Use a client-side task queue for readiness and user feedback, with the server
remaining authoritative for GPU admission. Many agents can inspect files, run
independent tools or wait concurrently. Inference on one selected GPU remains
serialized. Background title generation, summarization and review must also
enter the same admission policy if they use that GPU.

### The OpenCode integration in theory

OpenCode documents configurable provider endpoints. Its server documentation
also exposes session operations. These are useful integration surfaces; endpoint
configuration alone does not make its sessions COW-aware.
[OpenCode providers](https://opencode.ai/docs/providers/) and
[OpenCode server](https://opencode.ai/docs/server/).

Associate each OpenCode project/session with an explicit COW branch. When the
user branches an investigation, connect that action to a server fork from the
acknowledged snapshot, then keep the new session attached to the returned branch.
When the user revisits a prior message, select the corresponding consumed saved
boundary rather than rebuilding a prompt from visible chat text.

Place a COW-aware provider adapter between the agent loop and inference. It must
carry branch and revision identities, translate streaming events, retain operation
IDs and distinguish a durable partial answer from a completed turn. It can live
in a client change or a suitable extension; the supported extension surface must
be verified against the installed version before choosing either route.

OpenCode's file editing and execution history must continue to manage its own
workspace effects. A model rewind should not silently imply a filesystem revert.
If the client compacts or rebuilds context, make that an explicit new history
instead of moving an old checkpoint under a rewritten prompt.

### The Hermes integration in theory

Hermes documents custom inference endpoints and separately routed auxiliary
work. That gives a place to route calls, while branch ownership still needs an
explicit adapter.
[Hermes configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration/).

Associate each ongoing task or session with a COW branch, and give an independent
research or review task its own fork. Sharing a base does not mean sharing a
mutable agent scratchpad. When the parent receives a worker's result, append
that result as new information to the parent's selected branch; do not splice
the worker's private numerical state into the parent.

Route all model-using auxiliary work through the same one-request-per-GPU
coordinator, including automatic summaries or review calls. Keep external tools
concurrent when safe. A context compressor should publish a new summarized
history and retain the original branch according to policy; it must not pretend
that a summary is the same saved computation.

Provider fallback is a compatibility boundary. A different model or execution
profile needs its own computation, even if the agent presents it as the same
task. Preserve the task's relationship to both histories without claiming
checkpoint interchangeability.

### The Codex integration in theory

Official OpenAI documentation describes custom model providers and Responses
API provider configuration. Codex provider compatibility and COW branch semantics
are separate integration problems.
[Official advanced configuration](https://developers.openai.com/codex/config-advanced/).

Choose whether Codex is driving COW as a tool or using a COW-hosted model for its
own inference. A tool integration can ask COW to prepare, fork, inspect or resume
a separate local-model investigation while Codex keeps using its existing model.
That does not preserve Codex's own remote model computation.

Using COW for Codex's inference needs a provider bridge implementing the wire
protocol expected by the installed Codex version, including streaming and the
items used by its agent loop. The current dedicated COW HTTP service is not that
bridge. In particular, exposing ordinary Chat Completions alone does not establish
Responses compatibility. Preserve item identities and tool-call relationships
where the client protocol requires them.
[Official Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create).

To obtain full tree control, map a Codex thread and its forks or resumptions to
specific COW branch/snapshot handles. A generic proxy seeing only prompt text may
accelerate exact prefixes, but cannot safely infer every intended rewind or fork.
A client extension, explicit control channel or maintained client change would
need to carry those decisions. Treat that as proposed work, not a working Codex
configuration delivered by this prototype.

### Keep retries tools and adoption honest

Record an operation identity before sending a mutation. On lost connectivity,
inspect its result and canonical branch state. The current operation-status
facility is process-scoped; it is not a durable universal ledger after restart.
Preserve acknowledged receipts and use canonical saved state for recovery. Do
not blindly resubmit with a new identity or a freshly fetched revision and call
that the same request.

Tool integration needs its own qualification. It must preserve literal user
text, tool names, call IDs, arguments, outputs and their ordering. The current
bounded terminal draft is plain text; it is not a complete agent tool-call
implementation. Neither hiding tool tags nor appending rendered text is a
substitute for an exact protocol boundary.

After evaluating alternative branches, adopt a selected saved tip through a
revision-checked reference move where supported. Preserve the rejected branches
according to retention policy. Numerical state is not combined by merging
answers; if the desired continuation includes both answers, that is a new input
history requiring its own computation.

### Protect reuse and storage through the full request

Keep source checkpoint identity, compatibility validation and pool ownership
bound together. A serialized backend-cache alias must not bypass the RAM pool's
budget or keep evicted bytes secretly resident. The corrected implementation
borrows cold state under a pool pin and temporary reservation; native image
handles retain their distinct resident path.

Make disk state durable before marking its RAM entry safe for eviction and before
acknowledging the selected branch. Hold the necessary retention/pin ownership
through publication. Report capacity failure rather than deleting canonical
history or evicting a live source. Removing a numerical checkpoint may require
explicit recomputation later; strict restore must reject missing state rather
than silently replaying the prefix. Ordinary queued execution reports source
reconstruction separately. Canonical fork/retry actions do not require numerical
bytes; a present but corrupt checkpoint must still be rejected.

Disk admission uses process-shared byte reservations and digest pins. Protect
the payload digest before checking whether its file already exists, then grow
the same reservation only when payload bytes must be written. Keep those leases
through publication. An inactive branch bookmark retains canonical history, not
a permanent numerical-cache allocation.

A durable operation receipt can outlive its numerical cache. New publications
bind the original checkpoint identity into the atomic branch move. Retrying
that same operation may recover its old success receipt without loading model
state or moving a newer branch. It does not recreate evicted computational
bytes. Keep receipt recovery, exact restore and explicit reconstruction distinct.

For future multi-GPU work, keep physical GPU membership distinct from worker
identity. A split worker owns every member while serving its request. Independent
workers can share host pools only when the checkpoint layout and execution
profile are compatible. Model-free memory tests do not qualify real-model
placement, numerical behavior or throughput on those devices.

### Validate the client integration as an actual workload

Start with three stable real codebase roots and two independent questions per
root. Exercise selecting a tree, forking, answering, continuing, switching away,
returning, rewinding and restoring after restart through the client itself.
Include waiting requests and a live disconnect. Check server history and native
work, not only a plausible answer in the terminal.

Require the expected source snapshot and revision for each accepted request,
unchanged parent and sibling bytes, full consumed output/controller publication,
and zero saved-prefix computation on strict restore. Compare numerical states
with independent references under the stated profile. The latest concurrency
test checks lineage, controllers and counters but does not supply new oracle
comparisons; do not promote it into numerical qualification.

Measure initial preparation, context-switch/load time, time to first token,
inter-token timing, generation rate, capture and durable completion separately.
Include queue waiting when reporting client latency. Record actual output counts,
startup, cache warmth, model/build/profile hashes and resource placement. Report
a validation overhead above the user's 8% preference; do not discard correctness
or conceal the cost because that preference was missed.

### The next implementation order

The CLI's focused protocol, error and interruption controls now pass, together
with bounded Q4 and Q8 real-service tests. Package a private client trial with
clear server prerequisites. Do not make the friend drive an audit runner or
wait for browser polish.

Next choose one coding client and implement the narrow branch-aware adapter,
including its tool and publication boundaries. Run the several-codebase workload
through that full path, then increase context and answer lengths. Preserve a
measured baseline before optimizing.

Continue bounded pressure and failure tests on the same path: eviction while
other clients wait, failed saves, disk full, stale revisions, restart, interrupted
publication and later garbage collection. The helper startup-cleanup gap is now
fixed with CPU regressions and successful real-service shutdowns; retain its
failure evidence. Complete broader sanitizer,
soak, sampler and gate qualification separately. An early private draft and a
certified release have different completion conditions.

### Evidence to consult before making claims

The links below are local repository evidence, not promises of broader support.

- [Current handoff](../../experimental/branching_context/evidence/CURRENT-HANDOFF.md)
  and [status](../../experimental/branching_context/evidence/STATUS.md).
- [Q4 A and B numerical result](../../experimental/branching_context/evidence/Q4-CLIENT-PRESSURE-NUMERICAL-2026-10-15.json)
  and [Q8 A and B numerical result](../../experimental/branching_context/evidence/Q8-CLIENT-PRESSURE-NUMERICAL-2026-10-15.json).
- [Q4 restored continuation](../../experimental/branching_context/evidence/Q4-REVISIT-NUMERICAL-2026-10-15.json),
  whose single256 versus128+128 generation-call distinction remains explicit.
- [Q4 HTTP recovery](../../experimental/branching_context/evidence/Q4-CLIENT-RECOVERY-2026-10-15.json)
  and [Q8 HTTP recovery](../../experimental/branching_context/evidence/Q8-CLIENT-RECOVERY-2026-10-15.json).
- [Q4 three-client result](../../experimental/branching_context/evidence/Q4-MULTI-CLIENT-2026-10-15.json)
  with complete raw-result identity and server/native closure records.
- [Request latency chart and measurements](../../experimental/branching_context/evidence/figures/2026-10-15-browser-durable-latency/README.md).
- [Q4 real CLI campaign](../../experimental/branching_context/evidence/Q4-CLI-REAL-2026-10-15.json)
  and [Q8 real CLI campaign](../../experimental/branching_context/evidence/Q8-CLI-REAL-2026-10-15.json),
  including exact restart tokens/snapshot, canonical output decoding and native work.
- [Receiver-verified private backup](../../experimental/branching_context/evidence/Q8-RECOVERY-ORACLES-PRIVATE-BACKUP-2026-10-15.json)
  for completed audits and Q8 recovery. Newer client work needs its own checkpoint.
