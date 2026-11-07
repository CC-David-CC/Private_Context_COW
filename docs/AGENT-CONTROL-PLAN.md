# PCC model-platform agent control plan

Version 1, 2026-11-07. This is an execution contract for future implementation.
It does not assert that the proposed registry, state APIs or orchestration tools
already exist. Examples marked as templates require real discovered values.

## 1. Mission and completion boundaries

Turn PCC's individually wired integrations into a reusable serving and training
platform. A new model contributes its math, weight/input mappings and continuation
semantics. PCC provides branching, scheduling, persistence, recovery and resource
management. Prove this by adding Qwen3.6-35B-A3B UD-Q4_K_XL on three hardware profiles
while preserving currently qualified PCC behavior.

Optimize for correctness coverage and ease of extension. Tokens/s is not a goal
or a tradeoff against correctness. Bounded memory, termination and recovery are
correctness/practicality requirements; timing and I/O are diagnostic observations.

There are three distinct finish lines:
- **Slice accepted:** the reviewed change passes its declared tests, affected
  existing regressions and documented interface migration requirements.
- **Qwen operational milestone:** normal PCC loading/serving, G1 continuation,
  G2 persistence, strict restore and explicit reconstruction work for each claimed
  device/profile. Publish which physical-sharing features remain unqualified.
- **Fully qualified release:** all advertised combinations meet the applicable
  cumulative G0–G5 obligations, training regressions and interface migration tests.
  A G1/G2 demonstration is not a G3 sharing or production-robustness claim.

The specification, instructions and acceptance runner determine gates. An early
passing slice may merge without advertising unfinished Qwen operations. The
controller cannot redefine an unavailable mandatory gate as an optional pass.

BW and SSS backwards compatibility is not required. Their future rewrites may
follow a better PCC interface. Document every changed interface and its consumer
impact; use a small PCC-owned conformance client for current acceptance. Existing
history, model identity and training correctness remain mandatory.

## 2. Read before any work

Controller and workers read:
1. Root `AGENTS.md`, then every applicable nested `AGENTS.md`.
2. [MODEL-EXTENSION-PLAN.md](MODEL-EXTENSION-PLAN.md),
   [MODEL-SUPPORT-BOUNDARIES.md](MODEL-SUPPORT-BOUNDARIES.md), this document and
   [PCC-INTERFACE-CHANGELOG.md](PCC-INTERFACE-CHANGELOG.md).
3. `experimental/branching_context/AGENTS.md` and the **entire**
   `experimental/branching_context/spec/branching-context-api.md` before runtime
   state changes; relevant `plans/tickets.md`, `docs/test-matrix.md`,
   `docs/acceptance.json`, `docs/locked-references.json` and `scripts/verify.py`.
4. The concrete backend/source and model qualification documents for the ticket.
   Training tickets also read the existing native training and optimizer contracts.
5. The controller's current task ledger and private machine mapping. Read the real
   files; an older chat summary does not override changed source.

Useful initial inspection commands (run in the assigned checkout):

```sh
git status --short
git rev-parse HEAD
git branch --show-current
git remote -v
git config user.name
git config user.email
```

Use `rg` for searches when installed; box-git currently lacks it, so use `grep`
there. Missing convenience tools are not a reason to install system packages.
Inspect actual test entry points before running them; do not invent command names.

The gate runner supports G0–G5 although spec section 17's table lists G0–G4.
Retain G5 from the existing instructions, tickets, acceptance matrix and runner.
Do not edit the locked mathematical spec/registry to manufacture a passing gate.

## 3. Immutable problem contract

A correct implementation must preserve all these invariants:

| Invariant | Required observation / forbidden shortcut |
| --- | --- |
| Authoritative history | Parent-linked exact inputs and complete controller survive every numerical-cache eviction |
| Consumed boundary | Every recorded token is consumed; next logits predict the next token; no hidden pending sampled token |
| Branch isolation | A observes P+A and B observes P+B, independent of scheduling; never P+A+B |
| Snapshot immutability | Tail pages, recurrent buffers, position maps and quant metadata reachable from a snapshot cannot be overwritten |
| Complete controller | RNG algorithm/draw position, grammar, penalties, stop progress, sampler order and output decoder resume exactly as promised |
| Identity | Weights, adapters, input/template semantics, positions, execution profile and codec are validated before state reuse |
| Strict restore | Complete compatible numerical state is restored without hidden forward-token work; reconstruction is a separate explicit operation |
| Atomic publication | Branch revision CAS and retry receipt commit together; stale ABA updates fail; ambiguous outcomes can be recovered |
| Retention and access | Numerical eviction is distinct from history deletion; hashes are not authorization; inaccessible roots remain inaccessible |
| Training isolation | Training creates a new immutable candidate; optimizer resume binds the correct training stream; serving work pins its model identity |
| Honest capability | Inference, LoRA inference/training, full training, optimizer resume, restore and reconstruction never imply one another |
| Honest evidence | Test execution, profile, failures, missing tests and unsupported operations are reported separately |

Exact structural identity and numerical equivalence are different checks. No
floating-point tolerance excuses changed history, RNG cursor, ownership, grammar
or permissions. Never lower thresholds, remove negative controls or replace a
real engine with a mock to obtain PASS.

## 4. Architecture contract the controller must freeze

Separate six responsibilities; they may share types, but not hidden ownership:

| Boundary | Supplies | Must not own |
| --- | --- | --- |
| Model definition | Operators, tensor/input mappings, geometry, positions, modalities, recurrent/attention state and adapter targets | Request queue, branch transactions or disk policy |
| Execution | Device kernels, numerical schedule, supported tensor formats and native ABI | Authoritative history or unversioned model selection |
| Continuation | Complete typed numerical state, consumed frontier, export/import/sharing semantics, codec compatibility | Global mutable sampler or an alternate history store |
| Placement | Bounded weights/experts, live state, transfer buffers and checkpoint materializations | Silent quantization, history deletion or implied KV offload from expert streaming |
| Training | Differentiable operators, parameter subsets, optimizer state, immutable candidate production | In-place alteration of a serving model or automatic candidate promotion |
| Serving | Admission, queues, cancellation, publication, auth and recovery | Model-name branches that duplicate tensor or checkpoint semantics |

Start with a small checked registry and shared factory. A general plugin market,
compiler framework or new history database is not a prerequisite.

Each operation contract must record implementation/protocol version, model and
format constraints, device/numerical profiles, limits, prerequisites, resource
estimates, ownership/cancellation rules, codec and qualification envelope. Separate
`implementation_status`, `qualification_status` and `policy_enabled`. Admission
resolves an exact operation plan; a startup handshake alone never qualifies math.

Error contracts include a stable code, failed constraint, affected operation,
retryability, committed/not-committed/unknown-commit outcome when relevant and a
useful next action. Example categories: unknown model/backend, unsupported format,
unqualified operation, incompatible profile/codec/adapter, resource budget,
conflicting revision, corrupt state, cancelled request and uncertain worker state.
Names and wire schema are finalized in P01; these are design categories, not an
already implemented API. Unknown operations reject before native mutation.

Any required API change records old/new shapes and migration in the interface
changelog. Any persisted schema change additionally needs explicit versioning,
loader behavior, identity treatment and rollback tests. Do not silently relabel
existing persisted IDs or assume an HTTP rename permits a storage migration.

## 5. Four agent roles and three machine lanes

Use one controller plus at most three worker agents. Initial roles are:

| Role | Machine lane | Work owned by default | Boundaries |
| --- | --- | --- | --- |
| Controller | Coordination host / source repository | Contract revision, common registry/factory/admission edits, task ledger, integration branch, reviews and acceptance | No unreviewed main promotion; worker summaries are not evidence |
| Implementation worker | RTX4090, 24 GB | Qwen mapping, loader/adapter, complete continuation-state implementation; existing adapter characterization | Cannot change shared contract, numerical thresholds or oracle expectations alone |
| Conformance worker | Blackwell RTX PRO6000, 96 GB | Independent fresh-history oracle, fixtures, fault injection, training regressions and cross-profile review | Cannot derive expected outputs through SUT saved-state/restore path |
| Constrained-device worker | Surface GTX1060, 6 GB / 16 GB RAM | Pascal toolchain, bounded expert/cache placement, pressure/cancellation/cleanup and Surface qualification | Cannot silently reduce profile scope or bypass compiler checks |

These are initial assignments, not three independent projects. The controller may
reassign a worker after an explicit ownership handoff. One worker remains the
sole machine job owner even when another author needs a test on that device.
Source ownership and machine ownership are separate: send a commit and test
request to the machine owner instead of launching a competing process yourself.

Public docs use hardware roles. Host IPs, login names, key paths and private job
records live in an operator handoff **outside source Git**. Use existing keys in
place; never distribute private keys to workers, machines or repositories.

## 6. Git and filesystem isolation

- Pin `baseline_main_sha` and `integration_base_sha` by inspection at dispatch.
  Do not assume main remains at the previously inspected commit.
- Keep the existing preservation branch and archived upstream patch intact.
- Controller owns the integration branch. Each ticket has a separate branch such
  as `work/pcc-platform/P03-qwen-adapter` and a separate writable worktree.
- Worker branches start at the controller-specified accepted commit, not arbitrary
  latest main or another worker's unfinished tree. Record prerequisite commits.
- Each task lists exact writable paths. Common contracts/factories are controller
  owned until explicitly delegated. Two agents may not edit the same file at once.
- Workers commit source/tests/rebuild instructions using
  `David <David@code.contractors>`. They never merge main, force-push shared refs,
  rewrite others' history or stash/delete another agent's files.
- Controller reviews and cherry-picks accepted commits onto integration, resolves
  conflicts deliberately and reruns relevant checks on the resulting commit.
  Pre-cherry-pick results do not prove a conflict resolution works.
- Back up reviewed source commits to the established box-git bare repository.
  Publish reviewed changes through the authorized PCC repository/branch. Do not
  publish access handoffs, raw private evidence or credentials.
- Use unique build and run directories keyed by task and commit. Never build two
  toolchains in one CMake cache. Runtime checkpoints from separate tests use
  separate stores unless sharing is the explicit test.
- Existing model files are immutable shared inputs identified by hash. Reuse them
  over LAN. Do not fetch a replacement model, quant or template without recording
  and reviewing that change; do not redownload already available artifacts.

Source artifacts belong in Git. Model weights, executables, build trees, logs,
checkpoint payloads and machine job state remain outside source Git. Commit small
reproducible test fixtures and reviewed nonprivate qualification summaries where
repository policy permits. Use explicit `git add` paths, never a blind `git add .`.

### Bootstrap procedure for each ticket

The controller supplies these variables; placeholders are not runnable values.
The worker must use a full accepted commit, not a moving branch name. Example
source-side procedure, after inspecting the assigned repository:

```sh
set -eu
: "${PCC_REPO:?controller must supply repository path}"
: "${PCC_BASE:?controller must supply full accepted commit SHA}"
: "${PCC_BRANCH:?controller must supply unique work/pcc-platform branch}"
: "${PCC_WORKTREE:?controller must supply fresh absolute worktree path}"
git -C "$PCC_REPO" cat-file -e "$PCC_BASE^{commit}"
test "$(git -C "$PCC_REPO" rev-parse "$PCC_BASE^{commit}")" = "$PCC_BASE"
git check-ref-format --branch "$PCC_BRANCH"
test ! -e "$PCC_WORKTREE"
git -C "$PCC_REPO" worktree add -b "$PCC_BRANCH" "$PCC_WORKTREE" "$PCC_BASE"
git -C "$PCC_WORKTREE" status --short
git -C "$PCC_WORKTREE" rev-parse HEAD
```

If the branch/path already exists, stop and reconcile ownership; do not delete or
reset it. Record the worktree's HEAD and status before editing. When building on
another host, transfer the exact committed source and pinned dependency closure
over LAN, verify its revision/tree hashes, and record the remote source, build
and run paths. A remote copy of uncommitted files requires a dirty-tree manifest
and cannot be used as final accepted-source evidence.

During P00, discover the actual operational GPU lease entry point from source
and installed scripts. Confirm two competing test acquisitions cannot both hold
it, cancellation releases it and a running child cannot outlive ownership unseen.
If no usable operational lease exists, the controller must explicitly define and
test one before dispatching heavy jobs; workers cannot invent separate lock files.
A process-local context lease is not a substitute for an operational machine lock.

For every machine job, populate these commands in the private ledger **before**
launch: verified SSH connection, source verification, environment activation,
lease acquisition, configure/build, test launch, process/log status, graceful
cancellation, bounded escalation and post-cleanup verification. Derive the build
and test commands from that revision's actual entry points. Record exact process
identity/PID start time or service unit after launch. Use a harmless short child
to test launch/status/cancel wiring before the first expensive job. This is a
bootstrap requirement, not a claim the plan ships a new job-runner executable.

## 7. Controller ledger and task packet

Keep the operational ledger outside source Git, backed up with the private
handoff. It records contract revision, accepted integration SHA, task graph,
worker assignments, owned files, machine jobs, evidence locations, unresolved
failures and the next runnable ticket. This is a control record, not an automatic
scheduler already provided by the repo.

Before dispatch, fill every required field in this template with real values:

```yaml
task_id: P03-example
parent_task: P01
contract_revision: <reviewed contract commit>
objective: <one observable outcome>
invariants: [<named invariants>]
base_commit: <full accepted SHA>
branch: <unique task branch>
worktree: <unique absolute path>
owned_paths: [<exact files or bounded subtree>]
read_only_paths: [<shared contracts, oracle, other agents' paths>]
prerequisites: [<task IDs and accepted commits>]
machine_role: <4090 | 96gb | surface | controller>
resource_lease: <controller lease ID or none for source-only work>
allowed_actions: [<build/test/process restart within owned test environment>]
profile: <model hash, execution/codec IDs, compiler and device>
resource_bounds: <host RAM, VRAM, disk, process, duration and cancellation limits>
reference: <independent expected result and exact fixture hashes>
negative_control: <named deliberately broken behavior that must fail>
numerical_criteria: <pre-agreed tolerances or exact structural requirement>
test_commands: [<commands verified against this source revision>]
deliverables: [<source/test changes, interface record, result record>]
stop_conditions: [<ticket-specific failure conditions>]
evidence_dir: <fresh absolute path outside source Git>
acceptance: <controller checks required before ACCEPTED>
status: READY
```

Missing required values mean `BLOCKED`, not permission to improvise them. Unknown
measurements are explicitly unknown. Workers acknowledge the packet, inspect their
worktree and report conflicting assumptions before modifying source.

Task states:
`PLANNED -> READY -> RUNNING -> REVIEW_READY -> ACCEPTED`.
`BLOCKED`, `FAILED` and `CANCELLED` preserve partial evidence and explicit reasons.
Only the controller marks ACCEPTED. REVIEW_READY can contain known failures for
review; it is not permission to merge. Rejected work returns to READY with a
specific correction. Controller approves scope changes, not silent worker drift.

## 8. Machine ownership, resource control and safe cancellation

Three locks must not be confused:
1. A Git worktree gives one agent source-write ownership.
2. An operational machine lease gives one job owner permission to use hardware.
3. PCC runtime context/expert/transfer leases protect in-process model state.
Passing (2) does not prove (3).

Before a heavy job, machine owner inspects actual hostname/GPU identity, available
RAM/disk, existing process ownership, toolkit/compiler and Surface AC/battery.
Use previously verified SSH host identity. A mismatch blocks writes until resolved.
Do not terminate unknown processes or assume an idle desktop means an idle GPU.

Use a remote OS lock held for the lifetime of the owned job, plus a controller
record containing task, host role, commit, PID/process group or service unit,
start time, progress file and cancellation route. A timestamp file alone is not
an exclusivity lock. Inspect/reuse the repository's existing GPU lease where
applicable; do not introduce a competing lease namespace that bypasses it.
Child processes must remain within the tracked job lifetime. Loss of the SSH
connection does not prove completion or authorize another job.

Until explicitly testing concurrency, permit one heavy build/inference/training
job per machine. CPU-only source inspection may run concurrently. Controlled
multi-client/runtime contention tests run under a single machine owner and one
explicit resource plan. Surface compilation and inference are sequential.

Initial operational bounds, adjustable by the controller with a recorded reason:
- Use short fixtures first: pinned Qwen smoke envelope is context 2048, prefill
  256, MTP off, expert RAM budget 6 GiB, 128 output tokens. This is an initial
  fixture, not a universal production context limit.
- Surface GCC14/CUDA12.9/SM61 uses the archived engine's Pascal opt-in. GCC15 with
  CUDA12.9 is rejected; never use an unsupported-compiler bypass to claim support.
  4090 and Blackwell use their inspected GCC15/newer-CUDA combinations.
- Reserve at least 2 GiB available host memory for the Surface desktop as a target;
  measure full model mappings, resident set, swap and staging independently.
  Prior smoke used substantial RAM/swap; do not call it a qualified bounded profile.
  Stop the owned test if available memory stays below 512 MiB for 15 seconds,
  OOM kills occur, or the desktop/SSH becomes persistently unresponsive.
- Do not launch Surface heavy work on battery below 30%; warn the user. Stop owned
  heavy work below 20% or on thermal/driver fault. Do not change display, GPU idle
  or firmware policies just to get a test through. No suspend/resume tests.
- Preflight free disk against model/build/checkpoint estimates plus a recorded
  reserve; until refined, keep at least 10 GiB free and stop on ENOSPC.
- Set a measured per-job deadline and progress expectation. A first diagnostic
  inference may allow up to 10 minutes given the existing multi-minute startup;
  a build gets a separate explicit bound. If extending, record observed progress
  and revised deadline. Never leave an unbounded hung process without ownership.

These are operational safeguards, not measurements that the implementation meets
them. A stricter ticket limit takes precedence. Resource refusal is useful evidence,
not a reason to disguise a smaller context/model as the requested test.

Cancellation: use the task's recorded cancellation interface, then TERM only the
owned process group after its grace period; KILL only that verified group if
needed. Never `pkill python`, `killall`, global GPU reset or reboot as cleanup.
Record whether mutation or durable commit already occurred. Quarantine uncertain
worker state; preserve acknowledged snapshots/receipts. Verify resources released
and remove test services only after confirming ownership. Planned crash tests kill
only isolated test processes/stores, never the user's desktop or shared services.

## 9. Dependency graph and concrete tickets

Each ticket is split further if it cannot produce one reviewable outcome. These
are task definitions, not claims that their tests or APIs are already implemented.

| Ticket | Owner / parallel opportunities | Prerequisites | Deliverable and acceptance |
| --- | --- | --- | --- |
| P00 Baseline and evidence map | Controller; all workers inspect their lanes in parallel | Read instructions and verified access | Pin source/dependencies/models, inventory factories/state/training, map documented results to actual tests/receipts; run applicable baseline checks and record missing assets separately |
| P01 Contract revision | Controller, reviewed by all workers | P00 | Minimal descriptors, state ownership/transition tables, admission/error and interface-change proposal; fixture examples for current dense/MoE and Qwen; no hidden semantics |
| P02 Shared factory and existing routes | Controller, workers supply existing-model fixtures | P01 | Central registration/construction; migrate one existing dense and one MoE path without changed math/IDs; unknowns reject consistently; affected existing/training regressions pass |
| P03 Qwen adapter/state | 4090 worker | P01; native exploration may run alongside P02 | Inventory engine dependency closure; implement complete consumed-state adapter and optional-MTP metadata handling; integrate via P02; local mutation/frontier tests before admission |
| P04 Independent oracle | 96 GB worker | P00, P01 for adapter-facing interface | Forced-input reference and full-logit/controller comparisons, named negative controls, fixed criteria, independent no-checkpoint path; distinguish model-math vs continuation verification |
| P05 Toolchain and bounded placement | Surface worker | P00/P01; P03 for adapter runtime | GCC14/15 selection, Pascal/new CUDA builds, expert lease pressure tests and placement accounting; no eviction of live readers; cleanup and explicit admission failure |
| P06 Qwen G1 and placement equivalence | All machine owners test same accepted source | P02–P05 | Branch/frontier/controller/oracle tests; investigate all-resident vs hybrid differences on same forced histories; qualify exact device/format envelope only |
| P07 Persistence and recovery | 4090 implements; 96 GB tests; Surface checks constraints | P06 semantics | Strict zero-forward-work restore, explicit cold reconstruction, simultaneous tier copies, crash/retry/corruption/identity refusal and G2 checks |
| P08 Sharing and pressure | Ownership per exact file; all machine lanes | P07 | G3 actual work/copy/page/byte accounting, tail/recurrent isolation, transfers/leases, eviction and long-lived cleanup; BLOCKED if true sharing unavailable |
| P09 Serving and interface migration | Controller implements service delta; 96 GB conformance | P02/P06; recovery claims require P07 | PCC-owned client demonstrates setup/admission/serve/branch/restart; versioned changes and old-client failure behavior; no BW/SSS gate |
| P10 Training regression/extension proof | 96 GB worker, other device fixtures as needed | P02, correct training source/assets | Preserve current independent LoRA/full/optimizer contracts with actual fixtures; immutable candidates and resume mismatch rejection; no implied Qwen gradients |
| P11 Evaluated futures and release stress | Controller coordinates all workers | P07–P10, applicable G3 | G4/G5 mapped cases, >=1000 reproducible sequences, soak, native diagnostics, access/GC/upgrade/rollback; every advertised combination traced to evidence |
| P12 Final integration and handoff | Controller, independent review by conformance worker | Passing slice dependencies; full milestone as applicable | Clean integration/main build/test, source backup, support ledger/interface docs, exact qualifications and unresolved work |

Parallel schedule:
- Wave 1: P00 source/model-state inventory, oracle feasibility/fixtures and machine
  preflight concurrently. No simultaneous invention of incompatible common APIs.
- Wave 2: after reviewed P01, P02 common migration, P03 adapter, P04 oracle and P05
  build/placement fixtures proceed in disjoint files. Controller retains a slot;
  there are three workers, not one new worker for every ticket.
- Wave 3: all three test the same P06 integration commit using device-specific
  profiles. Then P07/P09/P10 can overlap where dependencies and file ownership allow.
- Wave 4: pressure/sharing, evaluated exploration, release stress and clean merge.

Training regressions are run when shared code changes, not postponed until P10
if an earlier slice can break them. P10 consolidates broader qualification.
Unavailable A100/HIP/SYCL hardware or real-model assets are explicit blockers for
those claims; the three machines are CUDA coverage, not a substitute for AMD.

P00 must also produce a mechanical support checklist: each factory branch,
operation, codec, currently registered model/profile and claimed evidence entry.
P01 selects an extension exercise: an existing compatible model is registered via
definition/mapping/fixtures only. Record shared-layer edits; unexplained scheduler
or store model-name edits fail the extensibility acceptance criterion.

## 10. Transition tables required before state changes

For each machine below, write concrete existing-to-proposed mappings. Use the
actual source's states; these are dimensions, not mandatory replacement enums:

| Dimension | Authority / guards | Tests that must exist |
| --- | --- | --- |
| Worker health and active leases | Owner, native validity, active lease count, deferred close | Wrong owner, double release, close while leased, rejected input vs partial native mutation |
| Request execution | Admission plan, cancellation boundary, accepted input frontier | Cancel before/after consume, pause vs EOS, exception cleanup, idempotent retry |
| Durable publication | Store transaction, branch revision, operation digest/receipt | Crash before/after commit, lost response, repeated operation ID with same/different args |
| Numerical materializations | Snapshot identity plus set of independent GPU/RAM/disk copies, each with validity, codec, checksum, readiness and readers | Export while GPU still live, RAM eviction with disk retained, disk deletion during active lease, failed transfer, corrupt replica |
| Expert/transfer cache | Cache key, budget reservation, in-flight transfer/read ownership | No eviction under a reader; budget exhausted by pins; cancellation waits or quarantines correctly |
| Training | Completed-step state, dataset/cursor/optimizer identity, candidate hash and explicit activation | Resumed vs uninterrupted, mismatched stream, cancellation, failed save, unchanged base and pinned inference |

Every transition row needs trigger, preconditions, locks/leases, changes, visibility
and durable commit point, allowed concurrency, postcondition, failure cleanup and
restart behavior. A snapshot with GPU+RAM+disk copies is a normal state. Export
creates a copy; it does not automatically evict the source. A failed replica does
not invalidate retained logical history or another validated copy.

## 11. Test design and non-negotiable counterexamples

Freeze the profile and thresholds before candidate evaluation. Establish baseline
roundoff using unchanged math and forced token histories. Compare full logits at
prefix frontier and every suffix step; do not compare diverged sampled histories
as if they were the same input. Report exact structural/controller checks and
numerical checks separately. Cross-GPU bit identity is not assumed.

Two independent checks are needed:
- **Model execution:** validated reference math/weights and declared numerical
  envelope; small independent FP64 fixtures where practical plus real-model
  comparisons. Readable text is insufficient.
- **Continuation machinery:** expected computation cannot read SUT checkpoints or branch caches.
  Reconstruct expected controller state from fixtures/history and reference logic.
  The harness may observe SUT controller bytes only as actual values to compare,
  never to initialize expected state. Sharing validated kernels may verify state
  mechanics but does not independently certify those kernels' math; record that
  limitation. Expected values cannot come from the failing implementation.

Minimum regression categories (map to existing test IDs, add missing tests):
- Prefix/page/window boundaries, BOS/empty policy, EOS and sampled-but-unconsumed
  token; stale/missing logits; recurrent/convolution state omitted on purpose.
- Interleaved A/B branches, rewind, repeated save, full/partial page mutation,
  identical suffix after different prefix, controller chunking and RNG draw cursor.
- Grammar empty support, penalties, stop strings and split UTF-8; no silent
  sampler reset or duplicate ingestion during restoration.
- Missing/corrupt/truncated checkpoint; wrong weights, adapters, template,
  positions, codec or execution profile; unavailable referenced input artifact.
- Strict restore with a deliberately instrumented forward call must fail;
  reconstruction must replay fixed historical tokens without RNG draws/tool effects.
- Simultaneous copies and readers, transfer failures, allocation OOM, expert cache
  pin exhaustion, cancellation at each mutation/commit boundary and process restart.
- CAS races, ABA, duplicate request receipts, timeout after commit and recovery
  of committed-but-unacknowledged results.
- Training gradient/update/optimizer resume checks appropriate to each declared
  operation, dataset/cursor mismatch, frozen base and explicit candidate activation.
- Unauthorized history access, retained-root GC, immutable evaluation, no-valid-
  winner handling, incompatible client versions and diagnostic error usefulness.

Show each selected mutant actually fails its named test. A test that only asserts
that a helper returned success is not a correctness oracle. Small fixture tests
and fake backends are useful if explicitly labeled; they do not replace real
native model, filesystem restart or hardware qualification.

Run G0 and applicable completed gates for code slices per repo instructions:

```sh
cd experimental/branching_context
python3 scripts/verify.py --gate G0
```

Run in a ticket checkout with generated evidence excluded from commits. Inspect
registered tests and evidence requirements before higher gates. The runner is
cumulative and stops at the first incomplete gate; never edit receipts to skip it.
Individual G4 behavior tests may run diagnostically while G3 is blocked, but
cumulative G4/G5 cannot be certified until G3 passes. No full-release claim is
permitted in that state. A G1/G2 opt-in slice may merge with its narrower label.
An evidence JSON with fields filled is not proof tests ran. A newly observed
baseline failure remains a failure; characterize and fix or block the affected
claim before depending on it.

## 12. Evidence packet and worker handoff

Every run records:
- UTC start/end, task/run ID, source SHA and dirty state, native/dependency/oracle
  revisions and executable hashes;
- model/weight/tokenizer/template/fixture hashes, devices, driver, toolkit,
  compiler, build flags, operation/model/profile/codec identities;
- actual command, exit status, seed/controller policy, prefix/suffix/positions,
  sequence lengths, branch counts and predefined numeric criterion;
- test counts/pass/fail/skipped/xfail/not-run, worst errors and failing fixtures,
  actual work/allocation/copy counters, peak memory, cleanup and residual processes;
- which facts are real native observations, simulations, historical results or
  assumptions; raw log locations and hashes, without secrets.

Evidence is immutable per run. Do not overwrite an earlier failure with a rerun.
Machine-readable totals and command exit codes must agree. A human summary may
explain failures but cannot relabel them PASS. Device-specific profiles bind
evidence; a 96 GB pass cannot qualify the 6 GB Surface by analogy.

Worker return template:

```text
Task / base / resulting commits:
Owned files changed / uncommitted work:
Invariant and observable behavior changed:
Tests: command -> outcome -> evidence path/hash:
Negative control and independent oracle:
Observed failures / skips / untested boundaries:
Interface or persisted-format changes:
Machine job state / resource cleanup:
Next dependency / exact blocker:
Review requested: <specific commit range>
```

Status updates explain completed evidence and the next uncertainty. Immediately
report numerical/structural divergence, unexpected identity, permission trouble,
resource/thermal faults, conflicting file ownership or required contract changes.
Ask the controller for a revised task, not the human for routine implementation
choices. Escalate material scope or permission changes to the human once, with
concrete evidence and independent work continuing where possible.

## 13. Controller review, merge and recovery loop

For each returned ticket:
1. Read the diff and source, compare owned paths and contract revision, verify
   full commit IDs and that no secrets/generated payloads are staged.
2. Inspect evidence/commands/exit codes, independent oracle and negative control.
   Require a reproduction of material new semantics by another worker or clean
   controller run; do not accept a worker's narrative as the only proof.
3. Check current support/training regressions, interface record and persistence
   implications. Historical receipts establish baseline claims; changed code needs
   fresh affected-regression evidence at the accepted source revision, or the
   affected claim remains blocked. Reject unqualified capabilities advertised as
   qualified. Missing historical hardware requires an explicit scoped qualification
   decision, never substituting another device's pass for that hardware.
4. Integrate the smallest coherent passing commit range. Re-run affected checks
   after conflict resolution and from a clean checkout of the resulting source.
5. Update support envelope and source backup. Merge to main only within existing
   authorization and after the slice gates pass; unfinished operations stay
   unavailable or explicitly opt-in/unqualified, never silently admitted by default.
6. Record ACCEPTED with the integration SHA and exact scope. Choose the next
   runnable task from dependencies, correctness impact and uncertainty.

Before a pause, compaction, controller restart or worker replacement, persist:
accepted SHAs; branches/worktrees and dirty paths; task graph; current source/API
contract revision; machine lock/job identities; exact resume/cancel commands;
evidence index; blockers; next task; any human approval still required.
After resuming, re-read that ledger and inspect actual Git/process state. Never
start duplicate jobs merely because the previous chat ended. A lost worker does
not release its running process; an expired heartbeat does not prove it is dead.

## 14. Stop conditions and authority

A worker stops the dependent action and reports when:
- It lacks exact model/source/toolchain identity, a verified machine or required
  task ownership; another job owns the GPU or files.
- A contract is ambiguous or requires a shared interface change beyond its ticket.
- A failure invalidates a prerequisite/shared contract or threatens machine safety.
  Stop dependent runs and request a contract correction or reassignment.
  An expected failing red test is part of test-first development: preserve it and
  proceed with the scoped repair. An unexpected failure within owned scope is
  reported and investigated/repaired within that scope. Do not abandon normal
  red/green work or train around a correctness defect.
- Native state validity becomes uncertain, required resources are unavailable,
  hardware errors appear or a job exceeds its recorded bounds.
- Required evidence cannot be produced on available hardware/assets. Do useful
  independent source/fixture work; mark the affected claim BLOCKED.

Routine isolated builds/tests and small deterministic training qualification
fixtures are part of this plan. They do not authorize broad learning campaigns,
model replacement, automatic promotion, firmware/kernel changes, unrelated user
process termination, host firewall changes, physical erasure or key copying.
Do not install new system dependencies silently; first use existing environments
and document the exact need. Preserve source and failed evidence even when a test
workspace can be recreated. Clean only explicitly owned temporary artifacts.

## 15. First dispatch after this plan is accepted

Controller starts P00 and fills the real ledger. Dispatch three bounded tasks:
1. 4090 worker: inventory Qwen engine/state closure and proposed adapter hook points;
   inspect current factory branches and one existing dense/MoE characterization.
2. 96 GB worker: map historical operation receipts and runnable tests; draft a
   forced-input oracle and negative-control fixture contract without using SUT
   saved state. Inventory existing training regression assets.
3. Surface worker: verify exact compiler/device/model and resource baseline; map
   low-memory expert leases and required Pascal build configuration. No competing
   GPU run until the machine lease and test limits are recorded.

Controller consolidates P00, resolves gaps and reviews P01 with the workers before
parallel runtime implementation. This document revision itself only inspects and
plans; it does not claim P00 baselines or new Qwen PCC qualification have passed.
