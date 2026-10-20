# Native-speed Context COW for one active request

Current entry point: [CURRENT-HANDOFF.md](CURRENT-HANDOFF.md). The integrated
`1538b8b3` source passed CPU 352/352, G0 51/51, BranchWrite client 19/19 and the
13-command real-Q4 seed/STOP/restart CLI flow. This adds real-GPU positive-path
coverage for the canonical reseed guard. The separate fresh numerical audit
passed 58 exact full-vector comparisons (51 fresh-repeat, seven SUT comparisons
covering six durable frontiers), maximum error zero. A current-source resident
8K CLI run measured 110.3-111.6 generation-loop TPS and 54.6-55.1 whole-CLI TPS,
without prefix recomputation or file reload. Publication analysis now passes
offline: prepare 0.510-0.524 seconds and writer publication 0.251-0.347 seconds,
with independent full-file readback. Assembled helper integration passes CPU
385/385 and G0 51/51. Preparation subphase attribution is the next diagnostic;
no validation or durability operation has been removed. Historical throughput
measurements below retain their original source/profile scopes. Full objective
and cumulative G1-G5 remain unqualified; Q8 deferred; private only; chat OFF.

2026-10-16 Q4 automatic CUDA allocation-failure recovery is FOCUSED-PASS,
including independent numerical/source audits. The same HTTP service now
preserves the original error, closes the poisoned native child before starting
a distinct replacement, waits for healthy state, and serves a new operation.
The failed operation is not automatically retried or acknowledged as durable.
A separately forked branch also resumes after a clean HTTP/native restart.

Two real-GPU fixtures passed: 128+4 consumed tokens and 7,995+4 (7,999) consumed
tokens. Each subsequent continuation consumes eight outputs with forward=8,
prefill=0, one durable branch revision, exact independent-reference tokens and
full greedy controller, and zero final-logit error. Original canonical records,
branch head, manifest and full payload bytes remain unchanged. Tiny lifecycle
elapsed 65.441 seconds; target 68.805 seconds, including startup, deliberate
pressure and shutdown, not throughput measurements. Pressure helpers held and
released 11,379,146,752 / 11,228,151,808 bytes; one actual cuMemCreate OOM 2 per
fixture. Old children exit -9 after poisoning; replacements and independent
restart children exit 0. Both HTTP services and helpers exit 0, all six owned
PIDs absent per fixture. All campaign and independent-audit wrappers exit 0/0.

Target independent audit checks 18 fresh/preserved vector pairs plus both
resumed final frontiers (20 full-vector comparisons), max error 0. Tiny has its
prior 18-pair fixture audit plus a separate two-frontier recovery audit. This
checks final resumed frontiers and every returned token/controller/history;
it does not assert every restored intermediate logit or arbitrary arrangements.
Target source payload is 327,985,908 bytes. Each target continuation loads
326,992,628 native bytes, copies 75,497,472 and device-hashes 1,736,441,856 bytes,
with zero integrity mismatches. These are work counters, not peak-memory claims.

Reviewed production fix and strict harness are integrated on feat/model-free-fleet
through 62b4e466; API hashes exactly match the frozen GPU-tested candidate:
coordinator bb6a1f8d, backend d692e003, native 4bd5d715, profile a7d53c45,
execution fed7e7d5. Combined checks pass focus81, HTTP141 and G0 51, zero
failures/errors/skips/xfails. Actual source import paths/hashes and unchanged
source manifests are retained. See Q4-CUDA-RECOVERY-{TINY,8K}-PRIMARY and
Q4-CUDA-RECOVERY-INTEGRATED-{FOCUS,HTTP,G0}-2026-10-16.json.

Preserved setup failures: first tiny attempt stopped before fault injection
because a browser-history helper required presentation metadata; corrected
runner reads leased canonical Store state and uses existing plain token forks.
The first integrated HTTP source package omitted strata_tokenizer and stopped
before discovery; the new package includes tracked Python dependencies and
passes unchanged tests. Earlier faulty durable-event guards remain in their
historical stages. No backend, registered assertion or tolerance was weakened.

The old baseline FAIL-RECOVERY remains valid for its prior source/profile.
Q4 only; Q8 deferred; chat server off; private PC/box-git only. Raw recovery
evidence is backed up to private box-git: 86,947,840-byte archive, independent
readback verifies all 765 regular members (86,138,771 bytes), transfer/readback
command/wrapper exits 0/0. See Q4-CUDA-RECOVERY-BACKUP-PRIMARY-2026-10-16.json.
Numerical .state and model files are excluded; retained originals stay on
llm-60. Cumulative G1-G5 remains unqualified. Remaining work includes wider crash/controller/pressure coverage
and current-version end-to-end speed/client qualification; goal stays active.

After verified backup, only the completed tiny/target child checkpoint manifests
and their unique payloads were evicted: 450,928,640 allocated bytes freed.
All 40 checked canonical files and both original prefix manifests/payloads are
unchanged. Raw vectors and executed audits remain; complete child-payload
re-audits now require recreating those derived states. No inference job is live;
chat remains off. See Q4-CUDA-RECOVERY-COMPLETED-CHILD-EVICTION-2026-10-16.json.

Next practical ticket: use the existing CLI/HTTP workload tooling to requalify
several independent codebase trees on the integrated Q4 profile, one active
request, with fork/retry, RAM/disk switching, client TTFT/TPOT and durable-ack
latency. Inspect run_cow_multi_client.py's presentation-bound fixture needs
before reusing it; the token-only recovery fixture is not a browser namespace.
Keep numerical checks independent and preserve the greedy/profile scope. Do
not restart chat, run Q8, or publish publicly. Recheck physical disk headroom
before scheduling new checkpoint-producing runs.
2026-10-16 worker-recovery candidate now has a fresh, source-bound Q4 fixture
and independent numerical references. Coherent frozen runtime/profile v3
selects candidate coordinator bb6a1f8d and backend d692e003; profile SHA
 a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755,
execution ID fed7e7d5e4b2405fc75e6d300deef7576fe4a0a8ddec88841331144568ada450.
Actual selected-closure and imported-path checks pass. Model contract, weights,
execution settings and hardware match the base exactly; native binary remains
4bd5d715. Model checks reuse 15 receipts with zero model bytes hashed. This
profile is not broad numerical/G1 qualification. V1's stale selected-closure
binding and V2's read-only-copy setup failure remain recorded; neither ran
inference. No old checkpoint is retagged or imported into fixture preparation.

The new GPU producer prefills 128 tokens using the native batched path, consumes
four greedy outputs, saves an active 132-token checkpoint (122,511,368 bytes),
and exits native 0. Two separate uncached oracle processes each compute the
same base and 12 generated tokens, exit 0 and agree at all 12 generated
frontiers. Prepared base/P4 frontiers also match. Primary read-only audit checks
18 full-vector pairs, all byte-identical, max error 0, including the saved
payload's frontier tail; exact canonical inputs and stored controller export
are verified. The reference helper also compares the full greedy controller.
All three native PIDs are absent. Reference emits 12 uninterrupted outputs;
producer/recovery boundaries are 4+8 under the same native operator sequence.
No arbitrary batch-arrangement equivalence claim. See
Q4-RECOVERY-CANDIDATE-TINY-REFERENCE-2026-10-16.json. This does not yet test
restoration or allocation-failure recovery on the candidate profile.

Current candidate bytes also pass 141 synthetic HTTP/control tests, command/
wrapper 0/0, imported API hashes checked and Python sources unchanged. Initial
HTTP test packaging omitted a browser helper and template; later one child
lacked inherited PYTHONPATH. Those setup failures are retained; tests are
unchanged. The producer's first reference attempt failed before oracle startup
on a tokenizer-constructor call; V2 uses verified Tokenizer.from_gguf and its
actual no-model preflight passes. Corrected reference command/wrapper are 0/0.
See Q4-RECOVERY-CANDIDATE-HTTP141-2026-10-16.json and reproducible scripts.

The strict same-service allocation-recovery runner is under separate review,
not GPU-executed. It must prove original error delivery, observed old-child
exit, a healthy replacement under the unchanged HTTP PID, one durable branch
move after eight consumed outputs, and a separate branch after clean restart.
Tiny must pass before expanding to the 7,995+4 consumed target fixture. Root
owns GPU launch; chat stays off and no GPU job is currently live. Q8 remains
deferred. Cumulative G1-G5 remains unqualified; goal remains active.

After verified compact backup, only the completed old-profile OOM test's child
numerical checkpoint/manifest was evicted (328,204,288 allocated bytes freed).
All 15 audited canonical files and original prefix checkpoint remain unchanged.
See CUDA-OOM-COMPLETED-CHILD-EVICTED-2026-10-16.json. New raw fixture/reference
and HTTP evidence is retained on llm-60 under q4-cuda-recovery-20261016-v1;
its separate private raw-archive backup is still pending. Code/summaries are
privately committed; no GitHub or upstream publication is authorized.

2026-10-16 actual Q4 CUDA allocation-failure baseline is FAIL-RECOVERY.
The no-model preflight passes after separating the HTTP source pin from the
profile-selected API source closure. Native cuMemCreate reports actual CUDA
OOM 2 during private-tail preparation: one SUT driver/OOM failure, zero forward
or prefill work, zero new capture and no failed-operation durable receipt or
branch move. The original 7,999-token source and 327,985,908-byte checkpoint
remain unchanged. Helper allocates/releases 11,228,151,808 bytes (5,354 handles),
exits 0. The baseline adapter hides the native cause as an invalid token event
and retains a poisoned worker: a new same-service operation also fails.
Explicit fresh service/native restart succeeds with eight consumed outputs,
forward 8/prefill 0, loaded 326,992,628 bytes, copied 75,497,472 bytes and hashed
1,736,441,856 device bytes with zero integrity mismatches. First native exit is
-9, clean restart native exit 0; both service exits 0, command/wrapper 1/1.
All five owned PIDs are absent; GPU idle and chat server remains off.
No fresh numerical oracle or live GPU parent audit after poisoning ran;
numerical maximum is null. Whole experiment is 47.282 seconds, not a speed
benchmark. Independent read-only audit passes without turning the failing
recovery result into PASS. See CUDA-OOM-Q4-PRIMARY/QUALIFICATION-2026-10-16.json.

Worker replacement candidate is private commit 11cf9976 on
work/luna-native-terminal-error, separate from main. Primary review reproduced
three CPU cleanup failures, corrected ownership through malformed-profile and
trace failures, and retained any unproven child for final shutdown. Fresh
frozen sources record actual imported paths/hashes and identical full source
manifests before/after execution: focused 63/63 and G0 51/51 pass, exits 0/0.
The earlier worker 59-test claim cannot qualify current bytes: its staged
source was overwritten; the subsequent 60-test run failed. Historical logs
remain and the proposal explicitly withdraws that qualification. The final
candidate is GPU-UNQUALIFIED and not integrated. Main API/native/profile bytes
remain unchanged. Next: derive a new candidate API execution profile and create
a fresh compatible prefix, then exercise automatic same-service replacement
and independent continuation checks. Never relabel the old checkpoint.

The 17,111,040-byte compact baseline/CPU archive is on private box-git under
20261016-q4-cuda-oom-baseline-cpu-replacement-v1. Independent readback verifies
all 588 regular members, 16,490,885 bytes, no links or extraction. Transfer and
readback wrappers are 0/0. Scope excludes model/numerical state payloads and
complete historical stages; original artifacts remain on llm-60. See
CUDA-OOM-Q4-BACKUP-PRIMARY-2026-10-16.json. Q4 only, Q8 deferred, private only.
G0 remains PASS; cumulative G1-G5 remains unqualified. Goal stays active.

The active milestone is practical reuse of long code contexts on one GPU, with
one coordinator-owned native worker. Original cumulative G1-G5 certification is
separate and remains unqualified. The chat server stays off; work stays private.

The current execution contract is schema 3 `identical-native-operators-v1`:
batched input Prefill (8,192-token chunks) and consumed native Verifier greedy
generation (`normal-verifier-decode`, `consumed-greedy-ack-v1`). The independent
reference uses those same ordered operators and boundaries. The earlier schema 2
batch-only research path remains separate and incompatible. Keeping ordinary
Strata serving's cold/cached disagreement outside this prototype does not mean
the native Verifier operator is excluded from the current COW request path.

## Executed checkpoints

| Requirement | Evidence and current scope |
|---|---|
| Fast native prefill and generation with immutable branching | Q4/Q8 64K source campaigns; six continuations and three rewinds each; matching native operators, sampled full vectors and parent audits. |
| Q8 long outputs above the 30 TPS target | Relevant-code text restart: 2,042/2,048 outputs, received46.163/45.908 TPS, zero prefix prefill. SERVING-TEXT-RESUME-Q8-2K-2026-10-14.json. |
| Resume preserved computation after process restart | New producer/consumer Python and native processes, zero old-prefix work, exact tokens/full greedy controller, prefix/four/final vectors. Same profile/operator contract only. |
| Shared RAM/disk pools and one active request per GPU | Full96 GiB RAM/128 GiB disk profile; pool/control106; real campaigns max_active_per_gpu=1. Model-free fleet coverage does not qualify multi-GPU model inference. |
| Cancel a GPU request without committing provisional output | Direct native cancellation and actual HTTP socket-disconnect recovery; following request/reference exact, parent preserved. REQUEST-STOP-GPU and SERVING-HTTP-CANCEL Q8 profiles. |
| Literal-safe text and retry admission | Actual Q8 tokenizer/template probes, text HTTP, pending UTF-8 continuity;44 supplemental controls. No tools/attachments/non-greedy admission in this slice. |
| Add a new question to the actual saved history | PASS: actual EOS2042, suffix59+outputs128, no old-prefix work,43.580 TPS and0.691s TTFT. Independent multi-turn oracle matched all tokens/full greedy controllers and root/both final vectors exactly. SERVING-TEXT-NEXT-USER-Q8-2026-10-14.json. |

Measurements above are bounded, usually single runs rather than performance
medians. Native-only throughput and received HTTP throughput are distinct.
First-token waits exclude engine startup where stated. Numerical vectors were
sampled as documented; exact token sequences are not every-frontier certification.

## Current phase

Canonical branch-reference ENOSPC now passes on Q4 (2026-10-16). A model-free
ext4 filler check and 20 CPU controls precede GPU attempt 2. Independent audit
verifies real writer errno 28, unchanged old/source bytes, exact prepared
checkpoint retained, no-work same-operation retry, one move and a successful
following request. Native work is 24 output forwards, zero saved-prefix prefill;
G0 51 passes. No fresh numerical oracle or new speed claim is added. Actual
inference allocation recovery and broader combined pressure/crash campaigns
remain open; cumulative G1-G5 is still unqualified. The aborted first attempt
is preserved below and does not describe the final outcome.

Latest canonical-volume attempt (2026-10-16) aborts before the target branch
writer: a large filler write gets ENOSPC but 417,792 bytes remain after fsync.
The strict filler guard rejects that incomplete setup. Independent full-byte
audit verifies unchanged source/old branch, a valid 16-output candidate,
zero prefix prefill, clean service/native exit and removed filler/mounts.
This is a harness failure, not a tested backend rejection. The filler must
first pass a real model-free ext4 check, then the GPU case gets a new output
directory. G0 passes 51; production/native/profile bytes remain unchanged.

As of 2026-10-16, save/recovery hardening is the active subphase around the
existing native-speed Q4 CLI path. Q8 is deferred, chat remains off, and all
source/evidence publication is private. The completed recovery checks are:

- Actual checkpoint payload writer EFBIG, clear client error, unchanged old
  branch, same prepared bytes retried without inference and a following request.
- Real postcommit SIGKILL with missed acknowledgment: new Python/native
  processes recover the original operation receipt once, then continue without
  saved-prefix prefill. Initial native wait status is unknown; later PIDs absent.
- Actual checkpoint-volume ENOSPC: 128 consumed outputs await publication,
  errno 28 reaches the client once with no durable acknowledgment; retry does
  no native work and eight more outputs complete. Independent full-byte,
  canonical-history and greedy-controller audits pass. G0 51, final harness 16
  pass. See ENOSPC-Q4-PRIMARY/QUALIFICATION-2026-10-16.json. This lifecycle
  experiment has no fresh numerical oracle or throughput qualification.

The next distinct case is canonical branch-reference storage exhaustion while
checkpoint storage remains available. A private 64 MiB fault filesystem and
separate checkpoint volume are prepared; the harness is under test-first
construction. This test device does not change the 96 GiB RAM / 128 GiB disk
deployment profile. No canonical-volume GPU pass is recorded yet. Actual
inference-allocation recovery, more crash stages, sustained tree workloads and
combined end-to-end pressure remain outstanding. The original intermittent
strict-stats inequality is retained as unexplained; later passing repeats did
not identify its cause. Cumulative G1-G5 remains unqualified.

The following observations describe earlier checkpoints, not new gate passes.

Newest actual-CLI Q4 workload passes automatic retention: one 7,231-token
codebase prefix, three independent questions, 967 total outputs ending at EOS.
The 5,000-output setting is a ceiling, not an executed count. All requests use
the same native root image with zero file-load bytes and zero prefix forward
work. Whole CLI rates including suffix and durable completion are 80-82 TPS.
No fresh numerical oracle ran; auxiliary state may still reside on the host.
The serving history resolver now distinguishes repeat publications by captured
branch revision. Real Q4 kernel write-failure/retry passes after HTTP writer
errors are distinguished from disconnects and duplicate terminal notifications
are removed. Same-operation retry does no native work, one branch advance is
published, and the next eight-output request succeeds. Parent bytes remain
unchanged. The 28 focused/141 serving/G0 51 suites pass with overlap. No fresh
numerical oracle ran in this recovery check. At that earlier checkpoint real
ENOSPC and broader crashes remained open; a TCP half-close extension was rejected for a cancellation
regression. Q8 remains deferred; no full-gate claim is added.

The practical CLI and opt-in COW-owned serving/presentation path now exist and
have bounded real Q4/Q8 client results. Combined source9eaf passes182 CPU methods,
141 HTTP methods and G0 51. New Q4/Q8 root reuse tests on that source each pass
8192 prefix/512 suffix/128 output: distinct operation IDs reuse one root,
new Python/native processes load it from disk without prefix computation, and
three comparisons match tokens, tested greedy controllers and prefix/final full
vectors exactly. These probes do not check suffix/every-generated frontiers.
See ROOT-REUSE-{Q4,Q8}-8K-2026-10-15.json and the current STATUS/handoff.

Finite cache capacity is now integrated through `21924f2b`. Inactive branch
roots/tips retain canonical history but no longer pin numerical copies forever.
Live read/publication leases still protect in-use copies. Process-shared byte
and digest reservations protect publication; missing-source execution reports
reconstruction while strict restore continues to reject missing checkpoints.
Main API files match the GPU candidate byte for byte. Combined checks pass 258
implementation/client/harness methods, 141 serving methods and G0 51; the later
size-parameterized harness passes 29 focused methods.

Actual pool-driven eviction passes twice on tiny Q4 and once on tiny Q8, with
four evicted numerical files, unchanged canonical history/controllers/references
and old receipts, metadata-only fork/retry, explicit reconstruction and exact
independent prefix/final comparisons. Q4 has also passed 1024/64/16 and
8192/512/128 fixtures; Q8 now passes the same 8192/512/128 target. These
are correctness/recovery observations, not isolated cache-speed measurements.
The first tiny Q4 rejection-control stats inequality remains unresolved;
subsequent runs retain full before/after records and keep the strict assertion.
Further real pressure, publication failure/crash, client retry and long-output
speed checks must cover this combined path.

The former eager backend's target generation observations were about 73.4 TPS
Q4 and 31.7 TPS Q8, excluding reconstruction/capture/durable publication.
Physical-prefix allocation is now integrated at `420a60b7`: stable virtual
capacity remains, while physical mappings grow with validated consumed extents.
Every allocated sealed page, including complete physical tail padding, is still
freshly checked before growth or detach. The original arena and full-capacity
integrity scratch remain allocated. Ten CUDA test programs and G0 51 pass.

Current-API Q4 tiny/1K/8K and Q8 tiny/8K each pass resident and fresh-process disk
reuse with zero old-prefix work and exact tested tokens/controllers plus sampled
prefix/final vectors. Q8 8K gives 36.16-37.59 generation TPS; the user accepts
30+ TPS as a useful baseline. The separate repeated Q4 comparison gives median
74.48 -> 114.87 generation TPS and 3.817 -> 2.411 seconds submit-to-durable across
six measured requests per backend. All 16 requests including warm-ups match
their reference final vectors. This is not a Q8 speedup measurement. Keep the
old-then-new ordering and separate generation versus whole-request timing clear.
Actual Q4 and Q8 8K pool eviction/rebuild on this newly integrated native candidate
passes, including strict rejection, canonical retention and explicit rebuild
agreement. Both commands/native workers/oracles exited zero. Real Q4/Q8
cancellation after three consumed outputs now also passes on the sparse backend:
no publication/reference move, then two independent 128-output recoveries per
model with exact IDs, tested greedy controllers and final full vectors (error 0),
no prefix work and unchanged parent bytes. Ten focused CPU checks, two validator
bypass mutants and G0 51 pass. See SPARSE-CANCEL evidence integrated as `bf3124c4`.
The test reuses verified previously executed independent reference vectors;
cancelled outputs are checked as IDs/receipts, not full logits. Actual Q4 CLI/service
requalification now passes on the sparse backend: 13 commands, 7,231-token code
root and four 128-output calls including exact restart continuation with zero
prefix prefill. Both native/service processes exit 0. No fresh logits oracle
or retry-request is claimed in that CLI check. Both A resumes load serialized
checkpoint bytes; this does not prove a pure resident-only path. See SPARSE-CLI
evidence and compact private backup. User deferred Q8 follow-up; next verify
automatic same-root retention and real write rejection/publication retry on Q4.
No explicit residency-lock API is being added. Publication crashes and longer
combined contexts remain open.

An isolated rolling CUDA SHA kernel preserves the full integrity policy and
measures 1.197% less event-total time for a 2.5 GiB comparison across six samples.
Keep that improvement even if small; whole-request TPS benefit is not measured.
See CUDA-HASH-ROLLING-MICROBENCH-2026-10-15.json. Canonical-only fork/retry is
integrated as `c58ba87e` with its tested coordinator bytes unchanged; it still
has only bounded qualification. Its real Q4 workflow now passes after all
numerical files are evicted: metadata-only fork/two idempotent moves, explicit
140-token reconstruction and exact prefix/final comparison. See
COORDINATOR-BRANCH-ACTIONS-Q4-TINY-2026-10-15.json. Shared reservation and
publication-admission fixes are now integrated; see the capacity and actual
pool-pressure evidence in STATUS.md. Full-byte validation remains enabled.

## Remaining practical integration

Keep the dedicated COW path and its presentation projection distinct from
ordinary Strata serving, whose cold-versus-cached disagreement remains outside
the supported prototype. Store refs select execution heads; presentation is
derived from durably bound receipts. No second execution history or provisional
output presented as an acknowledged save is allowed. Friend testing uses the
terminal client connected to the existing server; v5 is the current private
client-only package and does not install an independent inference engine.

Active output-limited answers currently resume the same turn; they cannot accept
a new question until actual EOS. Explicit user interruption needs its own tested
consumed-boundary operation. General sampling, nonzero RNG/grammar, cross-batch
equivalence, multi-GPU real-model serving and cumulative release certification
remain separately scoped limitations. These incomplete items prevent declaring
the full active milestone finished solely from the performance checkpoints.
