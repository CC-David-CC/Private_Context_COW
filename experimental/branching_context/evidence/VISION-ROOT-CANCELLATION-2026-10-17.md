# VC08 initial-image cancellation: focused PASS

If a client goes away while its picture is still loading, COW should stop that
new request and keep the user's saved work. Both initial image routes now do
that in the tested case. If publication has already started, it finishes and the
client must recover the original operation rather than assume nothing happened.

The missing path was HTTP root -> text adapter -> token adapter/coordinator.
Suffix requests already carried cancellation; create/context/prepare did not.
Cancellation now reaches root preparation as transient control, without becoming
part of canonical history or the operation signature. The worker checks again
after native prefill and before creating the root branch reference. A CUDA
prefill batch is not interrupted halfway through. Unpublished computed records
may be unreferenced GC candidates; they are not acknowledged snapshots.

Test-first evidence:

- REDv1: context incorrectly published after disconnect; the second arm used an
  already-cached image and failed its intended encoder-entry precondition. Both
  outcomes retained. Distinct images correct that fixture without changing the
  publication assertion.
- REDv2:3 actual distinguishing failures across2 methods: context and prepare
  publish after disconnect; cancellation after consumed prefill still publishes.
- GREEN: both methods pass. Related checks cover150 unique methods (expanded
  run160 executions because an imported TestCase exposed10 duplicate methods).
  Final supplement imports its fixture module instead; its2 checks reran green.
  Separate full BranchWrite357/357 and G0 51/51 pass. No final failures/skips.
- Supplement verifies prelaunch cancellation does no encoding/forward work.
  It also verifies cancellation after publication starts preserves the durable
  result, and the original operation ID returns that result without new work.
  These are real coordinator/store/thread tests with explicit CPU model fixtures.

Actual Q4 CLI/HTTP/GPU campaign, llm-60/GPU0:

1. Prepare and durably save an image-containing root.
2. Start a real CLI context request with a different image. Observe its exact
   server-owned encoder PID with14MiB allocated through nvidia-smi, then SIGTERM
   only the CLI group. The socket disconnect reaches the real encoder.
3. Repeat with the prepare route and another image.
4. Both operations become cancelled, with no durable receipt or new branch.
   Native forward/prefill deltas are0; original root payload/controller/heads
   unchanged; staging empty and RAM/disk reserved bytes0. Both encoder PIDs and
   CLI PIDs are absent.
5. Fork the retained root and answer with the512-budget thinking grammar. Thought
   closes naturally at33 tokens; final OK,36 consumed outputs including closure
   and EOS, forward36/prefill0. This is a continuity/control check, not a model
   capability benchmark. Original root remains unchanged.
6. Server145079/native145085 exit0, absent, no forced cleanup. Whole audited
   campaign27.302 seconds including model startup. No throughput benchmark.

A diagnostic wrapper around the real preparer records actual native counters
on their owning worker thread. Its context lifetime can include subsequent
consumption/publication for a successful root; both cancelled roots fail before
yield/forward. It substitutes no model inputs, outputs, cache or encoder. The
wrapper source and exact generated bootstrap are retained.

Fresh execution profile:
1fcece653938cf4795c522b5cd8c7595e7c353d707a956d037f8e8407d82f868
Runtime: cow-root-cancel-green-source-v1; profile: cow-root-cancel-profile-v1/profile.json
under /home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/. New store, unchanged
native executable/model/math/samplers. Do not relabel earlier checkpoints.
Runner: experimental/branching_context/scripts/run_vision_root_cancel_probe.py.
The executed launcher and full argv are in root-disconnect-gpu-run-v1.

Private backup: /srv/data/git/private_context_cow.evidence/20261017-vision-root-cancel-v1/evidence.tar.gz
3,810,840 bytes;203 members; SHA256
306e9a714df2b2bee3262e3580a231052ed60781bbd8bda818ebbed52cfbbcc7.
Receiver independently re-read every member and verified size/hash.258,277,392
optional native checkpoint bytes are excluded with individual hashes. Later
retirement is recorded separately; canonical records/controllers/inputs, PNGs,
raw counters, logs, profiles and source remain backed up. REDv2 sources were
reconstructed from basebb68716e plus unchanged executed tests after the run;
this is explicitly not a claim of pre-run freezing.

This covers parts of VC-022/VC-024, not complete VC08 or original G1-G5. Actual
GPU cancellation was during encoder startup allocation, not deep in an encode
kernel. The post-prefill guard is CPU-distinguished; the GPU requests cancelled
before model work. No full-filesystem/OOM/publication-crash/power-loss or fresh
full-logit oracle ran. Numerical maximum error is unmeasured. Cancellation
lookup remains coordinator-process scoped across this test; restart persistence
of cancellation decisions is not established. New-user thinking suffix support
remains guarded; this test uses initial roots/initial answer. Q8 deferred, chat
OFF, private LAN only, Windows steering only.
