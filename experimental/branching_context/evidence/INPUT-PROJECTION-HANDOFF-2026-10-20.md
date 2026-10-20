# Input metadata speed work: exact state at handoff

Worktree llm-60:/home/dflanag3/src/context-cow-input-projection-20261020
Private branch feat/derived-input-projection-20261020, base780e824a. Live API still
uses /home/dflanag3/src/context-cow-client-setup-20261020 at4b1f8fda and its original
Q4 native process1003805 (revalidate identity). Ordinary chat remains OFF. No GPU
job is scheduled by this controller. No live server files have been edited.

IP01 commit cf055d8d caches immutable derived span/token views and fast text append
while preserving the exact canonical bytes.98 regressions,51G0,4reference mutants;
CPU64K metadata20steps0.930s serving/0.747s base/0.255s candidate. IP02 adds backend
pure-hash cache reuse;105 regressions,51G0,4mutants. Keep the cursor's one fresh
reference calculation: the initial broader cache attempt failed an existing test
and was removed without changing assertions. New IP02 has a small total CPU gain
and faster complete publication audit. All reference hashes and schedules match.

Evidence:experimental/branching_context/evidence/runs/input-projection-20261020/.
These are CPU metadata measurements, not GPU throughput or G1-G5 qualification.
The original vision_execution.input_binding equations and full validators remain
unchanged. New tests include changed encoded bytes, mutable subclasses, image
metadata, independent branches, eviction bounds, detached reads and weakref
retention. No new persisted format or public runtime flag was added.

Next valuable ticket: measure faster fresh text-position byte construction using
stdlib array/strided assignment, preserving the exact little-endian iii bytes.
This must be compared byte-for-byte with the frozen scalar reference over empty,
text/image, grid/offset, large and overflow cases; do not skip the fresh check or
change hash formulas. Prefer a separate derived implementation with an unchanged
independent reference, and profile before selecting it. IP02 itself only saves
about1% total CPU in the64K microbenchmark; do not exaggerate it.

Then qualify a measured CPU winner on Q4 with a bounded real native campaign,
exact matching profile/boundaries, parent/sibling audit and full frontier checks.
One GPU job at a time. Inspect actual current source/profile/native executable
hash before stopping the owned API; preserve its launch configuration/key/store,
restore service afterward and keep chat OFF. The old binary's live SHA256 was
ae62045f75a8e11ba9d01b4da301d3bb87fe4ff342bab0ea7edc57276377824e;
revalidate rather than assume the filesystem binary matches it. Actual old
profile:62a8bc71db2436477fc6e5c010d4357287034c04ea41123a887418694a5d42b6,
Q4/FP16KV/capacity98304/prefill512/normal-verifier-decode. Do not claim MTP disabled
from the operator name; inspect argv if relevant. Same-profile qualification is
separate from paused cross-arrangement equivalence and cumulative gates.

BranchWrite current private worktree is /home/dflanag3/src/branchwrite-remote-admin-20261020,
private bounded-remote-admin branch at9577af5c. Installed dev54 verifier feedback
has460 CPU/client regressions and2 installed-shell cases, not a GPU self-repair
run yet. Dev53 Min-P follow-up completed in746.551s with exact local/deployed files,
formats and buttons checked in desktop/mobile Chromium. It reused50684 old cells
and evaluated exactly16780 new cells across9 native requests; no measured prefix
replay. Long-context metadata/capture overhead remains, and client/guest URL
scope needs clearer host facts. These product gaps remain part of the active goal.


## IP03 continuation

IP03 is implemented and CPU-tested. See runs/input-projection-20261020/IP03. Source changes only full publication-audit packing; no per-token scalar check removal. Fresh standalone binding benchmark is not a GPU speed claim. Full workflow has controlled reversed-order samples and identical digests. 114 focused regressions, G0 51 and four mutants passed. Next: derive a source-closure profile including packed_input_binding.py and current commit, preserve actual native executable/configuration and vision-config, then bounded real-GPU matched-schedule branch/frontier/controller/restore and interactive pause/steer qualification when the current BW Blender/carrier jobs settle. The old prepared IP02 profile does not pin current bytes and must not be launched unchanged. Do not kill or restart unknown/pending client operations.
