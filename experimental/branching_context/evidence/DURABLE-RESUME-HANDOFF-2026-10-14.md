# Durable resume handoff — historical broader-contract result

Current work uses the accepted matching-native-batch restriction described in
`docs/context-cow/PROFILE_BOUND_RESUME.md`. The broader mismatch recorded below
remains evidence but does not block the restricted milestone. Active checkout:
`C:\Users\dflanag3\Documents\fleet\context-cow-execution-parity`, branch
`feat/batched-durable-resume`. Implementation/control tests are committed in
`bc2de8f5`; tiny GPU qualification is pending in matching-resume-20261014-v2.
Consult STATUS.md for current executed results. This document retains the older
profile's measurements and must not be read as current batches-only validation.

Checkout: `C:\Users\dflanag3\Documents\fleet\context-cow-durable-resume`.
Branch: `feat/durable-computation-resume`, base `c8e9c09b`, initial checkpoint `c840ac81`; subsequent code/evidence is on this branch.
Dedicated private repository: `box-git:/srv/data/git/private_context_cow.git` (remote `context-cow-private`). Keep code and evidence here until the implementation works fully and is reviewed. No GitHub push occurred; Strata upstream and the Strata-a5500 origin were not updated. The earlier `strata-private.git` backup remains separate.

Read `docs/context-cow/DURABLE_RESUME.md` and its graph/summary first.

## Executed result

Fast batched P: 128 token IDs. Two different suffixes: four tokens each. Outputs: eight each. New Python/native processes loaded the 121,413,624-byte native root file with zero prefix forward work, in 0.385/0.352 seconds after ~67 seconds engine startup. Both continuations matched uninterrupted tokens, controllers and frontiers. A third process resumed after four generated tokens and reproduced the remaining four. Audit-inclusive Python continuation was ~0.76 TPS; native compute was much faster. This is not a serving speed claim.

Full-history agreement failed after P plus `[117,118,119,120,198]`: max logit error 1.9153637886047363 at vocabulary index 22023. Normal next top token: 320; full batched: 309. Restored and independently uncached normal decode are bit-identical; the same disagreement exists between the uncached operators. First differing layer remains unknown. Frozen atol=0/rtol=0 unchanged. 1K/8K stopped before execution.

## Counts and omissions

- G0: 51 PASS, no failures/skips.
- Supplemental control plane: 11 PASS, including ten process-kill points with synthetic native bytes and distinct old/new histories.
- Evidence writer: two PASS; normalized infinity at zero tolerance is explicit.
- Real audits: DR-001/002/008/009 PASS; DR-003 FAIL. Five executed, zero skips.
- Cumulative G2 stopped at G1: zero methods, four missing-CTT_PROFILE setup errors. G2 did not execute.
- Nonzero RNG/grammar/pending UTF8, full native crash/OOM/eviction matrix, target repetitions, RAM/VRAM peaks and larger contexts remain unrun. No formal G1/G2/G3 PASS.
- Experimental 16 GiB retention caused OOM: 17 images, 1.55 GB live pages, 1.22 GB COW copies, cuMemCreate error 2, zero integrity mismatches. Original 1 GiB bound removed it without deleting history or disabling checks.

GPU evidence predates final controller-rejection/reporting guards. Those have CPU evidence; current source has no complete GPU qualification. Raw artifacts and frozen source are bundled with documentation. Native payloads remain on the host, not in Git archives.

## Machines and retained paths

llm-60: RTX PRO 6000 Blackwell, Q8_0, FP16 KV, driver 595.91.07, Linux `python3` 3.12.3. Windows commands use `python` 3.13.15.

Root: `/home/dflanag3/branching-g3/durable-resume-20261013/`.

- `experimental/branching_context/`: frozen measured source. Do not edit; profile hashes these paths.
- `source/`, `counter-binaries/`: measured native sources, generated CUDA executables and build receipt.
- `tiny-bounded/`: canonical store, saved payloads, traces, frontiers and diagnosis.
- `tiny/`, `tiny-v2/`, `diagnostic/`: argument failure, baseline and OOM attempts.
- `reporting-api/`: later CPU-tested controller/reporting guards and final audit logs.

llm-49 completed the initial CUDA base build and hashed/verified 189,721,071,389 model/pack bytes. No inference campaign ran there; final counter changes were not qualified there.

The chat server is restored on llm-60: Python PID 1426679, native PID 1426681, port 8097, original config SHA-256 `4083878345d5cd966e187d3da9eae26ec538a7cfe8858c17880903af9d968227`. Authenticated workspace HTTP 200. It holds nearly all GPU memory. No campaign was left running. Inspect current jobs before GPU work. The user authorized temporarily stopping this server and restarting it unchanged.

## Next bounded ticket

User subsequently scoped the milestone to batches only. See
`BATCHED-RESUME-SCOPE-2026-10-14.md` and STATUS.md. Continue in
`C:\Users\dflanag3\Documents\fleet\context-cow-execution-parity` on
`feat/batched-durable-resume`; use the dedicated private remote. Normal decode
is a separate research track with further investigation/fixes not started.
Preserve the earlier exploratory normal-decode evidence above.

The batch-only uncached preflight already reproduces a discrepancy: 128+4+1
versus 128+5, identical 133 IDs, max error 1.4495954513549805. A diagnostic
preserved both output hashes and located the first observed mixed-activation
difference at layer 0's FFN input (0.0007425844669342041); its attention input
was exact. GDN-operation readback established identical FP16/BF16 input images
and the first observed unequal result at QKV projection (1.1444091796875e-05).
All diagnostic frontiers matched their uninstrumented hashes. See
`docs/context-cow/BATCHED_RESUME.md`, its summary and raw archive. BP-001/002 PASS,
BP-003 FAIL, no skips. Current-profile restart and larger fixtures did not run.
Missing primitive is batch-partition-invariant native dense projection under
the frozen tolerance. Isolate it using these exact input images and pinned
weights, then implement/measure a stable batched projection. Do not blame restore
or change tolerance. Final server restoration: Python 1441826/native 1441828,
authenticated HTTP 200; no diagnostic left running.

Fix or specifically characterize that primitive with a small deterministic regression. Preserve the mathematical contract, registered assertions and frozen tolerance. Rebuild touched CUDA code and retain fingerprints. HIP/SYCL remain unqualified until separately built/tested.

Then run current-source tiny restart into a new output directory/profile. Verify zero prefix work, finite frontiers, forced A/B suffixes and generated boundaries against fresh full history. Only after that passes, run advanced controller/crash/eviction cases and expand to 1K then 8K/512/up-to-128. One inference job at a time on llm-60; restore the chat server afterward. Do not report G2 PASS without cumulative G1+G2 evidence.

The frozen `boundary` command in the main document can repeat the preserved normal continuation. It does not resolve full-batched equivalence.
