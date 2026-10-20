# Priorities and design

Context COW is the focus. Host-RAM restore is a useful control and a valid
correctness fallback when physical GPU sharing is unavailable. The authoritative
representation stays one parent-linked immutable history plus complete controller
state; movable branch references and optional derived checkpoints serve it.

## 1. Keep the working, measured starting point

v0.1.0 has an aligned 8K batched prefix, two 512-token suffixes, exact fresh
frontier comparisons through 128 consumed outputs, resident/RAM/disk restoration,
parent byte audits, real page/copy counters, and independent corruption tests.
The current source base is the published Strata PR #2094 implementation.
The ordinary-main control showed no major serving-throughput regression, so
discarding this working base was not supported by the measurements.

The bottleneck was GPU write preparation: 1.66 seconds versus about 0.012 for
RAM, while model work was effectively equal. All-pool verification reduced
that to 0.191 seconds without reducing the byte coverage. That is why this
optimization came before a new streaming system or model-kernel rewrite.

## 2. Prepare ahead, with a small state machine

The current implementation allocates verification scratch at binding, then
constructs a linear page-address list and verifies it in one launch. No tree
search or separate history representation is required for that operation.
Its existing session transition is:

```text
unbound -> ready -> writing -> ready
                     |          |
                     v          +-> freeze / restore at a consumed frontier
                   failed -> restore committed state -> ready
```

The next candidate is a prepared writable window. During branch activation,
compute the upcoming tail range and reserve/detach its shared pages once.
During append, check the current position and ownership generation, and reuse
the private writable tail. Invalidate this preparation on freeze, restore,
reset, growth, ownership changes or failure. A prepared window is an optional
execution plan, never another source of history or a published snapshot.

Measure activation plus continuation together: moving work earlier does not
make its cost disappear. Cap speculative page preparation so short branches
do not pay for a long abandoned future. Re-run corruption after preparation,
partial-page writes, freeze/reuse, sibling order and allocation-failure cases.
Stored hashes describe expected bytes; they alone cannot prove the GPU bytes
still match. Successful old checks must not authorize a later corrupt reuse.

This next window optimization is **not implemented in v0.1.0**. The implemented
win comes from reserved scratch and one ordered verification work list.

## 3. Close correctness gaps before broadening the serving claim

Arbitrary batched partitions and ordinary serving repeatability remain open.
Keep the two-token split counterexample and A/B/A serving controls while
isolating operator/chunk and dispatch differences. Do not relax tolerances.
An unchanged prefix can be reused; after an earlier edit, recompute from the
edit under the same positional/model contract. Rephasing later state with
RoPE/YaRN is outside this milestone.

## 4. Reduce complete API cost, then scale

The durable Python loop still spends substantial time copying, serializing and
publishing the controller, including a large allowed-token policy. Improve that
measured work while preserving canonical bytes, RNG draw state, grammar and
the consumed-token publication boundary. Measure this clock separately from
native decode.

After the bounded workload remains correct and fast, increase prefix lengths
through 12K/32K/64K/128K with matching model capacities and independent oracles.
Compare resident COW, RAM and disk separately. Do not extrapolate eight fresh
prefills and present the extrapolation as eight measured requests.

If GPU sharing stops paying, retain immutable histories with private restoration
as the truthful fallback and label it accordingly; it does not earn G3 physical
sharing credit. G2 crash/restart and G3 pressure/ownership suites still need
qualification before relying on them beyond this bounded experiment.

## 5. Keep Strata integration reviewable

Preserve upstream ancestry and attribution, bring main forward in isolated
integration branches, and rerun the controls after each integration. Propose
small compatible changes to Strata as they become ready. This repository's
experimental release is not an upstream acceptance or production-certification
claim. HIP/SYCL/Windows and multi-request scheduling need their own evidence.
