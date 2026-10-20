# Batches-only durable resume

Superseding scope: the user subsequently accepted an execution-profile-bound
resume contract. See `docs/context-cow/PROFILE_BOUND_RESUME.md`. Exact continuation
is required for matching token history, execution profile and batch/request
schedule. Cross-schedule equivalence is excluded from that restricted milestone.
The earlier scope below and all measured failures remain historical evidence;
the original registered certification contract is not changed.

User scope: exclude normal verifier decode from this milestone. Use the native
batched Prefill operator for the prefix, changed suffix and consumed output
tokens, and compare with independent fresh full-history batched inference.
Autoregressive selection still requires one consumed output before selecting the
next. Such a one-row Prefill call is not the separate token-step debug operator.

Normal decode is a separate research track. Further investigation and fixes on
that track have not started. Earlier exploratory measurements did execute normal
decode; those facts are retained, not relabelled as unexecuted work. In particular,
the earlier first-output max error of 1.9153637886047363 remains an unresolved
cross-operator discrepancy, outside the batches-only milestone's acceptance.

This scope does not relax fresh-history equivalence for the selected batched
operator. A mismatch between restored batched continuation and fresh full batched
history still fails. Native counters must show no normal decode consumption,
exactly the requested suffix/output work and zero saved-prefix work on restore.
The new operator requires its own build identity and independent baseline; do
not reuse the earlier normal-decode execution profile or numerical admission.

All development and evidence stay on private_context_cow.git on box-git. No
GitHub or Strata upstream publication is authorized at this stage.
