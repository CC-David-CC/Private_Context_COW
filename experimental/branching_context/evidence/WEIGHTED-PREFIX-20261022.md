# Weighted prefix admission evidence, 2026-10-22

CPU controls use the real ContextCoordinator, BranchingModel, shared RAM/disk
pools, immutable store and a deterministic CPU backend. They do not simulate or
claim GPU speed, native numerical qualification or model task competence.

Linux Python 3.14.4 results:

* `PYTHONPATH=src:tests/implementation python -m unittest
  test_weighted_prefix_scheduler test_root_computation_reuse
  test_pool_coordinator test_missing_source_rebuild`: 45 passed.
* Additional weighted adversarial cases: `test_weighted_prefix_scheduler`,
  final 8 passed. Failed leader releases follower; queued cancellation does not
  steal the active claim; a missing prefix is rebuilt once and its derived
  checkpoint shared while unrelated work progresses on another worker.
* `python scripts/verify.py --gate G0`: 51 passed, zero required real-backend
  cases. No G1-G5 claim follows from G0.
* With repository root, tests and implementation tests on PYTHONPATH:
  `python -m unittest test_cow_server_setup tools.test_cow_server_vision
  tools.test_wen_training_import`: 12 passed, including unchanged default launcher
  arguments and explicit-only scheduler forwarding.

These runs overlap. No skips count as passes.

Initial default-policy fault injection failed:
`test_unread_failed_request_does_not_retain_rebuilt_checkpoint_traceback`
expected a failed request but saw durable. A new scheduling timing read had
shifted the injected post-rebuild counter read. The timing read now occurs only
under the opt-in policy; the existing assertion was preserved and the 45-test
regression passed. Initial missing test modules were development-archive/import
setup mistakes, corrected without changing product assertions.

The queue policy test initially failed because its new module was absent before
implementation. Default FIFO placement and admission are retained. Native source
code is unchanged by this queue feature. Real GPU scheduling, concurrent Wen
exploration, peak training VRAM and reconstructed PyTorch/native numerical
comparisons remain separate qualification steps.
