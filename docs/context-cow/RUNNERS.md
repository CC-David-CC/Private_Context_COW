# Included runners

All paths in the first column are relative to
[`experimental/branching_context/scripts`](../../experimental/branching_context/scripts).
The 8K runners and their setup helper are the supported starting point for
this experimental package. Older diagnostics retain their original scope and
may require the historical pinned execution profiles they describe.

| Runner | What it exercises |
|---|---|
| `build_fast_prefill_probe.py` | Build separate native batched COW and uncached full-history probes from the current CMake objects |
| `build_external_integrity_probe.py` | Opt-in caller-managed write integrity, with optional reuse of already writable private pages; original guard stays available |
| `prepare_speed_profile.py` | Verify actual model inputs, binaries, hardware and fixture; write an explicitly unqualified local profile |
| `cow_speed_8k.py` | 8K + 512 + 128; resident A/B, RAM A, disk B; logits, page counters, parent audit and durable publication |
| `ram_speed_8k.py` | Same checks for RAM-restored private GPU pages, A/B/A |
| `speed_8k.py` | Uncached ordinary serving A/B/A; automatic/fixed dispatch and CPU launch controls |
| `fast_prefill_probe.py`, `measure_prefill_paths.py` | Small batched/stepwise and split-boundary diagnostics |
| `long_prefill_probe.py` | Larger-prefix continuation experiment; retained 32K results are partial |
| `serial_speed.py` | Earlier stock/modified/branching serial-request comparison |
| `state_machine.py`, `state_machine_campaign.py` | Bounded operation sequences and seeded state-machine campaigns |
| `verify.py`, `mutation_smoke.py`, `real_mutation_smoke.py` | Gate registry and reference/real-backend mutation checks |
| `mutation_campaign.py`, `build_page_mutant.py`, `run_m05_gpu.py` | Explicit test-only negative fixtures, including shared GPU-page corruption |
| `run_compute_sanitizer.py`, `sanitizer_branching_probe.py`, `sanitizer_suite.py` | Native CUDA diagnostics and registered sanitizer workflow |
| `lifecycle_evidence.py`, `lifecycle_soak.py`, `allocation_probe.py` | Resource lifecycle, long soak and real allocation observations |
| `crash_worker.py`, `oom_worker.py`, `recovery_worker.py`, `recovery_observation.py` | Process-crash, memory-failure and recovery fixtures |
| `build_native.py`, `build_strata_profile.py`, `profile_g1.py`, `profile_g2.py` | Pinned native/profile preparation for registered gates |
| `calibrate_real.py`, `calibrate_strata_operator.py`, `calibration_inputs.py` | Numerical baseline collection; not a license to widen a failed tolerance |
| `admit_strata_profile.py`, `seal_execution.py`, `check_strata_api.py`, `analyze_strata_logits.py` | Profile admission, exact execution binding and diagnostic analysis |
| `run_ticket.py` | Existing completion-ticket harness |

Native COW tests live in [`tests/core`](../../tests/core). The new
`branch_prepare_test.cpp` checks one all-pool launch, unchanged hash coverage,
private tail reuse and rejection before any copy on late-pool corruption.
`branch_digest_pages_test.py` checks reordered page lists against Python hashlib.
They use actual CUDA allocations, not a mocked GPU.

[`tools/package_context_cow_results.py`](../../tools/package_context_cow_results.py)
exports passing, failing and partial reports with explicit path redaction and a
content-hash manifest. [`tools/plot_context_cow.py`](../../tools/plot_context_cow.py)
regenerates all four figures from those measurements.

The native branch driver remains isolated from default production serving.
The package is one-request-at-a-time; serving integration, arbitrary batching
and the complete registered certification matrix are still open.
