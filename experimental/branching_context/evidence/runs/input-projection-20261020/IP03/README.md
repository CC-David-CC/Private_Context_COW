# IP03: identical bytes with bulk packing in complete publication audits

Nine new deterministic tests passed; the new work-counter test first observed 16,398 scalar pack calls for an 8K mixed fixture, then zero after bulk packing. Exact native and position bytes, hashes, image grids, offsets, signed limits, changed encodings, subclasses and seeded append cases agree with the unchanged scalar reference. The full focused structural/host-protocol suite passed 114/114, zero failures/errors/skips. G0 passed 51/51; all four reference mutants were killed. No existing tests, reference equation, scalar cursor check or native executable was edited.

Only _InputScheduleProjection full-audit byte construction uses these helpers in production. The fresh_input_binding function is tested for potential later use but is not substituted for the cursor's independent reference call. The per-token validation call-count assertions remain unchanged and pass. All complete audit receipt fields and hashes are still checked.

benchmark-fresh.json is an isolated pure binding study, not a deployed hot-loop gain. benchmark-workflow.json is the controlled complete 64-step CPU workflow: baseline/packed then reversed order, independent warmups, shared empty-cache rules within each arm and identical final history/schedule digests. Initial unbalanced-cache measurements are retained and superseded, not used for conclusions. Report individual samples. No GPU TPS conclusion is possible from this experiment.

The first new test run also had a fixture exception-handler error (InvalidOperation is not ValueError), corrected in the new test with the initial log retained; no existing acceptance assertion changed. Full scalar/native reference files are unchanged.

Source/source closure changed, so the old prepared IP02 service profile must not be started as-is. Native GPU qualification and interactive pause/steer campaign remain outstanding. Ordinary chat stays off; no live service process or model controller was changed by IP03.

Measured full 64-step CPU workflow changes: 8K -2.72%, 64K -1.36%, 128K +6.25% (slower), despite full-audit finishing being 19-22% faster at all sizes. The 128K whole-workflow regression is retained and unresolved; the unchanged cursor dominates the total. No whole-system speed claim or automatic live promotion follows from the faster audit. This remains a private candidate pending actual GPU and larger-context measurements.
