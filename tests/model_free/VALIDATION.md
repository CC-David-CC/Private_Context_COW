# MF-01 fixture validation handoff

Branch: `work/model-free-fixture-luna`
Base: `dd2f996d7faac9df91eab9cec14c342d88f98522`
Source commit: `7c27764cc07e5e4e9b456e1fd4be8e24262536be`
Global-offset correction commit: `ed335270da0e7bd4f20af3ede33d48e7bfee6576`
Remote execution path: `/home/dflanag3/branching-g3/model-free-luna-fixture-20261014/tests/model_free`
Host alias: `llm-60`; interpreter: Python 3.12.3; compiler: g++ 13.3.0.

## Checks

- Initial test-first red: before `fixture_reference.py` existed, `python -m unittest discover -s tests/model_free -v` failed to import `fixture_reference` (`ModuleNotFoundError`). This output was observed on Windows before execution policy steering; it is historical and not counted as qualification.
- Current Windows local test/CPU execution: UNRUN for qualification, per task steering.
- Remote `python3 -m unittest discover -s tests/model_free -v`: 6 passed, 0 failed, 0 skipped. IDs: `test_seeded_known_answers_and_coordinate_sensitivity`, `test_logical_page_markers_and_physical_independence`, `test_offset_and_byte_order_are_observable`, `test_canonical_state_excludes_device_canary`, `test_canonical_known_hash_and_stream_order`, `test_float_values_are_finite_bounded_and_dyadic`.
- Remote command `g++ -std=c++17 -O2 tests/model_free/fixture_cpu.cpp -o /tmp/mf01-fixture-cpu` succeeded. Streaming seed `0x5eed`, dimensions `1 1 1 8192` through `sha256sum` produced `2eb34e7ed0da08b0012daa8bdb83c2555bf6e4d43d8c9c9b08a8e1c76287a7f3`, matching the Python oracle and frozen spec.
- Header-only C++ compilation/runtime check: UNRUN. GPU/CUDA/HIP execution: UNRUN. 175 GiB run: UNRUN.

Known byte answers at seed `0x0123456789abcdef`, coordinates `(region=2, layer=3, token=17, channel=5)` are offset 64=`8d`, 65=`db`, 4095=`46`; logical page 1 marker at 4096 is `43 46 58 31 01 00 00 00 00 00 00 00 a5 5a 3c c3`.

This fixture verifies deterministic data placement and movement only. It does not verify a model, inference result, GPU COW behavior, or physical page granularity.

## Global-offset correction

Review found that the original canonical Python and CPU streams restarted the
byte offset for each cell. Two focused regressions were added before the fix.
On `llm-60`, both failed as expected: the short-cell seed-sensitivity test had
three failing subcases (cell widths 1, 2, and 8), and the global-page test
failed because the 4096-byte marker landed at cell boundaries instead of the
flattened stream boundary. An initial targeted unittest invocation from the
repository root failed to import the test module; the tests were rerun from
`tests/model_free` to obtain the recorded behavioral red.

Both `iter_canonical` and the independently implemented C++ CPU streamer now
carry one monotonically increasing global byte offset across flattened
layer/token/channel/cell order. The tests compare the complete short-cell
payload against an explicit coordinate/offset traversal, assert a swapped-cell
mutation changes bytes, and verify marker vectors at global offsets 0, 4096,
and 8192 while checking cell starts do not restart a marker.

Remote Linux green: `python3 -m unittest discover -s tests/model_free -v` — 8
passed, 0 failed, 0 skipped. Remote `g++ -std=c++17 -O2` build of
`fixture_cpu.cpp` succeeded. CPU streamer and Python streaming SHA-256 values
agree for three cases: seed `0x5eed`, dimensions `1/1/1/8192`:
`2eb34e7ed0da08b0012daa8bdb83c2555bf6e4d43d8c9c9b08a8e1c76287a7f3`; seed
41, dimensions `2/3/4/8`: `75fac4455a1382a7765314fa907e4f3cecfdfbf804d02e3e40bc14acb025d894`;
seed 73, dimensions `2/2/300/7`: `988403aa81f7e674f75d0252fc38c4aba8c66095e5d82810a4b48d64e2c1f324`.
The one-cell 8192-byte hash remains unchanged. Direct `byte_at` vectors are
unchanged. Local execution remains unrun for qualification; GPU/CUDA/HIP and
175 GiB runs remain unrun.
