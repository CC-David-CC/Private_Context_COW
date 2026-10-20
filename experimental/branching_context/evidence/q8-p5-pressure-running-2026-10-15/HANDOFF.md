# Q8 P5 pressure run: running checkpoint

This is an in-progress execution receipt, not a qualification result. The first
attempt (`p5-q8-full96-run-20261015-v1`) exited 1 before native/model startup
because it passed an already-normalized fixture to a helper that requires the
raw `{contexts: [...]}` input. Its failure logs and numeric exit receipts remain
on `llm-60`; no inference ran in that attempt.

The corrected run (`p5-q8-full96-run-20261015-v2`) used the raw Q8 fixture.
CPU-only normalization equality was checked against the previously reviewed
normalized fixture before relaunch. The check and exact hashes are recorded in
`fixture-parity.json`. Frozen source was
`/home/dflanag3/branching-g3/p5-q4-root24b4-source-20261015` at authorized
commit `24b4fbbf85deec2b3c4beb8cc8af389c730ea32c`; no source changes were made.

At this checkpoint, v2 is still running on GPU0 in tmux session
`p5-q8-full96-v2`: wrapper/campaign PID 1927700/1927704, resume worker/native
PID 1933622/1933628, and telemetry sampler PID 1928450. The campaign profile is
bound profile ID `9572549eb5adab9d5573fbb632af2212657798914ad3ad2d453e704ad58e9e83`.
See `RUNNING-RECEIPT.json` and `RUNNING-UPDATE.json` for exact hashes, argv,
phase, counters, and contemporaneous telemetry. Do not interpret the
`numerical_max_error: 0.0` field while the result remains `RUNNING`.

The prepare phase has completed nine requests. Its read-only same-process
spill/reload trace validator matched three exact clean-eviction → disk-load
chains; details and hashes are in `prepare-spill-reload-witness.json`. This is
only a prepare-phase witness, not terminal or cumulative campaign evidence.

The compact monitor helper on the host is
`/home/dflanag3/spatial-artifacts/p5-q8-full96-run-20261015-v2/monitor-summary.py`
with SHA-256 `8fdd72ad7d0eb75f2e933a85a0e23ec9f2c230cb3583009e0bd169f492a338c6`.
Run it with:

```sh
python3 /home/dflanag3/spatial-artifacts/p5-q8-full96-run-20261015-v2/monitor-summary.py \
  /home/dflanag3/spatial-artifacts/p5-q8-full96-run-20261015-v2
```

It prints only phase/request counts, actual output counts and selected counters,
disk free space, and GPU status; it avoids dumping embedded histories/frontiers.
The initial sampler launch failure is preserved: the first wrapper referenced a
missing v2 copy of `telemetry/sample.py`; a sidecar sampler was started without
restarting or modifying the campaign. The incident is documented in the
running receipt and `sampler-launch-failure.txt` on the host.

At the latest checkpoint the resume phase was active. Continue monitoring the
same tmux session and PIDs; do not start a duplicate campaign. All numerical,
controller, eviction, resource, and terminal checks remain pending until the
campaign and fresh-reference comparisons finish.
