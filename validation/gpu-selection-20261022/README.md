# Explicit GPU selection

`cow_server_setup.py --gpu 4` creates a new profile for physical GPU 4.
`--gpu 4,6` selects independent workers behind the existing coordinator and
shared RAM/disk pools. The HTTP launcher reads that profile instead of enforcing
GPU 0. Dense models use explicit per-model llama.cpp device lists with no layer
split. Per-worker loading never changes global CUDA visibility. Startup takes
locks on every selected physical GPU and releases partially acquired locks on
failure. The original GPU-0 setup remains the default.

The new regression initially failed because the device-selection helper did not
exist. After implementation, 88 focused Linux controls and 51 G0 tests passed
with zero failures/errors/skips. The controls include nonzero/reordered physical
IDs, unknown or overlapping devices, unsupported groups, execution-profile
identity, old-library rejection for explicit selection, overlapping launcher
locks and partial-lock cleanup. Existing setup, vision-startup and shared-pool
regressions were included.

Both native adapters and their independent fresh-context oracles compiled against
the pinned llama.cpp revision. A real Qwen3.5-9B Q8_0 smoke on an RTX 4090 compared
three complete frontiers per adapter with exact equality, including two restored
alternate suffixes. Four restores computed zero saved-prefix tokens. The mixed
adapter fixture used text spans; no new image inference claim is made. Both native
loaders rejected an unavailable CUDA device without falling back to CPU. The
configured authenticated HTTP server started with the explicit-binding profile
and shut down cleanly. These results are recorded in `results.json`.

Nonzero physical mapping and independent multiworker topology passed CPU controls;
this run had one physical GPU. Multi-GPU inference, cross-GPU numerical behavior,
the A100 build and Wen architecture admission are not qualified by this test.
Grouped native checkpoint capture still fails before model loading. No cumulative
G1-G5 certification is claimed.

Six temporary-session filesystem/process controls passed, including normal and
exception cleanup, next-start cleanup after a killed process, active-session
leases and preservation of symlink targets/unmarked directories. Wen geometry
admission was added and tested; real Wen inference is tracked separately.

## Wen runtime checks

Wen BF16 at the pinned Hub revision loaded on both the RTX PRO 6000 Blackwell
and A100 GPU 4. The two-A100 campaign used GPUs 3 and 4 with one independent model
per card. Eight full-frontier comparisons across the three native campaigns had
zero error; eight restores computed zero prefix tokens. Sixteen native/Python
tokenizer comparisons passed, including combining marks and literal control text.

Each HTTP campaign completed two concurrent branch requests and preserved the
parent head. Both selected A100 workers handled work in the two-card run. Shared
A100 budgets were exactly 107,374,182,400 bytes each for RAM and disk. BranchWrite
completed a bounded committed chat request on the A100 in single- and two-worker
configurations. All three HTTP campaigns exited cleanly on SIGTERM and left zero
ephemeral session directories. The Blackwell campaign used a 32 GiB RAM budget
and 100 GiB disk budget. Full model hashes matched the Hub pin on both machines.

The first HTTP harness attempts used operation IDs that could begin with a digit,
which the API correctly rejected. The driver now prefixes IDs with a letter.
These were harness failures, not model-training examples. Native checks from
those attempts remain separated from the corrected HTTP runs. Exact phase
results and fixture limits are in `wen-runtime.json`. The driver is included
for reproduction. No long-context stress or broad model competence is claimed.
