# Bounded GPU observation

`python -m tools.pcc_gpu_observer --output /private/new-observer --interval-seconds 30 --duration-seconds 216000`

This optional owner-run command records every NVIDIA GPU's UUID, free/used
memory and device-wide compute/memory activity. It makes read-only driver queries
and does not allocate VRAM, control jobs, reserve a GPU or select training work.
The maximum observation duration is 72 hours. Existing output is never replaced.

`identity.json` records the observer process and source identity;
`samples-private.jsonl` retains samples and query failures; `status.json` is the
latest observation or terminal summary. Missing telemetry is unavailable, not
zero activity. A stale running status alone is not proof the process is live:
check the recorded PID and process start ticks. SIGTERM requests a clean stop.

Every actual training/evaluation launch still needs fresh resource admission and
PCC leases, per-device budgets and memory reserves. A device's utilization is
shared across all users, not a measurement of our job's compute share. Total free
memory across devices is not one contiguous allocation.

For campaign accounting, one allocation-hour on each of three GPUs is three
GPU-hours. Count the full held interval conservatively during shared contention;
record training, evaluation and failed allocations. Waiting without an allocation
uses zero GPU-hours. Telemetry alone cannot establish those held intervals:
original job launches and resource settlement must supply the cost evidence.
