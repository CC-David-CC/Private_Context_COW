# Private batched speed recovery — 2026-10-14

## Latest measured state

On llm-60, the independent native Q4 Prefill loop produced 45.06–45.10 TPS at
1K/64/16. Native COW with a capture after every token produced 10.38–10.57 TPS.
The original Python snapshot loop produced about 1.09–1.10 TPS. Sharing token
selection/consumption ownership and caching hash-verified ancestry parsing raised
it to 2.40–2.41 TPS; caching parsed snapshots raised it to 2.79–2.86 TPS.
These are individual probes, not three-repetition medians or long-context Q8
qualification. Hashes remain enabled. Raw layer and attribution JSON files are
retained beside this document.

The stage2 worker passed its four real-GPU ownership/publication/interleaving
regressions, tiny and medium matching-schedule campaigns, and attribution probe.
It restored the unchanged chat server and received HTTP 200. Windows cannot run
the store tests: its import requires POSIX fcntl. This limitation remains open;
the corresponding implementation tests execute on Linux.

The next milestone is native generation with one active request and save points
at turns/explicit saves. It has not yet been implemented or qualified. Existing
batches-only restart evidence does not qualify normal native generation.

Implementation branch: `feat/batched-resume-speed`, based on `a091db4c`.
Local checkout: `C:\Users\dflanag3\Documents\fleet\context-cow-execution-parity`.
Remote campaign: `/home/dflanag3/branching-g3/speed-recovery-20261014-v1` on llm-60.
Only the private box-git repository is authorized for this campaign.

## Executed preparation

- Linux G0: 51/51 PASS, no skips; receipt `20261014T060111794563Z-G0.json` in the retained Linux API.
- Windows G0: FAIL at import (`fcntl` unavailable); nine reference tests passed and two modules failed to import. This is the existing POSIX store limitation, not a GPU result.
- Local model inventory tests: 4/4 PASS. Supplemental nested wall-time accounting tests: 2/2 PASS. Neither qualifies GPU inference.
- Q4 inventory: 112,830,673,812 bytes actually hashed. Q3 inventory: 77,388,173,410 bytes actually hashed. Receipts are included here. No quantization inherits another model's numerical qualification.
- Q4 128-prefix / 4-suffix / 8-output matching-schedule restart campaign: PASS. The attached result records executed subprocess phases. Numerical comparisons and detailed counters remain in the remote campaign until collection.
- Q4 1K / 64 / 16 campaign started; outcome not yet collected.

## Code observations, not timing conclusions

`stream()` currently materializes once for selection and again for consumption.
`_save()` captures an immutable native COW image and then parks a separate full host snapshot. Strict matching-batch materialization does not use the host-cache fallback. Their costs are being measured before changing behavior.

The diagnostic runner wraps the unchanged lean API. It records inclusive and exclusive nested wall-clock costs, including native waiting and synchronization. These are **not GPU-event kernel timings**. Setup and final validation are included in its aggregate attribution; lean request timings remain separately reported.

The queued follow-up waits for the identified Q4 campaign to finish successfully and for serving to be restored. It then runs the resident timing probe and the new real-GPU stream regression tests. The baseline must fail exactly the duplicate-restore assertion; any additional failure is retained as an unexpected failure.

The preparation notes above describe the initial baseline queue. The latest
measured state supersedes their pending status. No new G1–G5 qualification is
claimed. Normal decode was outside that campaign and is now authorized for the
separate native-speed request milestone.
