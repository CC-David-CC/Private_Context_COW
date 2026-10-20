# Matching-batch restart evidence

Latest completed target and lean speed results: [FINAL.md](FINAL.md).
Target 8K/512/128 PASS with 12 zero-error sampled comparisons, three cold
restores with zero prefix computation, and four passing evidence assertions.
Lean 16-output resident/disk/fresh results are in `lean-*.json`; all passed
exact validation, but API generation remains approximately 1 token/second.

Private-only results. Original G1–G5 are not certified.

Tiny: 128 prefix, two different four-token suffixes, eight outputs each.
Twenty full-logit comparisons passed with zero error and zero tolerance.
MR-001 through MR-004 executed: four PASS, zero failures/errors/skips.
G0: 51 PASS. Supplemental recovery: 13 PASS. Pure receipt/source tests: two PASS.
Reference mutation smoke rejected all four deliberately wrong outcomes.

`tiny-summary.json` records timings, fingerprints, actual native work/bytes and
process IDs. `tiny-worker.json` records restored chat-server availability.
`tiny-inventory.json` inventories the full raw archive. The archive is stored as
`tiny-raw.zip.part00` and `tiny-raw.zip.part01`; concatenate their bytes in order
to reconstruct `raw-matching-resume.zip`, then verify `tiny-parts.json` SHA256.
Native state payloads remain retained on llm-60; their size/hash are recorded and
audited. Model weights, native binaries and server credentials are not bundled.

1K passed: 36 full-logit comparisons, zero error, two 64-token suffixes and 16
outputs each; MR-001 through MR-004 passed again, without failures/skips.
`medium-summary.json` and `medium-inventory.json` record the measurement.
The 8K campaign is pending. Peak RAM/VRAM, advanced controller states,
real native crash injection and power-loss durability are not qualified here.
Audit throughput is not serving throughput. Lean performance measurement follows
successful bounded validation, as requested by the user.
