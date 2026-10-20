# Post-question frontier audit: bounded Q8 PASS

The independent fresh references retained two complete 248,320-float vectors:
after the 65,536-token prefix and after each 8,335-token question. The original
long campaign compared 13 selected frontiers but omitted the latter boundary.
This separate audit restores P and compares both post-question vectors exactly.
Its two comparisons do not alter the original campaign's 13-comparison receipt.

Both A and B passed byte-for-byte, hence maximum finite logit error zero.
Each restore performed zero forward/prefill work. Each question evaluated
exactly 8,335 tokens through native Prefill in chunks [65536,8192] and
[73728,143], with no decode calls. Final consumed count was 73,871 (last
zero-based position 73,870). No generation was performed in this audit.

The 1,830,602,232-byte saved-parent payload matched prepared SHA-256
0752b4ac6abc7411cc89978bf4f7fbc38625a7886f149dcff2e69b3ad50cea15
before and after. All 32 canonical metadata files and the directory set were
unchanged. The separate preflight's fingerprint has 33 dictionary entries
because it includes the directory-list entry, not 33 files.

Native PID 2310820 exited zero; command and wrapper exited zero. Five observer
samples were recorded; this is not a continuous peak-memory qualification.
Runtime used pinned Q8 profile 3e3bf375e730d87b220a1516f5d8480fe18a46ccab93e272969aa530e6398529
and the same CUDA binaries as the long campaign. Full provenance, work counters,
reference frame hashes and native receipt rows are in
[the executed result](Q8-SUFFIX-FRONTIER-RESULT-2026-10-15.json), raw SHA-256
64d40798252b7ba4d49f5fe157395871160c60ede7ee4afda04c928d5c12ac61.

The auditor's nine focused CPU tests passed on llm-60 with zero skips and
command/wrapper zero at
`/home/dflanag3/spatial-artifacts/suffix-frontier-cpu-v9-run/`.
They cover valid retained files, changed schedules/tokens/root receipts,
truncated/nonfinite frontiers, wrong payload hashes and nested metadata.
An earlier Windows debug run is not qualification evidence; final execution
was Linux. Preserved earlier RED/setup failures remain in the remote run tree.

GPU receipts are under
`/home/dflanag3/spatial-artifacts/suffix-frontier-audit-q8-20261015/runs/audit-v1/`;
the raw output and observer remain alongside them. No canonical branch was
moved and no checkpoint published by this auditor. Original cumulative gates,
arbitrary batch-boundary equivalence and untested controllers remain separate.
