# Root presentation query CPU handoff

The targeted root-mode query now checks the hash-valid descriptor kind before traversing its source snapshot. A rewind receipt can target the root while referring to a large future snapshot; the query skips that source, while the full presentation catalog still validates it. Selected root annotations continue through the existing full validation path.

The new rewind-to-root regression failed against the prior implementation, then passed with the fix. The focused suite passed 11/11, the serving/Store suite passed 84/84 (83 prior tests plus the new case), and formal G0 passed 51/51. All command and wrapper receipts are zero. Full stage paths, hashes, preserved setup failures, test IDs, and scope limits are in [ROOT-PRESENTATION-QUERY-2026-10-16.json](ROOT-PRESENTATION-QUERY-2026-10-16.json).

Changed source is frozen at the hashes in the JSON. No commit was made; the parent owns review and integration. This is CPU-only qualification and does not include GPU or performance claims.
