# Image COW: available features and executed coverage

This is a Q4 Linux/CUDA prototype for one request at a time on GPU 0. A row
marked as tested means that its named examples ran. It does not promise every
possible combination, a Q8 result or a full G1-G5 certification. Native batch
boundaries and the complete execution profile are part of exact continuation.

| Feature | Implementation and executed coverage | Remaining boundary |
|---|---|---|
| Immutable text/image history | Canonical ordered spans bind original and embedding identities, grids and positions. Same pad IDs with different pixels stay distinct. | The full image-format and authorization interaction matrix is not qualified. |
| Image embedding cache | Shared RAM/disk budgets, LRU, idle expiry and active leases; retained originals allow verified re-encoding. | Sustained whole-host pressure is broader than bounded pool/loader tests. |
| GPU COW branches | Image-prefix A/B isolation, text/image switching, different grids and partial-page/batch crossings have bounded evidence. | Arbitrary contexts/layouts, multiple GPU workers and split-GPU images remain unqualified. |
| Disk restart | Strict restore performs zero saved-prefix forward work, including absent embeddings and unavailable encoder. Explicit reconstruction after eviction is separately reported. | Broader controller/crash combinations and power-loss durability remain unqualified. |
| CLI and authenticated HTTP | Upload, context, ask, append-user, continue, fork, retry and restart have actual client-to-GPU examples. | BranchWrite's own mixed-image `load_context` bridge is not implemented. The direct COW CLI supports ordered parts. |
| Greedy and named GBNF | Image branches, parser progress, fork and restart have bounded tests. | The complete pending-decoder, text-stop and reasoning-restart matrix remains open. Image text stops are rejected. |
| Temperature, Min-P, Top-K, Top-P | Per-generation controls, nonzero RNG progress, seed/reseed and changed-settings retries have bounded CLI evidence. | No cross-quantization or cross-batch equality promise. Legacy typed9/11 changed-controls retries remain unsupported. |
| DRY | Counts text with image barriers; actual penalties and CLI/restart exercised. | Requires checked native mixed-text command, not an ordinary token-tail sampler. |
| Typical and DynaTemp | Individually and together; also DRY+Typical, DRY+DynaTemp and all three on checked mixed-text V4. | DRY+XTC/Mirostat/Adaptive remains guarded. |
| XTC | Alone or with Typical/DynaTemp; exact RNG draw cursor checked. | DRY, Mirostat and Adaptive combinations rejected. |
| Mirostat 2 | Branch-owned feedback, sampling and restart examples. | Requires its V6 feedback capability; DRY combinations rejected. |
| Adaptive-P | Feedback, probability/entropy filters, settings changes and restart examples. | Requires its V7 feedback capability; Mirostat, active XTC and DRY combinations rejected. |
| Thinking with image turns | Corrected92df first admission/retry, new/repeated images, warm/cold continuation:14 exact full vectors and one full controller comparison. | Final `OK` constrained, thought quality not scored. Wider reasoning/controller combinations remain open. |
| PNG/JPEG and EXIF | PNG/JPEG admission; CPU normalization checks all eight orientations with independent pixel permutations. | Earlier92 tiny RGB PNG/JPEG GPU continuation now passes all eight EXIF orientations and swapped two-image order. Alpha/non-RGB/larger-image coverage remains open. |
| Fault handling | Bounded disk-full, CUDA allocation failure, encoder-loader ENOMEM, cancellation, disconnect and11 process-crash cases have evidence. | Not power-loss testing, sustained physical-memory pressure or every failure site. |
| Small-context speed | Earlier61 profile: about92 streamed TPS,3.72s whole CLI,1393-cell prefix and186 generated tokens. Hashes remain enabled. | This is not an8K or Q8 speed claim. Corrected92df8K timing now passes36 requests: resident47.81 streamTPS/6.953s whole CLI; RAM47.96TPS and disk47.65TPS. |

## Current profile boundary

The currently used execution profile is092f (full schedule validation projection).
Its tiny real-Q4 campaign checked102 full vectors with zero error and strict
process restarts. The92df8K speed/numerical results below belong to that earlier
profile;they are not automatically current092f long-context qualification.
BranchWrite H08 supplies a separate actual client/image-delivery example on092f:
local captured resources,two model pictures,600 native rows and unchanged pixel/
control checks. It did not run a new full-logit or inference-restart oracle.

## Evidence by profile

- [Current source-bound tiny validation](../../experimental/branching_context/evidence/VISION-SCHEDULE-PROJECTION-2026-10-18.json):092f,102 exact full vectors and strict process-separated restores;medium/target remain unqualified on this profile.
- [Autonomous asset client](../../experimental/branching_context/evidence/VISION-BRANCHWRITE-H08-2026-10-18.json):H08 real CLI/browser/native delivery,265 selected CPU checks;no new numerical oracle.
- [Fast schedule validation](../../experimental/branching_context/evidence/VISION-OWNED-SCHEDULE-SPEED-2026-10-17.json): profile61d56af, small-context measured speed; original timing runner failure and focused reference recovery are retained explicitly.
- [Medium numerical/restart](../../experimental/branching_context/evidence/VISION-OWNED-SCHEDULE-MEDIUM-2026-10-17.json): same61 profile,190 full vectors, every generated frontier.
- [Target numerical/restart](../../experimental/branching_context/evidence/VISION-OWNED-SCHEDULE-TARGET-2026-10-17.json): same61 profile,8192 text+1073 image=9265 cells, two512-token suffixes,128 outputs each;71 sampled vectors. Sampling is prefix/suffix and generated1/16/64/final.
- [Complete8K timing](../../experimental/branching_context/evidence/VISION-COMPLETE-8K-LATENCY-2026-10-17.json): earlier92df,36 actual requests,4 full prefix/final vectors exactly equal;8561-cell image prefix,25-cell question,115 generated tokens; warm/uncontrolled filesystem cache. No continuously measured VRAM peak.
- [92df progressive restart](../../experimental/branching_context/evidence/VISION-FINAL-ROUTE-TARGET-2026-10-18.json): corrected92 tiny/medium/8K now independently pass;363 vectors and21 controllers, with every-generated small cases and sampled target.
- [92df orientation and order](../../experimental/branching_context/evidence/VISION-FORMAT-MATRIX-Q4-2026-10-18.json): tiny RGB PNG/JPEG EXIF1..8,18 actual CLI/GPU cases,36 exact full vectors,18 parent byte audits; one512-thinking budget closure.
- [Corrected image-turn flow](../../experimental/branching_context/evidence/THINKING-NEW-IMAGE-FLOW-Q4-2026-10-17.json): profile92df,14 exact vectors, actual encoder/cache work, repeated image placement and producer-free disk continuation. Old8e failed duplicate-system admission and remains failed.
- [DRY plus entropy filters](../../experimental/branching_context/evidence/VISION-DRY-ENTROPY-2026-10-17.json): four bounded CLI campaigns and a separate real-penalty selector probe. Saved request boundaries are checked; native server exit codes were not individually captured in those older campaigns, although PIDs were absent.
- [VC ledger](../../experimental/branching_context/evidence/VC-CURRENT-QUALIFICATION-LEDGER-2026-10-17.json): remaining work and the other sampler, pool, failure and client reports.

Exact source/profile IDs stay separate. A passing earlier profile supplies
supporting evidence; it does not automatically qualify a later profile. Q8 is
currently deferred by the owner. General web health still advertises images
off; no web image interface is supplied here. Use the explicit COW CLI/server
configuration in the [guide](vision-cli.md).
