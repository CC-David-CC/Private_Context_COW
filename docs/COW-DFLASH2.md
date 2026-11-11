# DFlash2 with Context COW

DFlash2 is an experimental, explicitly selected draft model for the Flash
Context COW worker. The target still selects every output token. Draft tokens
are temporary: only selected tokens and the corresponding target state enter a
checkpoint. The current implementation is text-only, single-worker CUDA with
fully resident FP16 attention state. It does not implement DFlash2 for image
histories, HIP, SYCL, split-device sessions, or batched requests. Text-only
checkpoints remain restorable when carried by PCC's mixed-input checkpoint
transport; actual image placements are rejected.

Configure a new server using the normal Flash options to tools/cow_server_setup.py
and additionally pass both absolute file paths:

    --dflash2-pack /absolute/path/to/dflash2.pack
    --dflash2-vocab /absolute/path/to/draft_vocab.bin

Omitting both options keeps the normal worker. Supplying only one, an invalid
file, a dense model, or an image producer configuration is an error. Setup hashes
the draft pack and vocabulary into the execution profile; changing either file
requires a new profile. Do not edit the profile or swap files in place. Reserve
additional GPU memory for the draft model and speculative state. Setup does not
download draft assets.

The tested draft pack is Strata's tuned
qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack, SHA-256
4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c.
The tested draft_vocab.bin SHA-256 is
b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91.
The pack is a modified version of Inco AI's Apache-2.0 draft model.
[Its notice and license](../data/dflash2-tuned/NOTICE.md) describe the
provenance. PCC does not distribute these model weights.

The ported drafter and CUDA kernels come from Strata v0.2.13, commit
d2f096475817d4666c364c820b0ca830f751cfc5; Strata's code is MIT-licensed.
PCC uses its existing target verifier, sampler, and COW state handling. A
sampled token is drawn by the existing target sampler from target logits; a
draft is reused only if it matches that selected token. Request receipts name
the verified-dflash2-consumed-v1 operator and describe accepted-token
boundaries, not verifier batch shapes. The dflash2_offered, dflash2_accepted,
and dflash2_windows counters count speculative work; forward_tokens includes
target verification work and consumed_tokens counts accepted progress.

Do not assume this mode is faster. A small test using the UD-IQ3_XXS target
on an NVIDIA RTX PRO 6000 Blackwell measured 106–110 output tokens/s with
DFlash2 versus 126–131 without it, over two 16-token requests. These are
short exploratory measurements, not a throughput qualification or a result
for other contexts, samplers, or hardware. In three 128-token runs with
UD-Q4_K_XL on the same build, median decode rate was 114.8 tokens/s enabled
versus 122.3 disabled, about 6% slower. All three greedy token sequences matched;
the enabled runs reused 249 of 924 offered drafts. These runs used one prompt
and are not a general speed or accuracy qualification.

Checkpoints and execution profiles remain bound to their recorded native build
and execution identity. This mode does not relabel an old checkpoint as a
DFlash2 checkpoint. Cross-profile checkpoint reuse is not qualified.
