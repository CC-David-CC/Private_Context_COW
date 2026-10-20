# Q4 COW: complete 8K-source timing campaign

On llm-60 GPU0, corrected profile92df completed36 actual CLI requests and both
independent raw native references. Four full248320-value vectors match exactly,
error0. One warm-up plus three measured repetitions per path. These are ordinary
consumed/durable client events, not a numerical audit after every output token.

The image prefix contains8261 text/control cells plus300 image rows=8561 cells.
The text control contains8259 cells. Both use the same25-cell question; the
image answer has115 generated thinking/final/EOT tokens and text has141.
The requested source-code slice was8192 tokens; that is not the total prompt.
Final `OK` is constrained, so this does not score thought or image-answer quality.

| Path | Whole CLI median | Streaming TPS | First token | Mean token interval |
|---|---:|---:|---:|---:|
| resident | 6.953s | 47.81 | 382.3ms | 20.92ms |
| ram | 7.356s | 47.96 | 789.5ms | 20.85ms |
| disk | 7.966s | 47.65 | 1355.7ms | 20.99ms |
| text-resident | 8.049s | 48.45 | 352.1ms | 20.64ms |
| image-after-text | 6.975s | 48.27 | 391.0ms | 20.72ms |

Fresh encoding+prefill:8.087s. Reused embedding+fresh prefill:7.013s. Text-only
prefill:6.716s. Native resident reuse evaluates only25 question+115 output
cells, zero saved-prefix forwards/encoder calls. RAM and disk do the same work.
RAM source loads341722060 bytes through the native file loader from tmpfs;
disk additionally reports342715340 pool bytes read. These are logical transfers
with warm/uncontrolled filesystem cache, not physical NVMe reads.

Resident last-token-to-durable median is4.079s. This includes final validation
and durable publication; the campaign does not attribute each part separately.
Streaming48TPS and whole-request16.54TPS describe different intervals. The
client observer adds0.327% versus plain CLI; this does not measure all validation
overhead. Native/request/capture durations overlap and must not be summed.
Native counter totals104988 forwards reconcile to every measured job. GPU
and fresh reference processes exited0; wrappers0/0. Initial independent auditor
failed because it tried to compute decode TPS for zero-output preparation;
its corrected version passes without rerunning or changing the GPU experiment.

The model/build, CPU/GPU placement, FP16 KV and98K-capacity profile are pinned.
All32 CPUs were available; model expert RAM budget40GiB and vision headroom8GiB.
Shared pools are96GiB RAM/2GiB disk; disk size is a host-space constraint.
Process VmHWM is recorded in JSON; this includes mapped/file-backed memory.
GPU memory was sampled, not continuously peak-instrumented. Hashes remain on.

Earlier61-profile small-context92TPS is a different context/source profile;
this is not a controlled source regression comparison. Corrected92df is now
measured directly at the larger context. Full supported format/controller matrix,
Q8, split-GPU and original G1-G5 qualification remain separate and incomplete.
See [the machine-readable report](VISION-COMPLETE-8K-LATENCY-2026-10-17.json).

Private archive: `box-git:/srv/data/git/private_context_cow.evidence/20261017-vision-final-route-latency-8k-v1/evidence.tar.gz`
SHA-256 `9f68c7f65259f9bacbac0b2382b55981101b7880e4759222e32104220f7f4211`; all606 data
files individually receiver-verified. Optional1363026536B numerical
state retired afterward; canonical/originals/embeddings remain. Full frontier logs were later retired
locally after archive+manifest re-verification; they remain recoverable from
the private archive. See VISION-LATENCY-LOG-RETIREMENT-2026-10-18.json.
