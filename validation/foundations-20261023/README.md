# Native foundations CPU checkpoint — 2026-10-23

This checkpoint changes optional machinery. It does not update Wen, install a
candidate, or qualify the changed native source on CUDA, HIP or SYCL.

Python checks passed: 16 lesson/segment tests, 14 sequence-admission/native tests
and 13 compatibility-frontend/setup tests. Actual BW/PCC CLI integration passed
single, batch, partial-batch, duplicate prevention, continuation and reload, plus
schema-2 multi-turn lessons with both target policies, on a generated CPU fixture.

The final CPU binary passed ordinary chain tests, the efficient sequence chain,
masked-context gradient checks and concat backward on all four axes. An independent
FP64 reference checked 386 scalar derivatives across the tiny hybrid model's two
layers: maximum absolute errors were 1.72831e-8 and 5.88961e-8.

| Sequence | Mode | Before loss | Updated/reloaded loss | Largest reported tensor buffer |
| --- | --- | --- | --- | --- |
| 16,384 | LoRA | 5.5921854973 | 5.5921840668 | 5,411,209,152 bytes |
| 16,384 | Full | 5.5921854973 | 5.5841674805 | 5,415,993,280 bytes |
| 32,768 | LoRA | 5.5911083221 | 5.5911068916 | 21,559,803,840 bytes |
| 32,768 | Full | 5.5911083221 | 5.5831222534 | 21,569,372,096 bytes |

Each long fixture supervised 32 selected target positions. All gradients were
finite; 16 of 32 LoRA tensors and all 28 full-model tensors had nonzero gradients.
The tensor-buffer figure is not total process memory or measured peak RSS.
Original fixture weights remained unchanged. See [long-proof.json](long-proof.json)
for source/base hashes and receipt hashes.

An earlier 16K attempt failed before producing a candidate: the generic scheduler
tried to reserve roughly 646 GB for possible graph splits. The long CPU path now
uses a direct single-device graph allocator with revised memory admission. The
failed attempt remains in private engineering evidence.

The normal short path retains its existing sequence assembly and scheduler;
ordinary small-chain and independent derivative checks passed. The opt-in
`tiny-cpu-long-v1` path is CPU-only, admits bounded tiny geometry, caps sequence
length at 32,768 and selected targets at 256, and keeps prior input differentiable.
No PyTorch or section-adjoint mechanism was used for these ordinary-backprop proofs.

Backend builds/live qualification, real Wen token/template alignment, long Wen
inference/training, all-token 32K supervision and learning quality remain pending.
Follow [the integration guide](../../docs/SSS-FOUNDATIONS.md) and its S³ handoff.
