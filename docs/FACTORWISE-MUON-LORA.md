Experimental factorwise Muon LoRA
================================

The opt-in `lora_update_space: factorwise-v1` route applies Muon independently
to the A and B factors of hidden attention and feed-forward matrices. Other
factors, including the prediction head, use auxiliary AdamW. This is distinct
from orthogonalizing a full dense weight update and subsequently factorizing it.

Owner optimizer profile schema 3 binds this selection, mode and rank. Schema 1
and schema 2 retain their previous contracts. The update-space marker is part
of the native optimizer checkpoint configuration; a resumed job cannot omit it
and silently reinterpret the saved moments. Schemas 1 through 3 retain their
rank limit of 256. Schema 4 additionally requires
`lora_rank_profile: extended-4096-v1` and explicitly permits ranks through 4096.
The native training and merge entry points enforce the same opt-in bound.

Qualification on llm-60
-----------------------

The twelve tiny-tensor cases passed independent FP64 AdamW and spectral/SVD
Muon comparisons on CPU and CUDA. Cases cover ordinary and factorwise routing,
both singleton matrix orientations, weight decay, Nesterov selection and the
experimental fast/refine schedule. The numerical oracle checks weights,
moments and group roles. Tolerances remain unchanged.

Generated tiny ordinary-Llama LoRA tests passed for AdamW and factorwise Muon
on CPU and CUDA. One update followed by a saved-state resume for two updates
matched an uninterrupted three-update run exactly, for both factor weights
and optimizer moments. Removing the update-space marker from an AdamW resume
was rejected by the native checkpoint configuration check.

CPU and CUDA generated-model Muon adapters also passed standalone F32 merge
checks. Every merged tensor was checked against an independent FP64 base-plus-
BA calculation. Adapter versus merged logits differed by at most 8.95e-8 on
CPU and 5.97e-8 on CUDA, with equal measured losses. Wrong hashes and an existing
output destination were refused. This is tiny-model qualification, not a
measurement of quantized MiniCPM behavior.

Public summaries are in `validation/factorwise-lora-20261027/`. These tests
establish the tested optimizer mechanics and completed-state recovery. They
do not establish MiniCPM learning improvement, high-rank feasibility,
cross-device equality or interrupted mid-step recovery.

New rank-comparison training must await those relevant qualifications and use
the original frozen base and ordered practice. Existing completed comparison
arms and their runtime identities remain separate.

Extended-rank qualification
----------------------------

Generated rank-4096 models passed exact save/resume continuation on CPU, one
CUDA device and two A100 devices. AdamW and factorwise Muon were both checked.
CPU and CUDA Muon adapter merges passed independent FP64 tensor checks and
adapter-versus-merged forward checks. Actual authenticated PCC HTTP jobs at
rank 4096 passed for both optimizers, including owner-profile admission,
idempotent operation replay, rejection of unbound requests and the independent
completion reload gate at its unchanged 0.0002 absolute bound.

The offline parent resolver also passed 26 generated-model checks with a
rank-4096 Muon adapter, including exact state continuation and explicit fresh
moments on retained learned weights. Controlled nonmodel recovery coverage
includes factorwise AdamW, Muon and the extended layout; its 42 checks pass.
These fixtures establish mechanics, not language-model learning quality.

The memory planner and allocator now share the same optimizer-routing predicate.
Persistent-byte estimates were checked against actual generated-model base,
adapter and moment tensor sizes. Muon factors have one moment buffer; auxiliary
AdamW factors have two. Before this correction, factorwise Muon admission
overestimated its moment storage.

Actual MiniCPM persistent admission passed on one 96 GiB GPU for ranks 32, 128
and 2048 with both optimizers. Rank 4096 was refused on that card for both.
Contiguous two-A100 placement passed persistent admission for rank 4096. Actual
execution graphs, completed artifacts and assessment remain separate gates.
Public summaries are in `validation/highrank-lora-20261027/`.

Follow learning results in the separate
[S³ high-rank report](https://github.com/CC-David-CC/sss/blob/feat/minicpm-highrank-20261027/docs/MINICPM-HIGH-RANKS.md).
No improvement claim follows from these admission and recovery tests.
