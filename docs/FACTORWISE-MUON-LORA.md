Experimental factorwise Muon LoRA
================================

The opt-in `lora_update_space: factorwise-v1` route applies Muon independently
to the A and B factors of hidden attention and feed-forward matrices. Other
factors, including the prediction head, use auxiliary AdamW. This is distinct
from orthogonalizing a full dense weight update and subsequently factorizing it.

Owner optimizer profile schema 3 binds this selection, mode and rank. Schema 1
and schema 2 retain their previous contracts. The update-space marker is part
of the native optimizer checkpoint configuration; a resumed job cannot omit it
and silently reinterpret the saved moments. Profiles currently retain the
existing rank limit of 256. Higher ranks remain pending implementation and
qualification; these results do not establish their memory requirements.

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
