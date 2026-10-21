# English Research model adapter and comparison

This branch reuses the tested Qwen35 llama.cpp adapter through its architecture boundary. It does not rename the original Qwen model or reuse its tokenizer, checkpoints, or model contract.

The requested immutable model input is `/home/dflanag3/model-identity-rematch-codex-v0.1.0/work/english-v4-20261021/output/research-model.gguf` on llm-49. It is an existing, untrained user research artifact; this work performs inference only and leaves it unchanged.

The inspected model has 427 tensors with the same names as Qwen3.5-9B. The token embedding and output shapes change from 4096 x 248320 to 4096 x 127991. Other tensor shapes match. The research weights are BF16 (250 tensors plus 177 F32); the measured Qwen baseline has 250 Q8_0 plus 177 F32. Task results therefore compare complete model configurations, not quantization or rematching in isolation.

## Reproduce

Use the existing `scripts/prepare_qwen35_profile.py` with the research model path, the pinned selection build, FP16 KV, capacity 16384 and batch 512. This hashes the complete actual model and produces its own compatibility identity. Run `scripts/qwen35_restart_probe.py campaign` against that profile at 128/4/8, then 1024/64/16, then 8192/512/16. Stop on the first mismatch. The probe's independent native reference never imports a checkpoint. Each restore must consume zero saved-prefix forward tokens and must exactly match its own model's recorded native batch arrangement.

Configure the normal authenticated server with `tools/cow_server_setup.py`, using the research weights and the same pinned native selection build, capacity 65536, prefill 512, then start with `tools/cow_server.py`. Give the separate BranchWrite client the new server's model and execution identities and EOS values from authenticated `/cow/info`; do not reuse the Qwen connection pins or pending operation IDs. Keep API credentials in process memory on the client.

Use the same unchanged website prompt and browser acceptance checks as the Qwen baseline, then the PDF and CAD tasks. Retain failed attempts, actual output artifacts and independent checks. No operator-written replacement candidate counts as model success. Any retries receive distinct recorded operation identities according to the normal workflow.

## Limits

Matching-schedule greedy continuation is the initial correctness scope. This is not arbitrary batch-arrangement equivalence, full controller qualification or G1?G5 certification. Architecture alone does not establish vision compatibility. The integrated mixed adapter now passed a tiny real-image fixture: 38 exact comparisons and three zero-prefix-work restores, plus actual encoder ingestion and generation through the separate HTTP client. Larger research-model image contexts and broad visual competence remain unqualified.

Private operational evidence and live campaign status are under `/home/dflanag3/spatial-artifacts/cow-english-research-9b-20261021/`. They contain machine-specific paths and are not public release artifacts. Do not commit credentials or model weights.
