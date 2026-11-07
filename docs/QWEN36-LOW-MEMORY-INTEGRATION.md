# Qwen3.6 A3B Q4 integration

Status: planned; original PR source preserved, PCC serving support not implemented.

Source: Strata PR 2925, commit 7e36db2cf0f3b5b6dd177b22c723a279fd748633.
The patch and bounded cache header are in docs/archive/qwen36-low-memory/.
PCC main lacks src/qwen35/q36_engine.cu: cherry-picking alone is insufficient.

## Implementation sequence

1. Inspect the branching-context specification and current native backend ABI.
   Identify the smallest Qwen3.6 engine dependency set and model metadata contract.
   Compare restoring that engine against extending the current native adapter;
   select based on exact continuation and bounded expert memory, not chat alone.
2. Add deterministic failing tests for backend selection, unsupported formats,
   low-memory limits, leased expert eviction and unused MTP unloading. Preserve
   the original cache semantics; reject budgets exhausted by live readers.
3. Implement the opt-in backend and setup/profile controls for low-memory mode,
   expert RAM budget and MTP. Account separately for weights, live context,
   checkpoints, staging buffers and OS memory. Bind settings to profile identity.
4. Implement complete native state capture/restore or explicit history replay.
   Validate independent fresh-prefill logits, fork isolation, controller state,
   eviction/rebuild, persisted restart and incompatible-profile rejection.
   Deep-copy/replay correctness does not establish physical prefix sharing.
5. Wire normal cow_server_setup.py / cow_server.py and existing client APIs.
   Add cancellation, error propagation and resource cleanup tests. Never present
   standalone Strata chat as proof of PCC integration.
6. Build/test on llm-49; qualify the actual Surface16 6 GB VRAM / 16 GB RAM
   configuration with 2048 context, 256 prefill, 6 GiB expert RAM, MTP off.
   Measure peak RAM/VRAM, startup, prefill/decode and checkpoint costs. Validate
   CUDA; label HIP/SYCL untested until actually built and exercised.
7. Record source/build/model hashes and results, document setup and limitations,
   then publish reviewed feature branch. Main integration requires passing checks.

## Model acquisition

Repository: unsloth/Qwen3.6-35B-A3B-GGUF
Revision: a483e9e6cbd595906af30beda3187c2663a1118c
File: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
Bytes: 22360456160
SHA256: 707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450

Download once onto llm-49 model storage and reuse over LAN. Verify hash and inspect
GGUF metadata/chat template before execution. Model weights stay outside Git.
Historical PR throughput is not a new measurement or acceptance result.

## Initial engine checks (2026-11-07)

llm-49 generated 128 coherent tokens using the archived PR engine plus an optional
MTP metadata compatibility fix. Decode: 184.2 tokens/s; total process wall time:
12.58 s. This is standalone engine evidence, not PCC context/server validation.
The downloaded model hash matched the pinned SHA256. It declares qwen35moe
architecture and omits nextn_predict_layers; absent optional MTP means zero layers.
The fix is retained separately on box-git in ~/qwen36-smoke-fixes pending tests
and integration into the chosen backend.

Compiler requirement: support GCC 14 and GCC 15 through toolkit-aware selection.
CUDA 12.9 rejects GCC 15: select installed GCC 14, without bypassing the guard.
llm-49 builds with GCC 15 / CUDA 13.4. Surface Pascal requires CUDA 12.9 and the
archived engine STRATA_EXPERIMENTAL_SM60 option; its checks are still pending.
