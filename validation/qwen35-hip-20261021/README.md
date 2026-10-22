# Bounded AMD dense-model runtime validation

Tiny, 1K and 8K mixed-input campaigns passed on a real RX 7900 XTX with
Qwen3.5-9B Q8, FP16 KV, pinned llama.cpp HIP and the existing Qwen adapter.
There were 178 exact full-logit comparisons and nine strict process-separated
restores, with zero saved-prefix forward work. The large fixtures generated
16 tokens per branch. Native logs place attention and recurrent buffers on
ROCm0; sampled VRAM and utilization are recorded in runtime-final-native.json.

These restart stages reuse real image rows produced on NVIDIA. A separately
built local HIP encoder also passed actual execution, producing 48 finite
4096-wide rows. Cross-device equality of embedding bytes is not promised.
Normal launcher/authenticated HTTP integration remains pending.

Actual NVIDIA inventory regression confirms the hardware helper preserves its
previous CUDA profile fields; that inventory test is not an inference run.
No full G1-G5, arbitrary batch equivalence, visual competence or physical GPU
page-sharing qualification is claimed. runtime-incremental.json is historical.


Follow-up: the normal launcher/authenticated HTTP image path passed from a separate BranchWrite SDK client, including fork and unchanged-answer retry. A subsequent server restart restored the saved image-bearing prefix and produced 208 identical tokens with zero prefix work; see http-image.json and http-restart.json. The pending statements above describe the earlier native-only stage. An initial replacement server refused startup during GPU lock cleanup; wait for lock release after stopping the owned server. No model mutation was resent.
