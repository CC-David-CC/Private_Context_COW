# Bounded AMD dense-model runtime validation

The tiny and 1K mixed-input campaigns passed on a real RX 7900 XTX, using
pinned llama.cpp HIP and the existing Qwen adapter. Native logs place attention
and recurrent buffers on ROCm0. The independent reference uses matching native
batch boundaries. These initial stages reuse image rows produced on NVIDIA;
local AMD encoding is a separate pending check.

The 8K stage and local image encoder were still in progress when this incremental
record was written. No full gate or general visual competence is claimed.
Actual NVIDIA inventory regression confirms the hardware helper preserves its
previous CUDA profile fields. It is not another inference regression.

Source/build scripts and raw logs remain on the Linux workers. Check the newest
runtime summary before treating this incremental file as final evidence.
