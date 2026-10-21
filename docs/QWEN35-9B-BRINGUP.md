# Qwen3.5-9B COW bring-up (Linux, experimental)

This branch adds a separate pinned llama.cpp adapter. Strata remains the Flash
engine. Both use the existing COW immutable Store, branch references, controllers
and pool coordinator. Qwen checkpoints currently copy whole native state; this
is not GPU-page sharing. Vision is explicitly refused until its integration passes.

## Build and configure

Start from this checkout with the existing CUDA-capable COW environment. Install
`requirements-cow-server.txt` in a virtual environment if it is not already set up.
Keep a clean llama.cpp checkout at commit
`972d2313bc0bf0a45f634f77d95c9fb03aeab12c`, with its CUDA libraries built in
`build/bin`. The build helper rejects a dirty or different checkout. It records
bridge, oracle, shared-library and inspected serializer-source hashes.

From the repository root, using your own absolute paths:

```sh
python3 experimental/branching_context/scripts/build_qwen35.py \
  --llama /path/to/llama.cpp --output /path/to/qwen-build

python3 tools/cow_server_setup.py \
  --model /path/to/Qwen3.5-9B-Q8_0.gguf \
  --native /path/to/qwen-build/libqwen35ctt.so \
  --output /path/to/new-qwen-server \
  --context 16384 --prefill 512 \
  --ram-gib 4 --disk-gib 8 --min-available-gib 4 \
  --host 0.0.0.0 --port 18109

python3 tools/cow_server.py /path/to/new-qwen-server
```

Do not pass a Flash expert pack. Configuration fingerprints the exact GGUF,
compiled backend, selected runtime sources, hardware and execution arrangement.
The API key is in the new server folder, mode 0600; share it privately with your
client. Bind only on the intended trusted network. No ordinary chat-server
fallback is used. Do not modify selected runtime source files under a running
profile; create a new configuration after changes.

## Reproduce native restart checks

```sh
python3 experimental/branching_context/scripts/prepare_qwen35_profile.py \
  --model /path/to/Qwen3.5-9B-Q8_0.gguf --build /path/to/qwen-build \
  --output /path/to/probe-profile.json --context 16384 --batch 512

python3 experimental/branching_context/scripts/qwen35_restart_probe.py campaign \
  --profile /path/to/probe-profile.json --output /path/to/new-results \
  --prefix 128 --suffix 4 --outputs 8
```

Use a new result directory. Expand only after passing: 1024/64/16, then
8192/512/16. Independent fresh contexts compute the recorded complete batch
sequence; producer and consumer run in different observed-exited processes.
Native inference runs inside each Python process in this adapter. The engine's
full hybrid serializer stores attention plus recurrent/convolution memory; our
versioned payload additionally retains next logits. Sampler/controller progress
remains canonical in Store. Restore must perform zero forward tokens.

## Current evidence and limits

See `experimental/branching_context/evidence/qwen35-9b-20261021/summary.json`.
The initial three-size campaign passed 178 whole-vocabulary logit comparisons
with zero error. Nine strict restores performed zero forward work. Controllers
were greedy; midpoint continuation and saved-parent byte integrity were checked.
Earlier GPU allocation failures and probe field-name mistakes are retained.
Their failures are not counted as passes. The allocation cause is not established.

Normal HTTP/CLI qualification is ongoing. The first real VM client exposed a
system-only prefix rejected by Qwen's unmodified chat template. The new encoder
derives a stable prefix using two reference user messages and verifies exact
text/token decomposition; it does not remove the template's validation.

Non-greedy samplers, BF16, AMD, vision and broad crash/pressure cases are not yet
qualified. Existing cumulative G1-G5 certification remains separate. Never
interchange Flash/Qwen, different quantization, or different-profile checkpoints.
