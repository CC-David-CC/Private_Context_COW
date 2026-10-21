# Qwen3.5-9B backend bring-up

Status: NOT SUPPORTED. Branch feat/qwen35-9b-backend. No GPU execution or restore pass claimed.

## Verified starting point

The local Q8 GGUF reports architecture qwen35, GPT-2 tokenizer, 32 layers,
4096 embedding width, 12288 FFN width, full_attention_interval=4,
16 attention heads, 4 KV heads, key/value dimension 256,
SSM convolution kernel 4, state size 128, group count 16,
time-step rank 32, inner size 4096. Context metadata is 262144.
RoPE base is 10000000; dimension sections are [11,11,10,0].
These are metadata observations, not proof of operator correctness or usable capacity.

Current tools/cow_server_setup.py rejects architectures other than qwen4exp.
It requires an expert pack and embeds Flash-specific attention, position and
state identities. src/core/branch_driver.cpp initializes Flash Prefill and Verifier;
BranchSession is coupled to SessionState/ModelGeometry and QSA pool layouts.
Do not remove the rejection until a real compatible backend is qualified.

## Small, ordered milestones

1. Inventory and tests: record actual Q8/BF16 GGUF metadata and tensor names,
   tokenizer/chat template, and projector availability. Keep input files immutable.
   Test missing or contradictory dimensions and explicit unsupported admission.
2. Dense text computation: map tensors, implement/reuse only verified compatible
   dense FFN, GDN/convolution and full-attention operations. No invented expert pack.
   Implement actual native forward-token/operator counters. First execute tiny
   prefill and continuation on llm-49; compare against an independently initialized
   reference with matching input/operator schedule. Preserve independent oracle.
3. Native checkpoint completeness: inventory every mutable KV, recurrent matrix,
   convolution ring, position, token frontier and logits. Reuse existing COW page
   ownership where layouts permit; deep-copy unsupported regions truthfully.
   Test sibling isolation, partial-page writes, append, switch, retry and eviction.
4. Durable resume: terminate native and Python processes, reload into new ones,
   require zero prefix forward work through first frontier read. Match continuation
   against uninterrupted identical-profile execution. Exercise failures/rejection.
5. COW admission: separate architecture-specific model/profile construction from
   common fingerprints, store, pools and protocol. Version incompatible payloads.
   Preserve Flash default behavior. CPU contract tests plus real-GPU evidence.
6. BranchWrite text: actual CLI on a separate client calls authenticated HTTP;
   run coding, PDF and CAD with fixed requirements, then branch/retry/restart.
   Qualify greedy before stochastic sampler/controller persistence and JSON/GBNF.
7. Vision: verify matching projector availability, implement real encoder rows and
   native mixed-input positions; verify image branch isolation and restart. Run
   real rendered-page/CAD/Blender inspection and website/browser checks.
8. Performance: compare direct execution, resident COW and disk resume; measure
   startup, prefill, TTFT, TPOT, decode, state preparation, hashes, RAM and VRAM.
   Start small, then 8K/32K. A 262K metadata limit is not a tested promise.

## Boundaries

Do not substitute a stateless ordinary chat server for COW. A supported alternative
native engine may be considered only after inspecting its export/import coverage,
state ownership and counters. Normal inference support alone is insufficient.
Keep canonical history/controller records and existing branch references. Native
payloads are derived checkpoints; model differences do not create shadow histories.
No claimed cross-model/quant checkpoint interchangeability. No tolerances changed.
No GitHub push in this bring-up; local work and private LAN backup are authorized.

## Concurrent operation

llm-60 continues the Flash domain campaign. llm-49 GPU was idle with 24071 MiB free
at initial inspection. The existing BranchWrite client/auditor processes on llm-49
must remain untouched. Use a distinct server port and owned paths for this backend.
