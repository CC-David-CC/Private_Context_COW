# BW and standard CLI clients on one COW server

An optional second HTTP port serves text Chat Completions and Responses clients.
BW continues to use the native `/cow/...` API. Both listeners use the same
coordinator, native model workers and bounded RAM/disk checkpoint pools. This
does not load a second model or change BW's workflow ledger and recovery contract.

Configure with the usual model/native-build arguments, adding:

```sh
python tools/cow_server_setup.py --model /owned/Wen-1.1-BF16.gguf \
  --native /owned/text-build/libqwen35ctt.so --output /owned/server \
  --gpu 0 --context 65536 --prefill 64 --threads 8 \
  --math-profile f32-activations --ram-gib 100 --disk-gib 100 \
  --min-available-gib 8 --ephemeral --port 8099 \
  --openai-port 8100 --openai-model wen-1.1
python tools/cow_server.py /owned/server
```

Use the native COW address on port 8099 for BW. Configure the other CLI's custom
provider with `http://127.0.0.1:8100/v1` and model `wen-1.1`; GET `/v1/models`
lists the configured alias. The API key is the same private key created by setup.
Use SSH forwarding or an authenticated trusted-LAN bind when clients are remote.
No CLI source patches or cloud-model credentials are required. Configured servers
without `--openai-port` keep their previous listener and launcher arguments.

## Reuse and ownership

Standard clients usually resend their conversation. The compatibility boundary
renders and tokenizes it with the serving model's literal-safe tokenizer/template.
It looks for the longest exact stored token prefix under the pinned model/LoRA,
execution and template identity, forks that committed COW snapshot, and processes
the suffix. Both closed-input roots and completed generated histories can be reused.
Requests always get independent mutable continuations; siblings do not overwrite
each other. Wen's recurrent state is included with K/V and decoder state.

Editing earlier messages, changing tools, omitting reasoning or reformatting tool
arguments can change the prefix. Only a fully matching token prefix is reused;
otherwise prefill starts from a valid earlier checkpoint or a fresh root. Matching
the visible question or text alone is insufficient. COW's existing coordinator
checks native prefix work; this boundary does not bypass those checks.

The in-memory lookup is bounded to 128 entries and one million token IDs. The
normal `/v1` endpoint has no prototype 256-request lifetime quota. Derived numerical
state remains governed by COW's RAM/disk pools. Pool ceilings are not reservations
or whole-process limits: weights, active native buffers and canonical history are
separate. With `--ephemeral`, the existing marked, locked session directory owns
both numerical caches and history. Graceful exit removes it; the existing launcher
cleans abandoned unlocked sessions on the next start after a crash.

Choosing another client does not automatically migrate a live BW workflow. The
files, unfinished operations, permissions and task ledger need an explicit handoff.
The compatibility boundary is not a stateless dispatch fallback inside BW.

## Initial supported surface

Text `/v1/chat/completions`, `/v1/responses`, `/v1/models`, SSE text/tool events,
function/custom tool calls through Strata's existing parser/Responses assembler,
native supported sampling, seeds and text stops. The compatibility renderer uses
the nonthinking mode qualified by the CLI comparison. Output is capped at 2048
tokens per request by default; change `--openai-max-output` for an explicit limit.
The physical context ceiling comes from the pinned native profile.

This does not claim the complete OpenAI API: hosted tools, multimodal inputs,
forced tool choice, structured final-output schemas, background jobs, remote
compaction, stored response retrieval and `previous_response_id` continuation are
not implemented. Clients sending full history work without stored response IDs.
Their resubmitted requests can still reuse COW's numerical state.

The optional `/sessions/<32-hex-id>/v1/...` comparison endpoint imposes eight model
requests per session, with at most 128 sessions. Normal clients use `/v1`.
These are inference requests only: execution permissions belong to the CLI/BW
runner, not the model server. No training or automatic model promotion is enabled.

## Validation

Run CPU cache and launcher checks from the repository root:

```sh
python -m unittest -v tools.test_cow_compat_cache tools.test_cow_compat_setup
```

The experimental comparison installed Codex 0.163.0, OpenCode 1.19.4, Hermes
0.22.4 and Aider 0.86.2 on a 4090 client host, with Wen inference on a separate
96 GB NVIDIA host. Both original Wen and Wen 1.1 ran the same four owned task
intents with actual file-byte/code/writing oracles. This small pilot is evidence
of interoperability, not a general model/harness ranking. Raw prompts, answers,
credentials, weights and runner configuration homes stay outside public Git.

The answer-free [validation receipt](context-cow/results/cli-compat-20261022/summary.json)
records 31 passing Linux CPU checks, without failures or skips. In the normal
dual-port launcher, all four CLIs completed the code-repair control. A BW native
context was parked while they worked, kept the same branch head, and resumed to
the correct complete response afterward. Its 78 resumed output tokens required
78 native forward tokens, with checkpoint import rather than prefix replay.
Only one model allocation was present. Both text and mixed-capable dense builds
passed repeated text requests; the second request reused 12/19 prompt tokens.
Both ephemeral session directories were removed after exit and GPU use returned
to the 15 MiB idle baseline. The mixed test first exposed a typed-history
admission error; the fix and regression test are included in this change.

The separate four-task runner pilot passed:

| Model | Codex | OpenCode | Hermes | Aider |
|---|---:|---:|---:|---:|
| Original Wen | 1/4 | 2/4 | 2/4 | 1/4 |
| Wen 1.1 | 2/4 | 1/4 | 1/4 | 1/4 |

An additional procedural hint fixed the byte-format task for OpenCode and Aider
with both models; Codex and Hermes still failed it. These 40 trials are not a
full BW workflow comparison: they reuse BW task intents, with each runner's own
tools and prompts. BW's native-context coexistence was checked separately above.
The evidence supports keeping explicit file-byte controls and adding measured
check/revise feedback where useful, not replacing BW or declaring a universal
winning runner. No training occurred and no candidate model was promoted.

Official custom-provider references: [OpenAI Docs](https://learn.chatgpt.com/docs/config-file/config-reference),
[Hermes](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart),
[OpenCode v1](https://opencode.ai/docs/providers/),
[Aider](https://aider.chat/docs/llms/openai-compat.html).
