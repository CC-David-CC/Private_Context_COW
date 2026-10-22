# Experimental Wen CLI compatibility

This branch connects standard CLI protocols to the existing COW coordinator and
immutable Store. It is experimental and is not merged into main. BW keeps its
native COW contract; it receives no stateless replay fallback.

The standard client resends messages or Responses input. This frontend renders
and tokenizes the complete request with the serving model's tokenizer/template,
finds the longest exact cached token prefix, forks its committed snapshot, and
submits only the remaining input tokens. It indexes both closed-input roots and
successfully committed generated histories. Re-rendered tool calls/reasoning can
change tokens; those changes reduce reuse and must never be guessed equivalent.
No matching snapshot means a fresh prefill. No client source changes are needed.

The native state includes Wen's recurrent state, K/V and decoder controller.
The cache identity includes model/adapter and execution profile identities plus
the template hash. Snapshots stay immutable when requests fork. Numerical copies
are managed by COW's bounded RAM/disk pools, not by a second transcript database.
The lookup index is bounded; this first experimental process also stops accepting
requests after 256 to bound retained session metadata. Restart begins empty.

The entire Store and template live in a dedicated TemporaryDirectory and are
removed after graceful shutdown (including SIGTERM). Weights and private test
receipts are separate. SIGKILL or power loss cannot execute process cleanup;
orphan-directory recovery has not been implemented here. RAM and disk pool limits
default to 100 GiB each; these are ceilings for derived state, not reservations
or total process-memory limits.

Supported initially: text Chat Completions and Responses, function/custom tools
through Strata's existing output parser/Responses assembler, SSE streaming,
greedy or supported native sampling, stop sequences, and local model discovery.
The frontend advertises one pinned model at a time. No hosted tools, structured
final-output schemas, multimodal inputs, forced tool choice, remote compaction,
stored Responses IDs or background jobs are claimed. Unsupported operations fail
explicitly. Clients sending full history work without stored response IDs.
Requests cannot switch the pinned nonthinking template used by this comparison.

Run CPU cache invariants:

```sh
cd experimental/wen_cli_compat
python -m unittest -v test_cache.py
```

Start with a freshly prepared profile whose runtime-source closure matches this
checkout and an existing Wen native build (no model training or GPU is needed to
install the CLI tools):

```sh
python experimental/wen_cli_compat/server.py --profile /owned/profile.json \
  --model-id wen-1.1 --temporary-parent /owned/scratch --port 8100
```

Use an SSH tunnel or authenticated trusted-LAN bind. `WEN_COMPAT_API_KEY` supplies
the optional server credential without embedding it in source. `--trace` writes
private request/output receipts to a new file with mode 0600; never commit those
receipts, model files, prompts or reference answers.

The CLI comparison must use fresh task workspaces, identical underlying user
tasks, model bytes, output limits and file-content oracles. A runner uses its own
native tools/prompt format; feeding every runner BW's controller schema would
measure the wrong thing. Infrastructure failures are distinct from model/task
failures. Only complete responses and actual filesystem effects decide success.

Official provider references: [OpenAI Docs configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference),
[Hermes custom endpoint quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart),
[OpenCode v1 providers](https://opencode.ai/docs/providers/),
and [Aider compatible APIs](https://aider.chat/docs/llms/openai-compat.html).
Installed versions must be recorded with results; v1 and v2 OpenCode configuration
schemas are different and must not be interchanged.
