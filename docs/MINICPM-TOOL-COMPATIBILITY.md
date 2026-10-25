# MiniCPM5 tools through the compatibility API

This adds a CPU tested MiniCPM5 tool codec to PCC's optional Chat Completions and
Responses frontend. It translates the model's tool syntax into API tool calls.
It does not establish model task quality, native numerical correctness, or release
acceptance. The native COW/BW grammar path is a separate interface.

MiniCPM5 uses bare calls:

```xml
<function name="read_file"><param name="path">notes.txt</param></function>
```

Qwen's `<tool_call><function=...><parameter=...>` format keeps its existing parser
and renderer defaults. A client model name cannot select a parser. The owner
loading MiniCPM model metadata binds the exact template SHA-256 to
`minicpm5-xml-v1` and `minicpm5-json-values-v1`; the frontend verifies that binding.
The existing shared `renderer_identity` helper is the identity authority, including
the actual codec/serializer implementation hashes. The compatibility frontend
validates any owner-supplied renderer ID against it; unchanged version labels
cannot rejoin a cache after implementation changes. Missing or mismatched
encoder/template bindings are refused. Changing the serializer changes the renderer and compatibility-cache identity.
Existing consumed COW history is never rewritten or joined across identities.

A MiniCPM-family model with a valid text-only or custom template can still start
and render text when no compatible tool codec is available. That does not qualify
or authorize tools: tool requests fail closed before encoding, and unrequested
tool events cannot become actions on that text-only route.

## Supported data and refusal behavior

String parameters preserve their bytes. Strings containing `<`, `&` or a newline
use the official template's CDATA framing. A string containing `]]>` together
with characters requiring CDATA cannot be represented by this template and is
refused in submitted history. There is no invented escape. XML entities are not
decoded. Other parameter values are strict JSON: nested objects and arrays,
booleans, null and finite numbers. Python representations such as `True` or
`{'key': 1}` are not accepted as JSON arguments.

The MiniCPM-only Jinja finalizer serializes nonstring JSON values correctly when
rendering assistant tool-call history. Strings and Jinja `Undefined` are preserved.
The model artifact and the shared Qwen rendering environment are unchanged.

Tool schemas must have an explicit object root and declared wire parameters.
Supported schema keywords are `type`, `properties`, `required`,
`additionalProperties`, `items`, `enum`, `const`, `anyOf`, `oneOf`, `allOf`, numeric
bounds, string/array length bounds and `uniqueItems`, with description/title/default
annotations. Nested open JSON objects are supported. Top-level wire arguments
must be named in `properties`, even if the schema permits additional properties.
Root `allOf`/`anyOf`/`oneOf` schemas are refused: they can introduce fields that
are not declared on the root's named-parameter wire. Nested combinators inside
declared property schemas remain supported. Parameter schemas mixing string and nonstring wire types are refused because the
native syntax cannot unambiguously distinguish raw strings from JSON values.
Unsupported keywords such as `$ref`, `pattern` and `format` are admission errors;
there is no object-only validation fallback or claim of complete JSON Schema support.
Defaults are never applied to fabricate missing arguments. Historical API call
arguments must explicitly contain a JSON object (or its valid JSON encoding).
Malformed strings, scalar/array arguments, duplicate JSON keys and nonfinite
values are refused before the shared legacy history normalizers can repair them.
MiniCPM control containers (`messages`, `tool_calls`, `tools` and their object
elements/function objects) must use canonical arrays and objects. Nonstandard
JSON-string array wrappers are refused before legacy decoding; standard
`arguments: "{...}"` object encodings remain supported. Standard Responses
`input: "..."` remains a user message, even when the text looks like a JSON array.
Absent, null and empty no-tool fields keep their existing no-tool meaning.

Names must exactly match a declared tool; dotted namespace names are supported.
Unsafe attribute names, duplicate declarations/parameters, missing required fields,
invalid typed JSON and unsupported schemas are refused. Unsupported tool requests
fail before prompt encoding, cache admission or native work. Forced/hosted tools
remain outside this frontend's supported scope. `tool_choice: "none"` exposes no
action; submitted history still requires valid declarations. Set
`parallel_tool_calls: false` to allow at most one completed call.

## Streaming, literal examples and bounds

An unquoted function tag is reserved action syntax anywhere in content, including
after a prose prefix such as `Let me check. `. A prose prefix does not make the
call literal. Use inline backticks or fenced code to show an example as data.
Inline spans match their backtick delimiter width, including across line breaks;
backtick/tilde fences close only with a matching marker and sufficient width.
Calls inside code or owner-configured reasoning remain text. Adjacent completed
function blocks are allowed. CDATA is always data, including text that looks
like another call.
Unexpected unquoted `<think>` or `</think>` in content refuses the turn, whether
thinking is off or an owner-configured reasoning block has already closed. A
second block cannot silently turn thinking back on or expose actions from it.
Fenced/inline code and CDATA containing those markers remain data.
This quoting policy is explicit and narrower than interpreting every `<function`
substring as an action.

Partial tags and calls are held across chunks. A parser buffers at most 262,144
characters, accepts at most 32 calls, and limits cumulative call text to 1,048,576
characters. Schema declarations have a 256 KiB admission bound and explicit depth
and node limits. Tool events are emitted only after a clean natural stop: a later
unknown, malformed or truncated call prevents earlier pending calls becoming
actionable. A cancelled or length-limited turn is not a completed tool response.
Refusals are API errors, not silent tool-name repair or a model-learning lesson.

## Evidence and CPU checks

The protocol was checked against the locally pinned llama.cpp revision
[`972d2313`](https://github.com/ggml-org/llama.cpp/tree/972d2313bc0bf0a45f634f77d95c9fb03aeab12c):
[`common/parsers/minicpm5.cpp`](https://github.com/ggml-org/llama.cpp/blob/972d2313bc0bf0a45f634f77d95c9fb03aeab12c/common/parsers/minicpm5.cpp)
and its primary examples in `tests/test-chat.cpp`. Thanks to the llama.cpp
contributors and OpenBMB for the MiniCPM template and documented tool format.
The Python codec is an independent implementation of that protocol.

The actual local official MiniCPM5-2B template was read from GGUF header metadata,
without loading tensor data. Its template SHA-256 is
`cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c`.
Portable fixtures additionally exercise the semantic schemas from eight Hermes
and twenty DeepSeek harness tool declarations: object/array nesting, enums,
numeric bounds, `anyOf` boolean/array alternatives, required fields and additional
property policies. Only declaration schemas are included; task messages and
protected diagnostic answers are excluded. Annotation descriptions were removed
without removing actual properties named `description`.

Run the portable CPU gates from the checkout:

```sh
PYTHONPATH=.:experimental/branching_context/src python -B -m unittest \
  serve.test_minicpm_tools serve.test_minicpm_compat \
  tools.test_cow_compat_frontend tools.test_cow_compat_cache
```

Set `PCC_MINICPM_GGUF` to an existing local official GGUF to include the separate
actual-header/template/factory gate. It reads metadata only and performs no model
inference. The HTTP tests script a fake COW request to check both API dialects,
streaming, typed arguments, refusals and identity separation. Native model tool
loops remain a separate qualification step; CPU codec success is not model success.


## First native CPU qualification, 2026-10-25

The reviewed codec source is `f9265f69c2e06524341443c66cc485567c9d6482`.
A fresh source-bound CPU profile passed all 2,438 Python/native token-ID
comparisons, all 130,560 vocabulary-piece comparisons and eight framing checks.
The bounded branch/restore probe passed eleven checks, including immutable parent,
independent suffixes, checkpoint restore, incorrect-sibling detection and explicit
reconstruction after eviction. All six matching-batch logit comparisons had zero
maximum difference. Different fresh-context batching showed approximately 0.0155
maximum drift; that is reported separately, not hidden by changing tolerances.
The existing comparison tolerances remain absolute 0.01 and relative 0.001.
These are correctness observations on CPU, not GPU throughput measurements.

The real model completed one string-only two-turn echo conversation: its CDATA
string preserved newlines, markup, ampersands and quotes; the tool actually ran;
the model then returned the exact receipt after seeing the tool result. A separate
nested-object case emitted Python object syntax instead of strict JSON. PCC
returned an error and no tool ran. Adding explicit strict-JSON system guidance
produced the identical malformed output for that case. This is two variants of
one typed case, not a measured general failure rate or a successful mitigation.
Native consumed tokens also confirmed the assistant generation prefix was prompt
input. The typed failure is not evidence that thinking tags were wrongly generated
or that the prompt prefix was missing.

The original failure remains preserved in the owner's private, unreviewed
failure-class collection. No teaching pair has been approved and no weights were
updated. The service shut down cleanly with no queued, active or pending work.
String-only interoperability does not qualify arbitrary typed tools, model task
quality, production deployment or compatibility-API recovery/idempotence.

Sanitized evidence: [native CPU gates](../validation/minicpm-tool-codec-20261025/native-cpu-bn.json)
and [native tool probe](../validation/minicpm-tool-codec-20261025/native-tool-probe-bn.json).
Raw diagnostic inputs/outputs remain private; the original first HTTP error body
and raw request wire serialization were not captured and are not reconstructed
as observed evidence.
