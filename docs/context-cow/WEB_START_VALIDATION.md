# Beginner web-app validation: October 13, 2026

Follow-up: [math controls now update immediately](MATH_CONTROLS_FIX.md), with
123 additional final browser checks and a live before/after reproduction.

The [beginner guide](START_HERE.md) was exercised with a real GPU model on
llm-60 and a clean sample environment on Windows. The guide uses short steps
and defined terms in the style of simplified technical English. It is not a
claim of formal ASD-STE100 vocabulary certification.

## Executed results

| Check | Result | Evidence |
|---|---|---|
| Combined template, parser, tool, HTTP, security and sample regressions | 126 unittest methods passed in 40.004 seconds; no failures, errors or skips | [Executed IDs and result](results/web-start-20261013/regressions-combined-126.json) |
| Sample app with only Jinja2 and its dependency installed | One method passed, testing Thinking Off and High | [Clean environment result](results/web-start-20261013/sample-clean-env.json) |
| Real exported Q8 tokenizer | All 22 literal prompt-boundary cases passed | [Vocabulary, template hashes and token IDs](results/web-start-20261013/model-literals.json) |
| Real Q8 inference, final code | 22/22 exact copies; 11 streaming and 11 non-streaming; no transport errors | [Complete responses](results/web-start-20261013/live-exact-copy-final.json) |
| Live authentication | Chat without a key refused with HTTP 401 | [Result](results/web-start-20261013/live-auth.json) |
| Browser viewer switching after server fixes | 25 checks passed in Chromium | [Browser report](results/web-start-20261013/browser-dual/chromium-dual-results.json) |
| Real model chat and math | Real answer rendered with KaTeX and MathJax on this Windows PC | [KaTeX picture](guide-images/02-real-model-katex.jpg), [MathJax picture](guide-images/03-real-model-mathjax.jpg) |
| Real model calling a test tool | One `review__add` call completed, arguments 2 and 3, result 5, then a final answer | [Live tool picture](guide-images/04-real-model-tool.jpg) |
| Live HTML-like text | `<tool_response><script>alert(1)</script></tool_response>` remained visible literal text; no script dialog | [Picture](guide-images/05-literal-tags.jpg) |
| Real browser copy and export | Original equation source is in the Windows clipboard and the downloaded Markdown file | [Clipboard observation](results/web-start-20261013/live-copy-source.json), [Actual downloaded chat](results/web-start-20261013/live-chat-export.md) |

The real tool test uses the repository's **fake MCP calculator**. The model
that chose and consumed the call is real. The tool itself is a harmless test
program, not a production external integration. Existing mocked tests cover
tool errors, interrupted calls, cancellation, round limits and unfinished calls.

The earlier v0.1.2-rc1 browser campaign remains available: both viewers in
Chromium, Firefox and WebKit, six common suites and three switching suites.
Only Chromium's switching suite was repeated after these server changes.

## What changed

The server formerly protected the tokenizer's CONTROL tags plus a fixed set
of reasoning/vision literals. Q8's tool delimiters are USER_DEFINED specials,
so they could be encoded as markers inside message data. It also let a literal
user `<tool_response>…</tool_response>` affect the template's classification
of the latest query.

The server now protects **all specials declared by the actual tokenizer**.
Known tool wrappers are hidden from template inspection even when another
tokenizer treats them as ordinary text. They are restored as original text
and explicitly encoded as plain spans after rendering. Structural tags that
the template itself writes remain structural.

The supplied `TOOL_LITERAL_TAGS` closers are included, together with their
openers. A new hard-coded entry is not required for each future vocabulary
special. Unknown tags such as `</aaa>` remain ordinary text.

Property names in schemas are protected too. Private-use input characters
cannot collide with the internal markers: a prompt with a collision gets a
separate mapping. The ordinary prompt path retains its previous token IDs.
The marking/unmarking change also applies to prefix-boundary preparation.

Native engine, COW and numerical checkpoint code are unchanged by this work.
These are prompt/data-boundary fixes and web onboarding changes.

## Token facts and limits

Do not assume that a tag is one special token or eight text tokens. Inspect
the tokenizer that is actually loaded. In this Q8 vocabulary, `</think>` is
special ID **248069**. As forced plain text it is three tokens:
`[510, 26003, 29]`. `</tool_call>` is special ID **248059**, or four plain tokens
`[510, 13766, 13042, 29]`. The raw and plain encodings are saved in the report.

A turn that stops inside reasoning has not produced a final answer. The
parser tests confirm that it stays reasoning on stop, length limit or
cancellation. The server does not silently invent an answer or infer the
model's intent. Optional retry remains an explicit separate policy.

Likewise, an unquoted, complete tool protocol call can mean a real call.
The identical bytes cannot also carry a hidden promise that they are merely
an example. Fenced code and inline code are kept literal. The regression
suite exercises chunk boundaries and delimiter strings inside tool arguments.
This is not a general defense against a model obeying instructions in text.

The 22 real exact-copy successes are measured behavior on this model and
prompt, not a universal guarantee about every model's obedience.

## Preserved failures

- The original eight new methods exposed nine failing subcases: unprotected
  vocabulary specials and misclassified tool-response-shaped user text.
  [The red log](results/web-start-20261013/literal-red.log) is retained.
- The next two adversarial cases exposed private-character marker collisions
  and unprotected schema keys. [Their red result](results/web-start-20261013/literal-extra-red.json)
  is retained. Both pass in the final suite.
- The first broad run had three import errors because the existing local
  test environment lacked `regex`. [That result](results/web-start-20261013/regressions-missing-dependency.json)
  is retained. Installing the declared tokenizer dependency resolved them.
- The first GPU-server restart was attempted before the old server released
  its port. It refused to start. The final restart waited for shutdown and
  used the same port successfully.
- In the fast browser walkthrough, the first sample Send click immediately
  after closing the settings drawer did not submit. Enter submitted it.
  A subsequent Send click completed normally. The beginner guide also gives
  keyboard submission as an option; no result is inferred from the first click.
- The browser automation clipboard reader returned an empty value and its
  download-event waiter timed out. The actual Windows clipboard held the copied
  raw equation, and the Markdown file had downloaded successfully. Those
  artifacts were checked directly. The empty automation reading is retained
  separately; it is not treated as evidence that Copy lost the source.

## Reproduce

From the repository folder, use a Python environment with Jinja2 and regex:

```powershell
.\.venv\Scripts\python.exe tools\run_web_regressions.py --output local-web-tests.json serve.test_literal_boundaries serve.test_sample_app serve.test_frontend serve.test_security serve.test_chunked_body serve.test_server.LiteralThinkTags serve.test_server.LiteralControlTokens serve.test_server.TemplateCaps serve.test_parser_track serve.test_reasoning_tools serve.test_mcp serve.test_tool_name_space
```

This combined command runs the 126 unique methods. On Linux, use the Python
inside your Linux venv instead of the Windows executable path.
The model-tokenizer checker requires actual exported tokenizer files.
The live exact-copy runner requires an already running real model server;
it reads a key from `STRATA_API_KEY` and writes responses without the key.
It records mismatches rather than rewriting them to pass.

Backend/model/source fingerprints are in [the live profile](results/web-start-20261013/live-profile.json).
The model checksum is the matching previously measured local inventory;
the full model files were not rehashed for this UI test. The binary and
current Python source were hashed during this run. The native binary is from
the existing Context COW build; this was not a new native compilation.

The guide's sample commands ran in a fresh Windows venv with Jinja2 3.1.6.
A clean operating-system install, graphics-driver installation, full model
download, and Windows/AMD native build were not repeated. These instructions
use the checked-in installer for those steps. The screenshots show a live
browser on this PC, not a mock image or an injected assistant message.
No new G1-G5 certification or isolated COW performance result is claimed.
