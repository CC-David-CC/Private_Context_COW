# Optional structured Lark grammars

PCC can now register an explicit `lark` grammar alongside GBNF, JSON Schema
and bounded-thinking definitions. This lets a host retain action and array
rules as parser structure while keeping strings as lexical units. A BW
repetitive-array diagnostic that exhausted the original lexer's 250,000-state
limit accepted 512 saved tokens plus 4,096 synthetic tokens through this path,
with the same parser limits. This is CPU parser evidence, not a model-quality
or complete workflow-recovery result.

## Register and use it

An ordinary server-owned startup or inline registry can contain:

```json
{
  "schema": 1,
  "states": {
    "ANSWER": {
      "format": "lark",
      "source": "start: \"A\" | \"B\"\n"
    }
  }
}
```

The existing immutable grammar registration API owns admission. No source path
is opened from this definition. Lark also works as a thinking definition's
`final` format. Use the same bounded source size, pinned llguidance 1.9.1,
tokenizer binding, EOS policy and canonical controller as the other formats.
Grammar constrains syntax; the host still authorizes and executes tools.

Lark distinguishes parser rules and lexical terminals by name. See the
[llguidance syntax documentation](https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md)
and the pinned dependency's GBNF conversion implementation. Preserving every
single-character rule as a parser rule is not a general optimization: that
alternative hit the separate parser-item limit in our diagnostic. Select the
boundaries deliberately and verify both acceptance and rejection.

## Identity and recovery

The format and exact source bytes participate in the grammar specification
and engine identity. A changed definition cannot restore an old controller.
Existing GBNF definitions and their identities retain their original behavior;
a test pins a pre-change GBNF engine hash. No live study source, saved budget
or registered grammar was edited to adopt this feature.

For BW, use its corresponding optional `cow.grammar_syntax: "lark"` configuration
for a new run. This is not a way to resume an indeterminate old generation under
a different grammar. Reconcile the original operation ID and its branch/receipt
first. Preserve partial evidence if it cannot be resolved. A later new run has
new identifiers and must not be reported as recovery of an unknown operation.

## Evidence and limits

The [CPU qualification receipt](../experimental/branching_context/evidence/runs/lark-grammar-20261026-cb.json)
records source/log hashes, bounds and the alternate attempts:

- 40 PCC grammar, schema, thinking, identity and restore checks passed.
- G0 passed 58 tests; no cumulative G1–G5 qualification is claimed.
- BW's real CLI/HTTP, filesystem, journal and offline-reopen controls passed
  with a scripted CPU backend. These are plumbing tests, not model inference.
- Synthetic empty-string arrays accept 4,096 entries and reject 4,097. A
  deliberately widened array-bound mutant was detected by that test.
- A real MiniCPM tokenizer replay accepted the saved 512-token prefix plus
  4,096 synthetic tokens with the original resource limits. The missing
  original suffix was not reconstructed, so the original crash's precise
  token sequence and cause remain unproven.

The original repetitive model output remains a model-failure observation.
Parser exhaustion remains infrastructure evidence. Long arbitrary strings,
browser interactions and native GPU workflow behavior on this new path have
not been qualified by these controls. No training update or protected model
question was repeated to produce them.
