# JSON and JSON Schema beside GBNF

This opt-in mode makes the model choose tokens that fit a JSON schema. For
example, you can require a tool name and arguments, allow only two tool names,
and require a number where a number belongs. The host still checks permissions
and whether the requested action makes sense before doing anything.

The named grammar registry now accepts inline `json_schema` objects alongside
inline `gbnf` strings. Definitions are lowered and validated when the server
loads its registry. Continuations instantiate matchers from those stored grammar
definitions. Both use the same pinned llguidance matcher, token mask, canonical controller and
branch recovery path. There is no prompt-only or repair-and-retry fallback. The
unconstrained path does not load a schema when this mode is unused.

Example registry:

```json
{
  "schema": 1,
  "states": {
    "JSON_OBJECT": {
      "format": "json_schema",
      "source": {"type": "object"}
    },
    "READ": {
      "format": "json_schema",
      "source": {
        "type": "object",
        "properties": {
          "tool": {"type": "string", "enum": ["read_file"]},
          "arguments": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "enum": ["src/main.py", "tests/test_main.py"]},
              "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["path", "limit"],
            "additionalProperties": false
          }
        },
        "required": ["tool", "arguments"],
        "additionalProperties": false
      }
    },
    "DONE": {"format": "gbnf", "source": "root ::= \"DONE\""}
  }
}
```

Load it with the existing server `--grammar-registry` option and select `READ`
or `JSON_OBJECT` with the existing CLI `--grammar-state` option at a generation
boundary. The registry is host-owned. Requests select a name; they do not load
client-supplied file paths. An active partial generation keeps its original
schema. A new grammar starts only at the existing supported boundary.

## Exact scope

The initial tested constraints cover required properties, string enums for tool
names, nested argument types, integer bounds, rejected extra fields, boolean/null,
arrays with item/count bounds, escaped strings and Unicode. Syntax-only object
generation uses `{"type":"object"}`. An empty schema object describes general
JSON; it does not impose tool structure. Use `additionalProperties: false` when
extra keys must not appear, and keep host validation even with a schema.

The source is a JSON object, not a string containing JSON. Sources are detached,
bounded and canonicalized before both hashing and compilation. Property order
therefore follows canonical sorted schema keys. Reordering input dictionaries
cannot give one schema identity two different parser behaviors. This is an output
generator, not a validator promising to accept every equivalent JSON spelling.

Unknown schema keywords and non-local `$ref` references are rejected. No remote
schema fetch is permitted. `$defs` and local references, object/array/string/number
constraints and composition keywords are admitted to the pinned compiler; it may
reject unsupported forms. Only executed cases count as qualified. This is not a
claim of complete JSON Schema Draft 2020-12 coverage. `x-guidance` extensions are
not accepted, and compilation forces lenient/coercion options off. Metadata such
as title, description and examples does not constrain token choices.

The compiler has its own documented limits, including fixed property ordering
and duplicate-key limits for unspecified properties. See the
[llguidance JSON Schema documentation](https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md).
Implementation behavior is pinned to installed llguidance1.9.1 and tested
directly; current online documentation is not evidence that an untested feature
works in that installed version.

## Recovery and performance

Each branch owns parser progress, RNG, repetition history, decoder and sampler
state. Only immutable definitions are shared. Schema changes alter grammar
identity; incompatible saved controllers reject instead of resetting progress.

Compilation and startup must be reported separately from per-token mask/selection
cost. The feature adds no model forward pass, but schema complexity can still
affect CPU masking and end-to-end speed. No universal zero-overhead claim follows
from sharing the grammar path.

The 2026-10-17 real-Q4 CLI probe passed the tool-schema checks and a process
restart after four consumed output tokens. Its remaining40 tokens and output
bytes matched an uninterrupted reference with the same request boundaries. The
new process loaded119,422,656 checkpoint bytes, evaluated40 new tokens and did
not recompute the saved prefix. Continuation fields matched; distinct branch
stream identities remained distinct. No full-logit numerical oracle ran here.

The affected CPU suites passed122/122 and G0 passed51/51. A test-only mutant
that discarded schema constraints was caught by13 failed assertions across
three methods. The separate BranchWrite suite passed298/300; its two inherited
blanket JSON-policy assertions still fail and are recorded, not weakened.

For eight identical196-text-token outputs (one warmup plus three measured
repetitions per format), median whole-CLI throughput was29.65TPS for JSON Schema
and29.62TPS for GBNF. The roughly0.1% difference is within observed variation.
These timings include state work and CLI overhead. Native model-compute-only
medians were132.84 consumed tokens/s in both formats, excluding the rest of the
pipeline. This controlled constant-output fixture is not a long-context or
arbitrary-schema benchmark. The8% reporting threshold was not exceeded.

Using the actual Q4 tokenizer, engine construction (including tokenizer
adaptation/hash, source lowering, validation and initial matcher) took median
0.473s for JSON and0.476s for GBNF. New session construction took0.056s and0.058s
respectively. Tokenizer loading took0.308s separately. Those are startup/boundary
costs, not time added per generated token.

Machine-readable results and the verified private backup receipt are in
`experimental/branching_context/evidence/JSON-SCHEMA-2026-10-17.json` and
`JSON-SCHEMA-BACKUP-2026-10-17.json`. Full VC00-VC09 and G1-G5 qualification,
sampled-JSON GPU coverage and long-context measurements remain outstanding.

The former unit assertion that all JSON Schema formats were unavailable is
superseded by the owner's 2026-10-17 feature request. Its replacement checks a
valid tool action and rejects an unlisted tool. The original baseline failure,
new independent tests and invalid-schema rejection cases are retained. Numerical
reference equations and tolerances are unchanged.
