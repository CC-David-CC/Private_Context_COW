# Startup grammar registry, version 1

The host may load one local JSON registry when it starts the grammar-enabled
service. The file is server configuration; requests select a state name and
never provide a path or grammar source.

The document has exactly two top-level fields: integer `schema` set to `1`,
and a non-empty `states` object with at most 64 entries. Each state maps to an
inline GBNF definition:

```json
{
  "schema": 1,
  "states": {
    "read": {"format": "gbnf", "source": "root ::= \"read\""},
    "finish": {"format": "gbnf", "source": "root ::= \"finish\""}
  }
}
```

Each state entry contains only `format` and `source`; `format` must be
`gbnf`, and `source` is UTF-8 text. A grammar may not exceed the existing
1,000,000-byte parser input limit. The complete registry is limited to 4 MiB
and the sum of its source text is limited to 3 MiB. Duplicate JSON keys,
unknown fields, non-regular files, invalid UTF-8, and invalid grammars are
rejected at startup.

The service compiles the whole registry against its configured tokenizer and
EOS token IDs using the existing `GrammarEngine`. Its engine identity already
binds grammar names and source bytes, tokenizer identity, EOS IDs, and the
llguidance version. A changed definition therefore cannot resume a controller
created under the previous engine. Parser progress remains in the existing
per-branch controller and is restored by replaying canonical history.

The bounded Q4 image client path also accepts this registry. Parser scope and
repetition counts use actual text tokens from canonical mixed histories; image
rows remain part of model input and position receipts. Image branch continuations
and new image questions can select a named grammar with greedy selection.
Image sampling, reseeding and text-stop combinations remain unqualified.
Generation descriptor schemas 10/11 bind the grammar to mixed-source text or
typed image suffixes respectively. Existing schemas 8/9 keep their original
greedy semantics without a grammar field; checkpoint formats are unchanged.
See `evidence/VISION-GRAMMAR-CLI-2026-10-17.json` for executed scope.
