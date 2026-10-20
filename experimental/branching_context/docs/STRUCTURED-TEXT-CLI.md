# Explicit structured text loading

`load --file TEXT` continues to load one plain UTF-8 system message. A file that
happens to contain JSON is still literal text under this option.

Use the new explicit `load --messages-file MESSAGES` to preserve a JSON array of
text messages with `role` and `content`. Supported roles are system, user and
assistant. Order and exact string contents are preserved in the POST body.
This loads a closed context, without an assistant generation prompt; use `ask`
for its first question. For a prepared first answer, use the existing `prepare`.

The options are mutually exclusive. Structured files are bounded to 1 MiB and
1024 messages. Duplicate fields, extra fields, unknown roles, malformed JSON and
non-string contents reject before the request journal or model changes. This
option is text-only; image input continues through the existing ordered parts
and namespace-authorized uploaded IDs.

BranchWrite's CLI transport now uses `--messages-file` for `load_context`. It
previously wrote JSON but invoked `--file`, changing the intended messages to one
system string. The actual HTTP/CLI canonical-token regression reproduced this
bug before the fix, then passed. Plain-text loading keeps its original behavior.

Executed on llm-60, 2026-10-17: 32/32 affected COW CLI checks and G0 51/51 PASS;
the corresponding BranchWrite source passed its complete 324/324 tests. The
subsequent GOLDEN PATH source has additional tests, reported separately. These
are CPU control tests using real HTTP/CLI/store boundaries, not numerical GPU
qualification. No native engine, tolerance or reference equation changed.

Frozen source: `/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/cow-structured-green-source-v1`.
Profile e98e6438dad2e158e1ea8ce9ed2847212111d3feb9e567dc825d57895e8413df.
Keep distinct profile/source identities for the updated CLI closure. Old no-thinking
checkpoints are not retagged as thinking-capable. Full VC/G qualification remains
separate from this interface correction.
