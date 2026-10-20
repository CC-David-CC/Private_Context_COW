# CLI seed and text-stop controls: initial design

Status: design and test inventory only. No production implementation or GPU
qualification is included. Public reseed command syntax and exact controller
schema version remain for integration review.

## Verified behavior at `df02950584575762c5c98eec4f881cd62917f358`

- `controller.start(policy, stream, history, vocab_size)` derives
  `stream_key = sha256(stream)` and initializes `next_draw = 0`. The canonical
  controller records the stream key and cursor, not the user seed.
- Sampled generation starts through `sampled_controller.start`; its controller
  is schema 3 or 4 and uses the same stream/cursor state. Coordinator-generated
  stream identity defaults to the operation ID. Retry descriptors retain the
  original `stream_operation_id`; a transport operation ID does not need to
  become a new RNG stream.
- `real.py:set_random_stream(snapshot, stream)` is a legacy controller
  transition. It hashes the supplied string, sets the cursor to zero, and
  preserves other controller fields. It is not exposed as a sampled/text HTTP
  operation and does not store a seed.
- Legacy `controller._consume_owned` accepts token-ID `stop_sequences` and
  detects them using `stop_tail`. Sampled-controller validation rejects
  nonempty `stop_sequences` and `stop_tail`; grammar-controller validation
  does the same. `min_p.validate_policy` does not accept text stop settings.
- `CowTextAdapter` incrementally decodes token bytes for display, but does not
  detect or suppress stop text. The current CLI has no `--seed` or `--stop`
  option. The HTTP text routes reject unrecognized fields.
- The API specification requires decoder state to retain stop progress and
  calls out multi-token stops with split UTF-8 as a required boundary fixture.

## Proposed invariants

1. A seed is accepted only when a generation scope is started. Seed `0` is
   valid. The canonical controller must retain the seed/derivation identity
   alongside the stream key so restart, fork, and receipt validation can prove
   which stream was selected.
2. Starting a generation with the same seed, same source controller/history,
   same policy, and compatible deterministic execution produces the same RNG
   stream. A branch fork does not implicitly choose a new seed.
3. Continue and exact retry inherit the source controller's key and cursor even
   when the HTTP operation ID differs. They do not call a seed initializer.
4. Explicit reseed is a controller-only transition. It changes the stream
   identity and resets the draw cursor; it preserves canonical token history,
   grammar progress, repetition counts, stop matcher progress, output decoder
   state, and any active feedback-sampler state. It must be durable and
   idempotent under an operation ID.
5. Text stops are matched against the UTF-8 bytes of tokenizer pieces, not
   token IDs or decoded character chunks. A match may cross token and HTTP
   chunk boundaries or occur inside one token. The entire selected token is
   consumed into canonical history; generation then stops, while display omits
   the marker and all bytes after it in that token.
6. The stop matcher stores only bounded matching/display state, never a second
   copy of generated history. It withholds the longest ambiguous suffix that
   could still become a stop marker. If a generation is merely paused at its
   work limit, that suffix remains pending for continuation. On EOS, unmatched
   pending bytes become ordinary display text and UTF-8 finalization follows
   the existing decoder contract.
7. Exact replay reconstructs the same displayed prefix from canonical token
   IDs, original stop settings, and saved decoder state. Stop handling never
   deletes or rewrites canonical token IDs.
8. With GBNF enabled, the parser consumes the full selected token as usual;
   text stopping is applied after the token is accepted. A stop marker cannot
   make an otherwise grammar-invalid suffix inside the same token valid.

## First deterministic regression matrix

- New-scope seed behavior: seed 0; repeat same seed/source/policy; distinct
  seed with toy forced variates; controller stores stream identity and cursor.
- Continuation: after a nonzero cursor, continue under a new transport
  operation ID and verify no stream/cursor reset. Fork and service restart
  retain the complete controller.
- Explicit reseed: after nonzero draw progress and active Adaptive-P/Mirostat
  feedback, reseed and prove only the intended RNG fields change; retry the
  same reseed operation and prove it does not reseed twice or move the branch
  twice.
- Stop marker inside one token, across two and three tokens, and after a
  prefix that is also a prefix of a longer marker. Verify exact emitted text,
  stopped status, whole-token canonical consumption, and no subsequent token.
- UTF-8 cases: a multibyte marker split across token pieces; a token that
  contains text before and after the marker; an unrelated incomplete UTF-8
  prefix before a match; EOS while only an unmatched marker prefix is pending.
- Pause/resume: first request ends with an ambiguous stop prefix, second
  request completes the marker; compare with one uninterrupted request.
- Replay and lost acknowledgment: replay the committed request and compare
  exact display, receipt, canonical token IDs, and zero additional model work.
- Sibling isolation and GBNF composition: stop state is branch-local; parser
  state consumes the complete matching token and remains valid at the stopped
  boundary.
- Rejections: empty/invalid UTF-8 stop strings, duplicate or over-limit marker
  sets, malformed seed types/ranges, reseed on idle/stopped scopes, and
  accidental seed application on continue/retry.

## Integration boundary

The first production patch should add a versioned canonical controller
extension and a shared byte-stop matcher used by generation and replay. CLI
and HTTP plumbing should forward structured settings only after that canonical
contract is reviewed. Coordinator/intent/receipt edits are deferred until the
root integration order is set. The public reseed command syntax is not fixed
in this document; a dedicated reseed transition is favored over silently
restarting a continuation.
