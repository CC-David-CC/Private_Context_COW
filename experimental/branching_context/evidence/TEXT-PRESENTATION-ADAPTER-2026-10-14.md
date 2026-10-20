# Text presentation request admission

The text adapter builds presentation requests from the exact original user
text passed through its renderer/encoder. Optional namespaces use stable
operation-derived message IDs and a caller-pinned renderer digest. `prepare`
consumes the full first prompt including its assistant frame; `answer` uses
the prepared canonical history with no new prompt. `append_user` binds only
the new raw user text; `resume` keeps the prior published assistant ID and
reads the request-bound source snapshot. Earlier generated IDs are never
re-encoded. Full-transcript generation is rejected for namespace-bound chats;
their callers use these explicit paths instead. Default unbound calls still
omit the presentation keyword.

Eight new tests first ran with eight setup errors because the constructor
did not support the renderer identity. The red log is on llm-60 at
`request-stop-20261014/text-presentation-red.log`. Implementation passed those
eight, then two token-adapter forwarding tests were added and passed. The
supplemental suite executed56 tests, no failures/skips/xfails; G051 passed.
These are synthetic control/token fixtures and real socket regressions, not
GPU/model qualification. The namespace storage and coordinator implementation
are separate integration dependencies. Constructor-pinned identity is not yet
automatically derived from model metadata; HTTP/workspace paths and browser
integration remain outstanding.
