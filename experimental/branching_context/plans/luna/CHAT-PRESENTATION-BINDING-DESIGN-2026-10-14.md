# COW chat presentation binding design

Review-only note for the opt-in ordinary chat integration. This proposal keeps
the COW Store as the only authority for token histories and branch heads. It
does not connect the current SQLite workspace or change serving behavior.

## Findings

The existing workspace is a second conversation store. `serve/chat_workspace.py`
owns SQLite `chats`, `history_nodes`, `branches`, `branch_heads`, `branch_moves`,
and `operation_receipts` (schema near the top of the file). `create_chat`
creates a synthetic SQLite root and a branch immediately, before any message
exists. The `/workspace/*` routes in `serve/server.py` write those records:
`/workspace/generate` claims a request, executes ordinary serving over
`model_messages(...)`, then `finish` appends the assistant message. Fork and
rewind also move SQLite heads. These writes cannot remain active for a COW chat.

The COW root is a content-addressed model state, not a chat identity. In
`store.py`, `create_root` hashes model contract, input IDs, and controller via
`_put_object`; identical initial states can therefore have the same root ID.
`Store.branches(root=...)` groups by that computational root and cannot safely
be used as a chat catalog. A distinct presentation namespace must be explicit
and separate from snapshot identity.

Coordinator idempotency already has useful durability boundaries:

- `_submit` hashes branch, token IDs, step bound, request kind, expected
  revision, and principal, then writes a checksummed `pool-operations/<id>.json`
  intent before dispatch. The intent currently has only `signature` and
  `head`; create intents have `head: null`.
- `_worker` owns all native work, emits the provisional snapshot event, and
  hands prepared bytes to the writer.
- `_publish` writes and validates checkpoint files first, then
  `save_prepared_branch` calls `Store.move`. `Store.move` writes the new head,
  retry receipt, and ordered event in one per-branch reference-file rename.
  A crash after that rename is recovered from the Store receipt by
  `_recover`; the intent only binds the original request.
- `ContextCoordinator.fork` currently requires an acknowledged checkpoint,
  then calls `Store.fork`; neither layer accepts an operation ID, so fork
  retries do not have the same receipt semantics as moves.

`CowTextAdapter` renders the complete message list with the pinned template and
uses `exact_append_suffix` before calling the token adapter. `append_user`
adds only a new-user suffix after a consumed EOS; `resume` submits no prompt
suffix and continues an active controller. `CowPromptEncoder` uses
`Service.encode_prompt` for literal-safe encoding. Generated output remains
token IDs in COW history; `CowTextRequest` decodes a text view from those IDs
and the saved controller. The UI's current `/workspace/generate` path instead
stores original user text and generated text separately in SQLite.

The existing branch viewer can be reused at the presentation level. In
`serve/web/workspace.js`, `openChat`, `select`, `fork`, `rewind`, and
`messageActions` drive the current catalog and graph; draft text and attachments
are kept per branch in the in-memory `drafts` map (`saveDraft`/`select`). The
graph renderer in `workspace-graph.js` is a view over API-returned graph data.
An opt-in COW mode can preserve these rendering and draft behaviors while
switching the data endpoints to COW-backed projections. It must not call
`ChatWorkspace.append`, `claim`, `finish`, `fork`, or `rewind` for those chats.

## Canonical binding

Use a random, immutable `namespace_id` (the UI `chat_id`) for presentation
grouping. It is not derived from `root`, model contract, or token IDs. Two chats
with exactly identical initial COW state may have the same snapshot root while
retaining different namespace IDs and separate branch groups.

Keep namespace metadata outside branch heads and history snapshots. A small
owner-only Store metadata record supports empty chats before the first send:

```json
{
  "schema": 1,
  "namespace_id": "<random 32-hex id>",
  "title": "Chat title",
  "archived": false,
  "created_at": 1792000000.0,
  "receipts": {"<operation id>": {"digest": "<request digest>", "result": {}}},
  "events": [{"type": "create", "operation_id": "<id>"}]
}
```

This record contains no tip, revision, token list, or parent pointer, so it is
not a second history/head. Its atomic create means an empty chat survives
restart without inventing a model state. A SQLite cache may project this
metadata but must be disposable and rebuildable from the Store record. Rename
and archive change only this presentation record and append an idempotent
metadata receipt; they never mutate branch state.

Put the namespace association on the branch's immutable `fork` event, not on a
snapshot. Forking a branch copies the namespace ID and records the display
label/parent branch in the new branch event. This keeps branches from different
chats separate even when their root or tip IDs happen to be equal. The branch
file is already the atomic unit for branch metadata and its event log.

For every text request, bind a typed presentation descriptor to the existing
COW operation receipt. `Store.move` should accept an optional validated JSON
descriptor and include it in the request digest. On commit, persist it inside
that operation's receipt alongside `result`, while the same branch-file rename
publishes the new head. The ordered `move` event already names the operation
ID, so a catalog can join events to receipts without a separate authoritative
message/history table. Reusing an operation ID with altered text or token spans
must fail the digest check. A crash after rename leaves both the branch move
and its presentation descriptor visible; retry returns the same receipt.

Suggested descriptor v1 (fields intentionally reference canonical token spans
rather than duplicating token history):

```json
{
  "schema": 1,
  "namespace_id": "<chat id>",
  "kind": "initial_turn | user_turn | resume | rewind",
  "messages": [{"id": "<stable message id>", "role": "user", "text": "..."}],
  "source": {"snapshot": "<source tip>", "revision": 3, "token_count": 412},
  "input_span": {"start": 412, "count": 17, "token_ids_sha256": "<digest>"},
  "output_span": {"message_id": "<assistant id>", "start": 429, "count": 12},
  "renderer_id": "<template/tokenizer execution identity>",
  "turn_complete": false
}
```

For `resume`, `messages` is empty and `output_span.message_id` reuses the
active assistant message ID, extending the same presentation message. A new
user turn gets a stable user/assistant pair ID in the request intent; the
assistant span is populated from the consumed output at publication. `rewind`
has no input/output spans and records the selected target in the existing move
event/result. `initial_turn` carries the first user message and its initial
prompt token span. Exact user text and renderer identity belong in the
descriptor; generated text can be reconstructed from the canonical output
token span with the pinned tokenizer. Do not regenerate past user text from
tokens, because literal-safe template encoding can be lossy as a display view.

Extend the checksummed coordinator intent with `namespace_id` and the
presentation-input digest. The request signature must cover this value as
well as token IDs, otherwise the same operation ID and token stream could be
replayed with different text or grouped into another chat. For a first turn,
the branch's initial fork event binds the namespace before inference; the
final move receipt makes the user/assistant turn durable. If the process stops
between fork and final move, retry validates the same intent and resumes the
same branch, while the UI can mark the branch as pending until its receipt
exists.

One gap must be closed for actual first-send behavior: current
`CowTextAdapter.create` only creates a prompt root and does not generate an
answer. The current workspace UI creates an empty chat before its first user
turn. The COW opt-in route needs one `start` operation that consumes the initial
rendered prompt, creates/forks the source root with its namespace, runs the
greedy request from that cached root, and moves the branch to the generated
snapshot. It must not expose a prompt-only branch as a completed first turn.
The native worker owns all of these steps; the HTTP handler remains metadata
and streaming only.

## Projection and operation mapping

A COW catalog should return presentation namespaces, including empty ones,
then group Store branches by the recorded namespace ID. Build graph nodes by
walking canonical snapshot ancestry and join each namespace's operation
receipts to its per-branch move events. A snapshot can appear in more than one
namespace; it is a shared immutable computation object, not a unique chat
message. Namespace-specific text and branch labels come from metadata records.

| UI action | Canonical action | Presentation binding |
|---|---|---|
| New empty chat | Create metadata-only namespace, no branch | Durable namespace create receipt; later first send binds branch |
| First send | New `start` coordinator request | Initial fork event binds namespace; final move receipt records user/output spans |
| Continue with new user | `append_user` then coordinator `submit` | Move receipt records exact user text, consumed input span, assistant span |
| Continue active answer | `resume` with empty input suffix | Move receipt extends same assistant message ID with output span |
| Fork here | Durable Store fork with operation ID | New branch inherits namespace and carries fork event/label; retry returns same fork receipt |
| Rewind/clear | Existing Store CAS move | Move receipt records action and selected snapshot; all prior snapshots remain |
| Rename/archive | Namespace metadata-only operation | Checksummed idempotent metadata operation; never moves a model head |

SQLite is acceptable only as a strictly derived projection cache. Every row
must identify the source namespace/branch/revision/receipt digest; readers
rebuild when the canonical receipt set changes, and deleting the cache must
not lose anything. Do not write or trust SQLite `branch_heads`, history nodes,
branch moves, or operation receipts in COW mode. For the first implementation,
serving the projection directly from the owner-only COW metadata and branch
receipts is smaller and avoids invalidation bugs; add SQLite only if measured
catalog reads require it.

## Small first test-first ticket

Implement the storage contract before wiring UI or native workers. Exact owned
files:

- `experimental/branching_context/src/context_time_travel/store.py`
- new `experimental/branching_context/tests/implementation/test_store_presentation.py`

The first patch should add metadata-only namespace create/read operations,
optional namespace binding on branch fork, and optional validated presentation
metadata on `Store.move` receipts. The move request digest covers the metadata;
the existing per-branch reference rename remains the only atomic commit for
head, event, retry receipt, and presentation descriptor. Do not add SQLite
tables or UI routes in this ticket.

Test first with these deterministic cases:

1. Two namespaces created from identical prompt/controller input have distinct
   IDs but share the same content-addressed COW root; neither catalog group
   contains the other's branch.
2. An empty namespace survives Store restart and has no branch/head until a
   first-send branch is created.
3. Forking a branch copies its namespace while sharing the exact snapshot tip.
4. A move receipt binds source revision/tip, result tip, user text, input token
   span/digest, output span, and renderer identity in the same branch record.
5. Same operation ID and same descriptor returns the original receipt; changed
   namespace or text/span under that ID is rejected.
6. Fault after branch rename followed by restart/retry returns the committed
   descriptor exactly once; fault before rename exposes neither move nor
   descriptor.
7. Rewind records the target and namespace without deleting prior metadata or
   snapshots.

The next ticket should carry these fields through coordinator intent, initial
`start`, text `append_user`/`resume`, and fork/rewind APIs, then compare the
resulting COW projection with the existing workspace viewer actions. Only
after that adapter is complete should an opt-in UI route be switched; ordinary
serving remains unchanged and no chat is allowed to write both head stores.
