# Branch-aware presentation history

This database is the canonical **presentation history** for the web application.
It is not a second native token snapshot representation. The native immutable
token store, controllers and derived KV checkpoints remain separate experimental
components. The current serving adapter reconstructs a request from the selected
message path and explicitly reports `ordinary-serving-history-replay`.

## Relations and normalization

`chats` holds chat metadata. `branches` holds branch metadata, with a unique name
within each chat. `branch_heads` holds the movable tip and monotonic revision.
`chat_roots` associates a chat with one immutable root. `history_nodes` stores
only a parent reference and a message reference: it does not duplicate a chat ID
that could be derived from its parent. Forking creates a branch reference without
copying the prefix messages.

`messages` holds scalar message fields. Independent ordered attachments and tool
records use separate `message_assets` and `message_tools` relations. Neither
relation contains the other set; no attachment-by-tool Cartesian product is
stored. Each has the candidate key `(message_id, ordinal)`. Assets and tool records
are content addressed. A tool's protocol payload is an opaque JSON value, not a
relationally queryable set of arguments. The same applies to request settings and
operation receipts. Under that atomic-value boundary, the declared functional
dependencies have candidate keys as determinants; independent multivalued sets
are decomposed into their own relations. Arbitrary JSON protocol values are not
claimed to be internally normalized into SQL columns.

`branch_moves` retains old and new tips keyed by branch and revision. SQL triggers
refuse UPDATE and DELETE on immutable history relations. Metadata, branch heads
and request status remain mutable. There is no destructive history garbage
collection in this UI version. Clear means moving the head to its root.

## Publication and recovery

Mutations use `BEGIN IMMEDIATE`, foreign keys, WAL and FULL synchronous mode.
Compare-and-swap revisions reject stale writes and ABA reuse. Operation receipts
make append/fork/rewind retries idempotent. One server holds a lifetime file lock
for each database; a second server cannot mark its live requests interrupted.
Startup marks unfinished presentation requests interrupted while keeping their
committed user input. A completed response is committed before the final stream
receipt is yielded. Provisional streamed text is not a native consumed-frontier
checkpoint or a saved RNG/controller state.

The browser keeps a rendering projection and selected branch ID. Existing local
browser history is imported once with retry receipts; the legacy browser copy is
retained. Historical tool results are replayed as recorded results, rather than
running old tools again. Normal API authentication and origin checks apply.

## Limits

Native COW freeze/restore handles are not wired into ordinary web serving. UI
tests with a scripted model verify storage and routing, not a numerical oracle.
Formal G1-G5 status is unchanged. There is no numerical snapshot or prefix-sharing
performance claim associated with this interface. Backups, quotas, retention,
multi-user ownership and native adapter qualification remain separate work.
