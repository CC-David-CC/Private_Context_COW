# Durable empty chats and canonical span validation

Empty chat namespaces now survive Store restart without creating model roots,
numerical checkpoints or branch heads. Distinct creation operation IDs produce
distinct32-hex namespaces, even when later computations share a root. Create,
get and list APIs are owner-only; files0600/directories0700, canonical JSON with
checksum and atomic file/directory durability. A same-input retry returns the
original creation record; changed input conflicts. Namespace11 tests cover
restart, concurrency, corruption/deletion, before/after-rename faults, permissions
and read-only listing. This does not yet implement renaming or archive.

The new `presentation_binding.validate_binding()` checks a typed display record
against actual canonical consumed input/output spans, source revision/root,
controller scope/progress and stopped status. It covers root preparation,
initial answer, new-user turn, same-turn resume and ancestor rewind. It rejects
wrong namespaces, stale sources, altered token-span digests, hidden fields,
unsupported roles and invented output counts. Caller data is detached. It does
not publish a head or a second token history. The frontend must still verify
text encoding against the recorded tokenizer/template; this validator cannot
derive that from a renderer ID alone.

First red was one unittest loader error because the module did not yet exist;
the eight intended methods did not execute in that attempt. Initial implementation
then passed eight. An added adversarial test found bool True could compare equal
to an integer progress count1; the rerun executed nine with one failure, native
SSH command exit1. Requiring an exact nonnegative integer corrected this without
changing the test. All nine now pass. The red log is retained as
PRESENTATION-BINDING-BOOL-RED-2026-10-14.log.

Combined qualification on llm-60's Python3.12.3:135 CPU/filesystem controls
(previous106 plus catalog9, namespaces11 and binding9),46 supplemental serving
controls and G051 passed. No failures/skips/xfails. Machine profiles list actual
executed IDs and source hashes. This is CPU/storage qualification only; frozen
serving-v6 retains earlier GPU/numerical evidence. The new validator is not yet
attached to atomic COW receipts or coordinator intents; ordinary UI remains
unwired. A shape-valid descriptor alone is not a durably published chat message.

Next ticket: versioned namespace association on fork and typed presentation
binding committed with the same atomic branch move/receipt. Then carry the
descriptor through checked coordinator intents, create/generate/resume/new-user
and fork/rewind; project the existing UI from those receipts. Preserve ordinary
default behavior and prevent SQLite from becoming a second execution-head store.
