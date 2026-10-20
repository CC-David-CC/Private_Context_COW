# Canonical branch catalog for practical serving

`Store.branches(root=None)` returns a locked, consistent tuple of canonical
BranchHead records. It checks reference integrity, snapshot ancestry/model
contracts and existing read authorization. Restricted principals do not receive
unreadable roots. It does not move branches or acquire native execution leases.

The initial implementation traversed/assembled each branch's history separately.
A failing scale test measured128 canonical object validations for32 heads sharing
four ancestors. The revised query memoizes root/contract facts transiently and
validates each unique ancestor once: measured4 reads. No token concatenation,
additional persistent history or head aliases were added. Nine catalog tests
passed; the unchanged32 logical-store tests and G051 passed on llm-60.

`CowAdapter.branches()` and authenticated `GET /cow/branches?root=<snapshot-id>`
now expose that read-only catalog. The HTTP query permits at most one optional
root and rejects other parameters. The first supplemental run failed the two
new tests (missing adapter method/404); implementation then passed all46 serving
controls. Integration reran catalog9, unchanged pool106 and G051 successfully.
Machine evidence records counts, source hashes, failures and executed IDs. This
is CPU/control qualification only; GPU evidence remains tied to frozen serving-v6.

Next: durable presentation bindings and the existing chat/graph UI. Distinct
chats may intentionally share the same computational root; presentation grouping
must preserve separate chat identities without maintaining another execution
head. Initial messages, new questions and resume output ranges need recoverable
bindings to COW durable receipts. SQLite may cache that projection, but cannot
independently decide branch revisions or input histories in COW mode.

Next-user raw data, checkpoints, frozen serving-v6, native executables and oracle
build/run artifacts are backed up directly to private box-git as
`text-next-user-q8-full-v1.tgz`,7,914,436,247 bytes, verifiedgzip/SHA-256.
TEXT-NEXT-USER-Q8-BACKUP-2026-10-14.json records the archive scope/exclusions.
No GPU inference is running; the chat server stays off. Original cumulative
G1-G5 status remains separate and unqualified.
