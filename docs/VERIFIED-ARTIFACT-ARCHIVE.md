# Verified artifact copies to owner storage

`python -m tools.pcc_artifact_archive` copies one pinned artifact over the
owner's existing SSH connection. It is an owner tool, separate from model/client
tool permissions. It never deletes or moves the source.

Supply `--source`, `--sha256`, `--host`, `--mount`, `--mount-uuid`,
`--destination` and `--receipt`. The destination must be under the specified
mounted filesystem. Keep the receipt with the experiment's private records.
The archive contains model bytes; publishing a receipt is not authorization to
publish those bytes.

The command hashes the local source, checks the remote filesystem UUID, writes
an exclusive partial file, fsyncs it, and independently reads back its SHA-256.
It publishes the destination and receipt without replacing an existing file.
An already verified matching copy can be reused with a new local receipt.
Changed source metadata or a conflicting, incomplete or corrupt destination
stops the operation. A partial copy is never overwritten automatically.

A timeout or lost connection leaves the source retained. Inspect the original
remote destination, receipt and partial file before further action. A verified
destination with its complete receipt can be checked again. An incomplete
publication requires operator review; it is not permission to delete either
copy or dispatch another training update.

Archiving and retiring a local artifact are separate decisions. Before retirement,
verify that its owner is terminal, no active reader needs it, the remote receipt
still matches the bytes, and the experiment's milestone/continuation policy
allows removal. This command does not claim to perform those retirement checks.

The explicit LAN fixture in `tools/test_artifact_archive_lan.py` tests a real
SSH transfer and independent byte readback, matching-copy reuse, wrong source
hash, wrong drive UUID, existing local receipt, source-symlink refusal and
preservation of an interrupted copy's original partial bytes.
It requires explicit owner host/storage arguments and does not run by default.
No GPU, model inference or training is used by that fixture.
