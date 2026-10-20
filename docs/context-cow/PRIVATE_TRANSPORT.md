# llm-60 to private box-git — verified 2026-10-14

The default SSH route now supports both `box-git` and `git-box`:
llm-60 -> 10.0.7.83, account `dflanag3`.

- A dedicated Ed25519 private key was generated on llm-60 and stays there.
- Only its public key was installed on box-git. Prior authorized keys were
  preserved, with a permissions-preserving backup before the append.
- The new key is restricted to llm-60's observed LAN/Tailscale addresses
  (10.0.7.68 and 100.89.200.66), with SSH's `restrict` option. Shell commands,
  Git and file transfer are allowed; agent/port forwarding and PTY are disabled.
- Host verification remains enabled. The previously trusted host fingerprint
  matches box-git's actual Ed25519 host key:
  `SHA256:bvx5684zHcsZwOy//i6Gw3+6eXHyKX4OT1d7EpzdHhc`.
- Existing explicit `config_codi_boxgit` and `config_codi_recovery` profiles
  remain unchanged. Those use restricted Git/recovery identities, explaining
  why a default `ssh box-git true` did not previously work.

Executed checks: login through both aliases; private branch lookup; shallow
fetch; `git push --dry-run` to the existing private branch. The receive path
reported `Everything up-to-date`; no branch was changed by that check. Existing
repository remotes and GitHub routing were not altered.

## Raw evidence backup

The complete archive is on llm-60 at:
`/home/dflanag3/branching-g3/native-resident-evidence-20261014.zip`.

It was transferred directly from llm-60 to box-git at:
`/srv/data/git/private_context_cow.evidence/20261014-native-resident/native-resident-evidence-20261014.zip`.

Both measured copies are 3,781,330,625 bytes, SHA-256:
`bc040a2e4f1eeebb09ae289d79cb7a90c1c5d776bfd8d82dba53ae4e461fc5e5`.
The archive contains 490 files: raw numerical frontiers, canonical store and
checkpoint evidence, executable/build/profile records and tested source.
It does not bundle model weights or the chat server's secret configuration.

The user questioned the need for a full PC copy. That transfer was stopped;
only the incomplete file at the agent-created output path was removed after
verifying it was shorter than the complete archive. Code, summary results and
the archive receipt remain in local Git and private box-git. No public GitHub
publication occurred.

No additional permission is required to run the authorized serialized GPU
campaign or restore the identified chat server. Remaining work concerns actual
long-context admission/performance, supported sampling, serving integration and
the separate full-certification scope. Test launches do not qualify those.
