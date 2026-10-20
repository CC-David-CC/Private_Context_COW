# Matching-batch durable-resume implementation

Private branch: `feat/batched-durable-resume` in
`C:\Users\dflanag3\Documents\fleet\context-cow-execution-parity`.
Backup destination: `box-git:/srv/data/git/private_context_cow.git` only.

Implemented successful-native-request receipts, absolute chunk boundaries,
token-range hashes, schema-2 checkpoint binding and requested-schedule rejection.
The fresh native reference accepts explicit request lengths, writes request and
generated frontiers, and records completed chunks. It shares neither mutable
state nor tested checkpoints. Generation consumes selected tokens through Prefill.

Executed control evidence:
- Windows Python 3.13.15: initial reference test failed with missing function;
  receipt test failed with missing module after isolating the POSIX-only package
  import. Both pure tests subsequently passed, one method each.
- llm-60 Python 3.12.3: 13 supplemental recovery methods PASS, 0 failures/skips.
- llm-60 G0: 51 methods PASS, 0 failures/skips; receipt
  `api/evidence/runs/20261014T034640870296Z-G0.json` in the v1 staging directory.
- Native matching reference and receipt-instrumented branch binary compiled.
  Fresh SHA256: `5458342aebc6acbe912e35a6bb4daca9db58cfd96b876c9f10bdb0b938365fa4`.
  Branch SHA256: `19f8e23a00122c26b2e2335495d58bd29c0f8c7bf44941eb6987ba2374ec8002`.

Runtime campaign: `/home/dflanag3/branching-g3/matching-resume-20261014-v2`.
Profile: `ba8e3cf161b4db63155c01b87ccbae28a315e50a211eaf15c883bf32edc880e3`.
Q8 model identities are inherited through 11 verified model-file receipts;
zero model bytes rehashed in this profile-preparation process. Source and binary
hashes are pinned independently. Staging v1 and its profile remain unmodified;
the additional reference-receipt audit is pinned in v2.

Tiny campaign launched: 128 prefix, A/B four-token suffixes, up to eight outputs.
GPU numerical/work/timing results are pending, not zero by assumption.
The campaign stops on its first failure and restarts the authorized chat server.
Medium and target campaigns are not started. Registered G1–G5 are unqualified.
