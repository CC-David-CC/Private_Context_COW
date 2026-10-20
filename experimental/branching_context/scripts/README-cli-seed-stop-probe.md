# Seed + text STOP BranchWrite CLI probe

`run_cli_seed_stop_probe.py` is a bounded candidate acceptance probe. It drives the public BranchWrite `cow_cli.py` process against the normal authenticated `cow_http` service and records the Store receipts, immutable branch heads, persisted controllers, canonical token spans, source module hashes, native worker trace deltas, and process shutdown receipts. It does not write Store history directly.

The workflow prepares one source context, forks sampled branch A and grammar branch B, starts a short seeded sampled SAMPLED grammar turn on A, applies a typed reseed, continues, and verifies exact replay before and after restarting the service on the same Store. On B it stops READ grammar output at `read_file`, checks the marker remains in canonical consumed bytes while hidden from display, then appends a new user turn under TEST grammar and verifies the Store-bound input suffix contains the real text prompt encoder's EOT boundary.

The SAMPLED GBNF has multiple legal first-token choices and enough required continuation tokens to keep the controller active through the initial continuation and a second continuation after service restart. The runner refuses success if either continuation is terminal or produces no tokens. It also checks that the new service process continues the same persisted RNG stream. Budgets are rejected unless both continuations leave the controller active; EOS behavior is unchanged.

Run only from a fully staged runtime and a profile whose runtime closure has been verified. Example command shape:

```text
python -B experimental/branching_context/scripts/run_cli_seed_stop_probe.py \
  --runtime-root /absolute/runtime/source \
  --profile /absolute/profile/profile.json \
  --store /absolute/run/store \
  --run-dir /absolute/run/campaign \
  --question "short bounded question" \
  --dependency-root /absolute/dependencies
```

Use the approved external `run-command.sh` wrapper with a new output directory outside the source tree. The CLI launches the production service/native worker. A CPU FlowModel can exercise test orchestration but is not native or numerical qualification. `status=CLI_SEED_STOP_FLOW_PASS_NUMERICAL_AUDIT_PENDING` deliberately does not claim independent numerical agreement, registered G-gate qualification, or GPU proof beyond the actual service receipts in that run.
