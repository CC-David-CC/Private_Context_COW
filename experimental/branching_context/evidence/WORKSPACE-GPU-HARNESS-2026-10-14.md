# Next real-GPU transport qualification: harness checkpoint

UNQUALIFIED: new namespace-bound GPU path has not executed yet.

Extended existing `scripts/gpu_http_probe.py` rather than adding another runner.
Its text-fixture mode now creates a namespace, prepares the code context, forks
through the namespace-bound HTTP route, submits two distinct first questions,
and reads each answer through the authenticated workspace projection. It checks
exact encoded suffixes, canonical projected text/snapshot identities, streamed
versus durable output, idempotent retries, actual suffix-only forward counters
and unchanged parent bytes. After closing the native worker, independent fresh
matching-schedule processes must match tokens, root/final full vectors and full
greedy controller state. No checkpoint is supplied to those references.

The output bound is configurable with `--max-tokens` (default128); report actual
output counts and natural EOS. Model startup, received TTFT/token gaps and durable
request intervals remain separate. The historical frozen probes/results are not
modified. Current token-only mode remains separate from the namespace text path.

Executed so far: import/argument discovery through `--help` on llm-60,
Python3.12.3, exit0. That is not a numerical, GPU, throughput or milestone pass.
No model process was started. GPU inspection before staging showed zero
utilization and 15MiB used on llm-60; ordinary chat remains off.

Next: integrate fork-provenance and browser work, freeze a complete new source
tree/profile, then run Q4 small, Q4 workload and Q8 validation using executed
tokenizer/template fixtures. Preserve failures and stop expansion on mismatch.
Use96GiB RAM/128GiB disk budgets and96K capacity, never the old quick-iteration
2GiB/12K profile. Private PC/llm-60/box-git only.
