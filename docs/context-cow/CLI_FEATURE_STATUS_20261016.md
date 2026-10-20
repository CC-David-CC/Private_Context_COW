# Context COW CLI feature status, 2026-10-16

The private CLI implements ordinary context and branch work, streaming answers,
durable request receipts, sampler controls, and evaluation/adoption commands.
Only G0 is formally PASS. New G4 controls completed a bounded real-GPU CLI
workflow and a separate exact numerical audit. This inventory describes checked-in
code, not a claim that the paused chat server has been deployed or restarted.

| User task | Current CLI |
|---|---|
| Inspect service and operations | `status`, `operation` |
| List saved contexts and branches | `contexts` / `namespaces`, `branches` |
| Read saved message history | `history` through the configured COW workspace |
| Create a context collection | `namespace` |
| Load a context file or prefill a first prompt | `load`, `prepare` |
| Ask, answer, append input, continue | `ask`, `answer`, `append-user`, `continue` |
| Fork at a current or prior snapshot | `fork` |
| Move a branch to an earlier snapshot | `rewind` |
| Return a saved answer or regenerate it with new settings | `retry --from-operation`; CPU and bounded Q4 GPU pipeline passed; independent numerical audit pending |
| Recover the exact pending request | `retry-request`; a completed local entry currently prints its receipt only |
| Retry failed publication | `retry-publication` with the original operation ID |
| Set sampler controls for a generation | Flags on generation commands; combined text-workflow qualification remains incomplete |
| Choose a named server-owned grammar | `--grammar-state` on `ask`, `answer`, `continue` and `append-user`; combined CPU and bounded Q4 CLI pipeline passed; matching-profile numerical audit passed 38 exact comparisons |
| Evaluate alternatives | `evaluate`, `inspect-evaluation`, `select-evaluations` |
| Adopt the server-selected winner | `adopt-evaluation` with an expected branch revision |
| Reuse RAM/disk checkpoints and restore after restart | Automatic server behavior behind ordinary requests |

The current combined CPU source passed 259/259 tests and G0 51/51, with no
failures/errors/skips. The obsolete prepared-presentation spies were reconciled
to the durable checkpoint receipt contract, adding stronger metadata/restart
assertions and a metadata-stripping mutant. Production validation, mathematical
equations and registered acceptance assertions were unchanged. Earlier failures
remain recorded. See
[combined evidence](../../experimental/branching_context/evidence/CLI-GRAMMAR-HARNESS-MERGED-CPU-2026-10-16.json).

The CLI calls the COW service. It loads no model and creates no second canonical
history. The normal server entry point supplies its history workspace; earlier
sampler-only audit fixtures did not cover the complete saved-text workflow.
The ordinary chat server remains off. Isolated test services are separate.

Sampler flags include temperature, Min-P, Top-K, Top-P, DRY, Typical, DynaTemp,
XTC, Mirostat 2 and Adaptive-P. Supported presets have bounded Q4/Q8 sampler and
restart evidence. Invalid combinations reject explicitly. Retry integration
fixed the text presentation validator's rejection of changed sampler settings
on a namespaced `continue`, using an explicit bound generation descriptor.
Current combined CPU tests cover that behavior. The bounded Q4 retry workflow
also passed all 14 CLI commands, including a new-process saved-answer replay;
it did not run an independent numerical oracle.

## Evaluation and adoption

G4 code is integrated at `5f6c6ae7` from private worker commit `28ae4690`.
Twenty new CPU tests passed. The broader run executed 125 tests: 123 passed,
and two existing prepared-presentation exact-kwargs assertions failed. Both
failures were independently reproduced on exact base `5143e447`; no assertion
or checkpoint binding was weakened. G0 passed 51/51. Eight changed source/test
files match the frozen executed Linux stage by SHA-256.

An additional harness candidate passed eight owned CPU tests, including a real
CLI-subprocess path through prepare, fork, answer, isolated scoring, inspection,
selection, adoption, history and continuation. On llm-60 the frozen candidate
then completed all 14 CLI stages with Q4: 181 prompt tokens, eight outputs per
candidate, and eight more after adoption. Native counters show no prefix replay
and no forward work for evaluation, selection or adoption. Both processes exited
zero. The candidates use the same prompt and greedy settings, so their identical
answers are expected; this is a workflow check, not answer-quality validation.
The workflow harness is integrated at `90a9322c`. Its original shutdown receipt leaked the
temporary test key into private evidence. A recursive redaction fix passed its
new regression (29 combined CPU tests) and a separate 14-stage GPU repeat.
The separate numerical audit now has 47 exact full-frontier comparisons,
maximum error zero, matching tokens/controllers, and six cleanly exited fresh
reference processes. It covers this earlier frozen workflow source and its
181-token prefix, two eight-token answers and eight-token continuation, rather
than the later retry/grammar merge or full registered G4 campaign. Restart
qualification of the combined workflow remains pending. See the
[evaluation guide](CLI_EVALUATE_ADOPT.md) and
[CPU evidence](../../experimental/branching_context/evidence/G4-CLI-CPU-2026-10-16.json),
plus the [GPU workflow evidence](../../experimental/branching_context/evidence/G4-CLI-Q4-WORKFLOW-2026-10-16.json).
The [independent numerical evidence](../../experimental/branching_context/evidence/CLI-WORKFLOW-NUMERICS-2026-10-16.json)
records the exact operator boundaries and oracle provenance.

## Still missing or under implementation

- Exact-answer `retry` is merged at `6093d2d4`, with a final evidence update at
  `cfcf8b5b`. Its 11 focused tests are included in the 67-test worker regression
  run, not additional passes. A subsequent lost-acknowledgment fix at `eff2d2b9`
  passed two new recovery cases after their failing runs. The latest pre-grammar
  merged CPU union executed 199 tests: 197 passed and the same two old
  prepared-presentation assertions failed;
  no errors or skips remain. G0 passed 51/51. A separate Q4 retry pipeline passed
  14 CLI commands; independent numerical retry qualification remains pending.
  Unchanged settings replay the saved text/tokens and original receipt without
  moving the branch; changed settings bind a new operation to the original
  source/controller/RNG and check the destination's current revision. Legacy
  receipts without a complete descriptor support unchanged replay only.
  Deep retry-chain performance remains unqualified. See
  [combined evidence](../../experimental/branching_context/evidence/CLI-COMBINED-CPU-2026-10-16.json).
- State-dependent GBNF CLI/HTTP controls are merged at `63a3d30f`. The operator
  enables a fixed registry with `--grammar-registry`; clients send only a named
  state. Worker checks passed nine canonical grammar tests, 45 HTTP/CLI/retry/
  text-adapter tests and G0 51/51 across recorded CPU stages. The full merged
  union first ran 242 tests: 240 passed and the same two old assertions
  failed. The reconciled union passed 253/253; adding the six workflow-harness
  tests passed 259/259 and G0 51/51 on the merged source. The bounded
  Q4 native CLI grammar flow passed eight command outcomes: independent READ
  and TEST branches, partial READ continuation and rejection of an active-state
  switch before native work. Both consumed EOS; native counters showed 22
  prefix tokens and 24 outputs with no prefix replay. The separate numerical
  audit passed 38 exact full-frontier comparisons with zero maximum error and
  controller agreement at the three saved boundaries, using four fresh raw
  reference processes. This covers the earlier bounded workflow; restart of a
  partial grammar CLI flow remains pending. Parser progress stays
  in branch-owned canonical controller state. See the [grammar guide](GRAMMAR.md).
- General explicit seed/reseed and text-stop controls are being implemented on
  separate isolated branches. Seed/reseed integrates first; neither is yet
  qualified or exposed in the integrated CLI.
- Manual strict checkpoint restore, numerical-copy eviction and history-GC
  administration remain library features. Ordinary persistence/reuse is automatic.

The first real-backend CLI workflow ran through an isolated authenticated server
on llm-60, with native work traces and independently checked source pins.
Retry, restart and numerical qualification remain explicit pending stages until
executed. No API-only result substitutes for this workflow.

Historical Adaptive-P Q4 resident API timing is about 92 generation tokens/second
for 8K/512/128. It is not a new CLI or Q8 throughput measurement. See the
[measurement guide](measurements/adaptive-p-20261016/README.md).
