# Preserve grammar completion when replaying an answer

After a lost response or server restart, PCC could recover the exact answer
tokens and text while dropping `grammar_complete` and `stopped`. BW correctly
refused that answer as a completed constrained tool call. Both PCC replay paths
now reconstruct these fields from the saved canonical controller, exact grammar
identity and accepted-token history, without another model forward pass.

The correction covers coordinator recovery of an original request and explicit
replay of a committed text answer. It does not infer completion from an EOS token
or from text that happens to parse. A grammar may accept a prefix before the
turn stops; a text stop may stop a turn whose grammar is incomplete. The two
flags remain separate. An unavailable saved grammar still causes refusal.

The [qualification receipt](../experimental/branching_context/evidence/runs/replay-grammar-metadata-20261026-cc.json)
records 18 passing targeted CPU checks, including eight completion/restart
controls, plus G0's 58 passing tests. The controls cover complete, incomplete,
accepting-but-active, text-stopped-incomplete and unavailable-grammar cases.
Restart controls compare model-work counters before and after replay. No native
GPU qualification or cumulative G1–G5 claim is made for this patch.

The S³ recovery campaign exposed the coordinator path after the first fix to
the explicit text-retry path passed its local controls. Both failures and their
separate corrections are retained in the test evidence. The default generation
path and ordinary unconstrained replay are unchanged. No CUDA, HIP or SYCL file
was modified, and no native binary was rebuilt.

The existing model/profile and runtime-source checks still apply. Install this
fix before creating a new qualified runtime that needs grammar-aware replay.
Changing source underneath an old saved evaluator is not an implicit migration.
