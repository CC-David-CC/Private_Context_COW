# Numerical publisher presentation bridge

`BranchingModel.save_prepared_branch` accepts an optional presentation descriptor
and forwards it to the same Store move which commits the branch and receipt.
Numerical checkpoint publication, fsync and byte validation still precede that
move. No descriptor means the legacy positional Store call, with no new keyword.
The checkpoint format and returned acknowledgment remain unchanged.

Four new plumbing regressions first executed on llm-60 with three errors:
the keyword was unsupported. The red log remains at
`/home/dflanag3/branching-g3/request-stop-20261014/prepared-presentation-red.log`.
After implementation all four passed: forwarding occurs after saved-byte
validation; legacy calls have no additional keyword; rejected presentation
cannot acknowledge/move the branch; corrupt numerical bytes never reach the
presentation commit. These tests use synthetic native bytes and an intercepted
Store move to isolate the publisher seam. They do not prove actual presentation
storage, a native GPU result, or complete UI integration.

Combined remote results:140 CPU/filesystem controls,46 serving controls and
G051 passed, without skips/xfails. Adjacent JSON profiles include executed IDs
and tested source fingerprints. Actual versioned Store publication is Luna's
separate ticket; its integration must pass real Store/coordinator tests before
chat presentation is qualified. No inference job was launched for this bridge.
