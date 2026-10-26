# One PCC checkout for the learning loop

The `feat/learning-runtime-unified-20261026` branch brings PCC's native learning
API and execution fixes into one source tree. It merges the learning/completion
recovery line at `4a536ab6e26a7d97f200452479a201b414ce55fb` with the explicit
Lark and grammar replay line at `56a80cb45e1a15cf7581906867524a9f8e2c93ad`.
Their common ancestor is `98a46fc37f5d06a0d3e92829c020356089a00671`.
The merge required no conflict resolution or additional runtime implementation.

This removes a deployment complication in the earlier research campaign, which
imported the HTTP/learning API from one immutable snapshot and context execution
from another. New installations can point both paths to the same PCC checkout.
It does not rewrite an old evaluator, profile, controller or failed receipt.

## What each project owns

- PCC owns inference, COW contexts, replay receipts, native backprop, immutable
  candidates, optimizer state and native learning-job recovery.
- BW owns the actual workflow interpreter, tools, grants, ledger, completed-turn
  lesson capture and read-only budget-failure archives.
- S³ owns skill definitions, protected assessments, independently verified
  practice, learning/forgetting reports and the owner-configured learning policy.

Use this PCC branch with BW's
[`fix/capture-completed-chunks-20261026`](https://github.com/CC-David-CC/BranchWrite/tree/fix/capture-completed-chunks-20261026)
and S³'s
[`feat/learning-stack-20261026`](https://github.com/CC-David-CC/sss/tree/feat/learning-stack-20261026).
Their [workflow recovery guide](https://github.com/CC-David-CC/sss/blob/feat/learning-stack-20261026/docs/WORKFLOW-RECOVERY.md)
explains explicit original-ID recovery, filesystem admission and the distinction
between a bounded failed assessment and a verified lesson.

## Qualification and the next native run

The [CPU receipt](../validation/unified-learning-runtime-20261026/qualification.json)
records 108 passing checks across native completion recovery, optimizer-policy
and lesson-mask contracts, Lark/replay metadata, BW capture, S³ recovery and
budget-failure settlement. G0 separately passed 58 reference checks. Scripted
proposals exercised real HTTP, filesystem and workflow-ledger boundaries; they
are not model-generated proposals or evidence of improved model capability.
No native GPU job, model training or binary rebuild was used for this merge.

Before the next native learning round:

1. Preserve the selected Git revisions and construct a new source-bound PCC
   profile. Use the same unified checkout for the API root and execution root.
2. Observe the actual model, native-library hashes, device, capacity and free
   disk/RAM/VRAM. Source merging does not qualify a GPU build or reserve storage.
3. Qualify the new profile on the intended native backend and verify that parent
   and child evaluator identities agree. Keep its GPU qualification pending
   until that work passes.
4. Register a new learning policy/baseline when its sources, practice membership
   or runtime change. Keep the frozen 68-update study and its questions intact.
5. Use newly owned, independently verified practice. An unfinished protected
   assessment trace remains ineligible for training; it describes a failure
   class from which new exercises can be designed.

Historical layered runs remain reproducible with their original roots. The
unified branch is an integration checkpoint, not a claim that the complete
joined native learning loop has been qualified or that either research
candidate has been promoted.

## Native follow-up and storage lifecycle

The [native follow-up receipt](../validation/unified-learning-runtime-20261026/native-followup.json)
records a bounded native COW smoke check and recovery through the exact owner CLI
after deliberately killing a new owned evaluator child. The same original
request ID was explicitly replayed; three completed operations were preserved.
The one-step continuation left the coding task unfinished, with an independent
artifact failure. This qualifies recovery mechanics, not a learned capability or
a complete joined learning round. No new training or weight publication occurred.

The [closed-store retention command](CLOSED-STORE-RETENTION.md) provides bounded
maintenance beside existing experiment workers. It has separate CPU/filesystem
qualification and does not modify the selected native execution files.
