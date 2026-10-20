# Reusable codebase prefix, then first question

The namespace-bound presentation route now permits `initial_turn`: an idle
codebase prefix, one new user message, a nonempty prompt suffix, and consumed
generated output. This preserves the codebase-only root as the reusable point
for different questions. It is distinct from `answer`, which produces output
from an already prepared complete prompt, and `user_turn`, which appends after
an actual stopped prior answer. No new canonical history/controller format is
introduced; this is a typed presentation annotation around the same append.

The coordinator signs and durably records original text/namespace/message IDs,
then constructs consumed spans from the actual target. Schema2 intents are
opt-in; legacy schema1 request signatures and numerical checkpoints remain
unchanged. The Store catalog returns one namespace's heads and ordered receipts
under a read transaction. It is full annotation projection, not bounded graph
previews. An active first answer can fork and keep its assistant message ID.

`CowTextAdapter.ask` encodes only the first question after an idle closed-frame
context. `CowPromptEncoder.append_initial_user_suffix` requires the qualified
non-thinking profile and an actual im_end/newline input boundary; it adds no
fake EOS or controller stop. Byte-tokenizer parity matches uninterrupted prompt
rendering, including quoted special-looking text. Actual model-tokenizer/GPU
qualification of this new seam remains outstanding.

Red executions on llm-60:

- Core annotation/coordinator suites:38 executed,three errors for unsupported
  initial_turn (`initial-question-red.log`).
- Text admission:11 executed,nine setup errors for unsupported constructor
  callback (`initial-question-text-red.log`).
- Prompt encoder:five executed,one error for missing suffix operation
  (`initial-question-encoder-red.log`).
- New synthetic flow fixture corruption test:three executed,one failure because
  the fixture ignored stored frontier bytes (`presentation-flow-frontier-red.log`).

Logs are under `/home/dflanag3/branching-g3/request-stop-20261014/`.
The fixture now serializes every float, checks exact byte counts, deserializes
the actual frontier and hashes those values. The unchanged negative test passes.
This was a synthetic fixture bug; it is not evidence of a native GPU defect.

Final executed results:179 CPU/filesystem controls,61 supplemental serving/flow
tests,G051; zero failures/skips/xfails. Profiles record executed IDs and source
hashes. Flow tests use genuine controller transitions and the actual coordinator,
Store and adapters around synthetic computation; no native/model qualification
is claimed. Their combined prepared-answer/fork/resume/EOS/new-user/restart
route passed. Their full flow does not yet cover initial_turn; that seam has
separate Store/coordinator/text/encoder component tests.

No HTTP endpoint or browser switch for the new namespace paths has been made.
Next: derive the trusted renderer identity at model-metadata load, expose the
namespace and typed text paths, then test the complete route on Q4 and Q8 with
the independent matching-operator reference. Ordinary chat stays off; private
Git only. Original G1-G5 certification remains unqualified.
