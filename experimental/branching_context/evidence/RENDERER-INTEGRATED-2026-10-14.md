# Loaded renderer identity and first-question tokenizer check

Renderer helper integrated asa8c8b8d5. The model text loader now computes its
identity once from loaded tokenizer metadata, exact template and kwargs,
qualified model contract, literal-encoding implementation hashes and dependency
versions. It does not hash model weights or reconstruct this identity per token.
Missing metadata/configuration fails instead of guessing from model filenames.
The source/helper scope and actual Q8 metadata evidence are retained in
serve/cow_renderer_evidence.json. During development a quadratic list-membership
check was found by the real-sized metadata run, which was stopped (exit143),
then replaced with set membership before the successful rerun.

The root integration additionally loaded the actual Q8 tokenizer/template via
model_text_adapter with contract
fc6d0d643e4b9ae020964f8c4e5006c0da7199b19cce5285c573023dbe82ca46.
No engine or GPU was constructed. A17-token system context plus25-token first
question exactly matched the full prompt render; quoted im_end and
tool_response text passed the literal-data encoder. Renderer identity with
the new first-question encoder source was
0075a8b96ac71994eab911263040752670dc8f5d132021abe494c2467864b503.
This supersedes the older helper-only identity for this source, not its result.
The exact executed CPU probe remains on llm-60 at
request-stop-20261014/serving-v8/first-question-tokenizer-check.py.
The reusable parameterized probe is scripts/check_cow_first_question_tokenizer.py;
run with --model and --model-contract under the same Linux PYTHONPATH as the
serving controls. It loads tokenizer metadata only, never an inference engine.

Integrated serving suite67 and G051 passed; zero failures/skips/xfails.
Core controls179 passed for the same unchanged core source in the preceding
initial-question slice. Actual model inference, numerical/logit equivalence,
HTTP namespace endpoints and browser integration are not qualified by these
metadata/control tests. Existing real Q8 speed evidence remains scoped to the
earlier frozen serving candidates. Next is HTTP/projection wiring, then the
complete route on Q4 and Q8. No ordinary chat server or GPU job was started.
