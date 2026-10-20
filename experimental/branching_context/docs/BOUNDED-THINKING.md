# Bounded thinking before a constrained answer

## Current question and restart support (2026-10-17)

You can save an image context, fork it, and ask each branch a different text
question while thinking is enabled. A completed answer can accept another text
question. The tested cold restart loads saved computation without running the
encoder or computing the saved prefix again. Keep the same source-bound profile;
old checkpoints from another renderer are incompatible.

Use the existing `context --parts-file` command for mixed text/image roots,
then `fork`, `ask --grammar-state REPLY --max-tokens 1024 --temperature 0` and
`append-user` after the prior turn has consumed EOS. The server registry owns
REPLY's immutable512-token thinking budget and final grammar. This is not a
per-token budget switch. Incomplete answers use `continue`, not `append-user`.
`retry-request` with the original operation ID returns the original saved answer.

The bounded370-cell Q4 flow passed with exact prefix/final full-logit references
and a producer-free restart. See evidence/THINKING-SUFFIX-FLOW-Q4-2026-10-17.json.
The reference forces observed tokens; final grammar is OK and thought quality is
not scored. New-image suffix under thinking and the broader controller matrix
remain unqualified. BranchWrite's `load_context` CLI bridge still accepts only
plain-text message files; call the mixed COW context command for image roots.
Older observations below retain their historical source/profile scope.


Current operating policy: new task runs start512 thinking tokens. Increase
only when the limit was reached AND the task failed. Stop on natural closure or
success; no new no-thinking/128/256 comparison unless requested. The earlier
experiments below remain historical evidence.

The final HTTP completion now adds answer.thinking for a thinking scope:
schema1, snapshot, progress, complete, reasoning_text and final_text. These are
derived views of the canonical consumed tokens/controller, never new authority.
A partial scope has complete=false and final_text=null, even when partial output
looks like valid JSON. Only the durable terminal completion supplies final_text.
Do not execute streamed/provisional text. Verify the snapshot equals the durable
receipt, then independently validate the final action and host permissions.
A literal closing-tag string in reasoning or final JSON is not a text delimiter:
the server uses canonical token/grammar progress, preserving literal final tags.
Same-operation committed replay derives the same view without native work.

Actual Q4 CLI/HTTP/GPU checked both partial and complete views against canonical
records at512: natural358, LEFT. CPU5 focused/102 related, G0 51 pass. See
THINKING-FINAL-VIEW-Q4-2026-10-17.json. General BranchWrite task consumption is
next; this does not itself complete the H04/H05 loop. Native/model/matcher and
sampler unchanged; no new independent full-vector oracle or Q8 run. Work is now
performed on Linux; Windows drives SSH/transport only.


Latest selective escalation (Q4,2026-10-17): the previously failing short greedy
question answers LEFT at256, but still hits the thinking budget. At512 it closes
naturally after358 thought tokens and answers LEFT. Stop at natural closure,
whether the final answer is correct or wrong; do not keep increasing until a win.
Budgets1024+ were not executed. Previously passing expanded and sampled cases
were not repeated. The earlier128 failure remains recorded. See
evidence/THINKING-NATURAL-END-Q4-2026-10-17.json.

The first real Q4 CLI/HTTP/GPU probe passed on 2026-10-17. The model can think,
close early, then answer under an existing GBNF or JSON Schema. Fork a saved
thought to explore a different continuation. Thinking does not grant permission
to run tools or prove an answer correct.

This initial implementation selects a server-owned grammar state. There is no
general `thinking_length` request option yet. Example registry entry:

```json
{"schema":1,"states":{"THINK128":{"format":"thinking","source":{"schema":1,"max_tokens":128,"final":{"format":"gbnf","source":"root ::= \"LEFT\" | \"RIGHT\" | \"NOT_VISIBLE\""}}}}}
```

Select `--grammar-state THINK128` through the existing CLI answer/resume
commands. A generation's ordinary maximum includes reasoning, closing, final
answer and EOS tokens. If that request limit is reached first, the published
active snapshot can be resumed with the same grammar state. It is not a complete
final answer. Budgets are integers 0..32768 in immutable host definitions, not
instructions in user text. A different budget changes the grammar identity;
an active scope cannot silently change to another definition.

Use a newly pinned profile with `text_template_kwargs.enable_thinking=true`.
Prepare a fresh root under that renderer. Never retag a non-thinking checkpoint.
The current model adds its own reasoning instruction through the existing
renderer. Even a zero-budget control therefore has a different prompt from the
old non-thinking profile; do not attribute every quality difference to reasoning.

The matcher has a simple reasoning/final state machine. Reasoning permits valid
UTF-8 model text. Only the verified dedicated `</think>` token changes phase.
This Q4 tokenizer represents it as GGUF user-defined type4, ID248069; it is not
an EOS. No assumption that all model delimiters have this representation is made.
Other tokenizers without one dedicated control/user-defined delimiter reject.
At the thinking limit the legal mask permits the closing token. It is selected
and consumed normally, not inserted into history. Closing overhead is counted
separately. Final selection uses the ordinary compiled GBNF/JSON matcher and
existing sampler. No fresh sampler is created at that transition.

Phase and counters are derived by replaying canonical accepted tokens under
the immutable definition. The same controller retains parser identity, scope,
UTF-8 pending state, repetition history and RNG/adaptive fields. There is no
second transcript or separately authoritative phase file. Forked sessions own
their progress. The last reasoning token cannot strand an incomplete UTF-8
codepoint; natural closure is unavailable while one is pending.

The bounded probe `scripts/run_thinking_cli_probe.py` uses BranchWrite's real
client, CLI, authenticated HTTP service and GPU. It retains both image questions,
compares 0/128 budgets, pauses after four output tokens, forks, and resumes greedy
and Min-P siblings. Root checkpoint bytes and partial controllers are audited.
Expanded: greedy72 thinking tokens, Min-P88, both LEFT, natural closure.
Short: greedy128 NOT_VISIBLE (quality FAIL), Min-P128 LEFT (one quality PASS).
The short zero-budget control remains NOT_VISIBLE. These are observations, not
an accuracy study. The legal labels and expected LEFT remain unchanged.
Read-only inspection found both short-case thoughts describe the square as
upper-left; the greedy final contradicts its own text. Both hit the budget
mid-sentence. The failure is not evidence that greedy failed to see the square.
Final-label scores were not captured, so the cause is unresolved. The sampled
arm uses temperature0.7, Min-P0.05, Top-K60 and Top-P0.95 together; it does not
isolate Min-P's effect. Varying seeds/budgets and auditing final legal scores
remained a follow-up to that initial run. The selective256/512 follow-up above
addresses this one greedy case; seed sweeps and final-label scores remain unrun.

The diagnostic runner now accepts --questions, --budgets, --greedy-only and
--until-natural-end. Its defaults preserve the original0/128 two-question,
greedy/sampled campaign. For a failing greedy case, select only that question,
strictly increasing budgets and natural-end mode. Each budget forks the same
prepared root into a new immutable grammar scope; it never changes an active
scope's budget. A naturally closed wrong answer stops escalation too. If the
largest explicitly requested budget is exhausted without natural closure, the
run reports failure to reach natural closure rather than claiming success.
This is diagnostic selection, not a new general serving thinking_length option.

The new runner/parser checks pass13 focused/74 affected CPU checks; G0 51/51.
One initial G0 launcher failed publishing its report into the frozen source;
--no-write corrected the launcher, with the original nonzero receipt retained.
The first RED import failure is also retained. Only the diagnostic script/test
changed; the native engine, matcher, renderer and sampler are unchanged. The
new profile binds this updated source closure. No independent logits oracle
or Q8 ran. Both service and native process exited0; chat remains OFF.

Executed CPU suite69/69 and G0 51/51, zero failures/errors/skips. No independent
full-logit oracle, peak-memory measurement, isolated speed benchmark or Q8 run
was performed. Numerical max error is unmeasured. Native engine is unchanged.
See evidence/THINKING-CLI-Q4-2026-10-17.json for exact results and fingerprints.

Still outstanding: general per-generation/per-state budget controls; automatic
READ/EDIT/TEST scheduling and insight retention in BranchWrite; larger budget
sweeps beyond the selective256/512 case; DRY/XTC/Typical/DynaTemp/Mirostat/Adaptive combinations on real GPU;
reseed/retry/cancellation and process-restart thinking campaigns; broader new-image turn
appends under thinking (plain-text new questions now pass as described above); Q8 end-to-end;
original VC/G qualification. Existing non-thinking behavior remains the default.
