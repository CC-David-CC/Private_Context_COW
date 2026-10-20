# Thinking image context/questions: bounded Q4 PASS

A saved screenshot can now receive a new text question with thinking enabled.
Two branches keep their shared beginning. A completed answer can receive another
question. After closing the server and model, the same saved beginning can be
loaded from disk and questioned again without rereading embeddings or prefilling
its370 cells. This test checks continuation mechanics; it does not score thoughts.

Production e01aa264 uses the new renderer/profile0c0bc546, unchanged native/model.
Actual CLI context --parts-file creates70 text/control+300 image rows (20x15).
First A:28 new prompt+65 output forwards; first B:42+98; completed-turn followup:
25+107; cold A-equivalent question:28+65. Zero saved-prefix forwards in all four.
Thoughts end naturally at62/95/104/62, below512. EOS is selected and consumed.
Cold restore loads127779792 bytes, with no configured encoder or remaining
Store embedding files. Source controller/history and parent payload remain exact.

Original request replay before and after restart returns saved tokens with0
new native jobs. Three separate checkpoint-free references force the observed
output tokens under matching prefix/suffix batches. Seven full248320-float
comparisons (prefix/final and cold-vs-resident) are byte-identical, maximum error0.
Both server/native pairs and three references exit0; all7 PIDs are absent.
This is not an independent constrained-token selection or every-frontier oracle.

CPU:416 serving methods,12 actual GGUF-tokenizer cases,4 new witness-validator
methods and G0 51 pass. Counts overlap and must not be summed as gate methods.
Initial missing-observer packaging failure and BranchWrite's plain-text-only
load_context bridge rejection remain preserved. The actual mixed context CLI
route is supported; the mixed-image BranchWrite bridge itself remains work.

Private archive64520784 bytes/379 members,378 data files individually verified.
SHA256b50375ccb130f642e7ffa84f306bb55ce032494f4fba55fa09249747cb277aa5.
Destination:box-git:/srv/data/git/private_context_cow.evidence/20261017-thinking-suffix-flow-v1/evidence.tar.gz.
Optional state payloads527001192 logical bytes retired after verification,
recovering527015936 physical bytes. Canonical/originals/embeddings, full vectors,
exact runtime closure/profile/harness, failures and receipts retained.

Next: measured cold encode, embedding-only reuse, resident prefix, RAM/disk
restoration with warm-up+3 repetitions and same-profile text control. The test
runner's observer can overwrite its incremental result with probe status between
phases; terminal result and raw jobs are intact. Separate progress/flow files in
the next harness revision. Image-aware DRY+XTC remains deferred. Original G1-G5,
full VC09, Q8 and split GPU image inference are unqualified. Private LAN only.
