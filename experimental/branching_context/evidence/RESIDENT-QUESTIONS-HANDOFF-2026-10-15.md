# One resident codebase, three independent questions

FOCUSED-PASS on llm-60 GPU0, Q4, current sparse native executable SHA
`4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451`.
No explicit residency-lock API or production API/native changes were added.

Worker: `work/luna-resident-questions`, base `ddf57e7e4e06ddb799340c94f3ebb615a2416c3a`.
Controller worktree: `C:\Users\dflanag3\Documents\fleet\context-cow-resident-questions-luna`.
Root took ownership after Luna froze the harness; it corrected expected-controller
construction with an executed behavioral RED before rerunning the GPU campaign.

One actual CLI/service/native process prefills the 7,231-token codebase once.
Three question branches start at that same immutable root; each question is a
30-token suffix and asks for three short paragraphs. The requested ceiling is
5,000 outputs or EOS. Actual outputs are 320, 317 and 330, all ending at EOS.
This does not demonstrate 5,000 actual generated tokens.

All three requests have zero saved-prefix forward work, zero native checkpoint
file-load bytes and zero integrity mismatches. The same native root capture
handle 0 is restored throughout and is not dropped during the campaign. The
canonical root and inactive siblings remain unchanged. Each complete controller,
exact token ancestry, CLI stdout and saved payload hash is checked. This proves
bounded native KV-image retention; auxiliary recurrent state may reside on the
host and this is not a permanent GPU-residency guarantee.

Durable CLI request times: 3.9163, 3.8699 and 4.1234 seconds, corresponding to
81.71, 81.91 and 80.03 output tokens/second including client startup, suffix
prefill, capture and publication. They exclude model/prefix startup. Total
campaign time is 38.62 seconds. The HTTP and native processes both exit 0.
No new independent numerical oracle or quality qualification was performed.

Final runner SHA: `5868d2239355b7808cf5c6f2eafe20698e40f96541c9751c7a1e741a7d0a59e0`.
Final tests SHA: `89c2345d3454de4b5a07f46e37f339dce4954416daf333dae0230a3dac133cf5`.
CPU final: 10 PASS, no skips. G0 reruns 51/51 on unchanged reference source.

Retained failures: initial absent-runner RED (eight explicit assertion failures),
an initially invalid lineage counterexample corrected before implementation,
and GPU-v1 stopping after its first 320-token answer because the expected EOS
policy was constructed as a Python tuple rather than the canonical JSON list.
The added canonical-list regression failed with that exact tuple, then passed
after the helper used the profile's list. Full controller equality was retained;
there was no native/model/tolerance change. Earlier sibling and UTF-8 harness
review findings were corrected before GPU-v1; assertions remain enforced.

Raw source, stdout, traces and results:
`llm-60:/home/dflanag3/spatial-artifacts/resident-questions-20261015-v1`.
Primary audit SHA: `e6d8d1b1f1438d260653b255991e0dfccc6b4deeb5e70d40040377bffb11b2b5`.
Q4 profile SHA: `f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c`.
Q8 is deferred by the user. Original cumulative gate qualification is unchanged.

Next unrelated integration failure: the Q4 publication-fault campaign stopped
before fault injection because history projection rejects a canonical snapshot
revisited through two resume operations as ambiguous. Preserve that failure and
fix its receipt/branch selection before rerunning the real EFBIG/retry campaign.
