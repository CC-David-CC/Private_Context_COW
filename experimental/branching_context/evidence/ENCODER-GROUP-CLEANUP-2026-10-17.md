# VC08 encoder process-group cleanup: focused PASS

A real subprocess fixture demonstrated a host lifecycle defect: timeout and
cancellation killed only the encoder leader, while a same-group helper survived.
The expanded regression also reproduced the leak after a successful leader exit.
REDv1 records2 subtest failures; REDv2 records3. The exact REDv2 inputs were
reconstructed from base49ac024b and the unchanged executed test; they were not
frozen before that run. Original logs and both versions remain available.

The encoder already starts a private session. Its finally block now terminates
that invocation's process group and reaps a still-live leader. This also runs
before a successful return reaches the caller. Existing timeout/error semantics,
RAM/disk reservations and staging cleanup remain. No native kernel, position,
input, sampler or checkpoint-format code changed. The trusted encoder contract
does not admit hostile descendants escaping that group.

Focused8 methods and related72 unique methods pass, with no failures/errors/skips.
The distinguishing method exercises completion, timeout and cancellation, checks
helper disappearance within2 seconds and verifies zero staging reservations.
These use a real fixture process, not a GPU substitute or a numerical oracle.
G0 passes51/51.

Actual pinned GPU encoder checks on llm-60/GPU0:

- Normal encoding reproduces all3,072,020 bytes of the retained H04 green-page
  SVE payload, SHA27b819e691cd1325011f3487e0b81f7d5e6ed7373785fbd02cb6fa0eae9865ee.
  Rows300, grid20x15, width2560; byte differences0. Process exits0 and is absent;
  staging and reservations empty. Whole diagnostic1.619 seconds.
- Startup cancellation waits until nvidia-smi observes the exact owned native
  PID with14MiB allocated. CancelledError returns, PID is absent, staging empty
  and both pool reservations zero. Whole diagnostic0.832 seconds. This is not
  cancellation after a completed encoding or a language-model worker OOM test.

Fresh-profile H05 CLI/HTTP/COW/GPU integration also passes in60.060 seconds,
including startup and all three stages. All3 generated token-ID sequences equal
the earlier run; each natural thought ends within512. Initial vertical report,
exact column-to-row edit, protected fresh-page checks and horizontal report pass.
Both service/native exits0, absent, no forced cleanup. No full-logit oracle ran.

The changed Python encoder implementation deliberately has a new identity. New
execution profile871115bf2b535fd31ae643284003a965fd123d7edc9e84e8b2645fd09136ceb6
uses cow-encoder-group-green-source-v1 and a fresh store. Do not relabel older
checkpoints or claim old profiles match the changed source closure. The profile,
preflight, exact source hashes and model/build identities are in the raw backup.

Private box-git backup:
/srv/data/git/private_context_cow.evidence/20261017-encoder-group-cleanup-v1/evidence.tar.gz
10,775,949B;245 members; SHA
d388c63ed447537ebc7451d836d12eb093a858086082c750a75855008a1313f1.
Receiver re-read every member and verified size/hash.806,066,892B optional native
checkpoint data is excluded with hashes. Later retirement is recorded separately.
The earlier H04/H05 completed stores already retired1,344,884,788 derived bytes
after their complete canonical/evidence backup; see its private retirement receipt.

This closes one bounded VC08 cleanup defect. Coordinated publication-stage
crashes, actual capacity/OOM pressure, later-stage GPU cancellation, reconstruction
mismatch and the progressive8K/performance/support matrix remain separate work.
No complete VC08/VC09 or cumulative G1-G5 claim. Chat OFF, Q8 deferred, no public
push. All work/archives stay on Linux/private LAN; Windows steers only.
