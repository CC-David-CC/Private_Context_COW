# Image save on a full filesystem: bounded Q4 result

The actual CLI/HTTP/COW/GPU path preserved the old branch when the canonical
filesystem filled. After releasing space, the original operation published its
already-prepared state exactly once. Two CLI retries and a repeated server retry
returned the same receipt. A fresh CLI process explicitly replayed the saved
answer, including the complete receipt-bound thinking view, without native work.
The retained image root subsequently answered OK and remained unchanged.

The final run used a disposable, journaled64MiB ext4 loop image in a private mount
namespace. Checkpoints and logs stayed on separate storage. Real writes filled
55,541,760 bytes before kernel errno28; zero allocatable bytes remained. The actual
2,211-byte branch-reference write raised ENOSPC. The old reference/controller/
payload and prepared129,608,784-byte checkpoint were unchanged, with no staged
reference files or durable-success acknowledgment. CLI exit2 correctly reported
an indeterminate operation; lookup reported failed and publication_retryable.

Six new harness methods and42 related methods pass; G0 passes51/51. Original
RED failures and all three incomplete GPU attempts are retained. The first lacked
a required filler receipt field; the second assumed JSON stdout for a receipt-only
CLI command; the third attempted a new stale-revision dispatch rather than the
existing saved-answer replay operation. These were harness errors, not evidence
of backend corruption. No production, native, model, tolerance or registered test
changed for this experiment. The final run passed in29.346s, including20.366s
service startup. Both thoughts closed naturally at33 within the512 budget.

Native totals were438 forward tokens:366 initial mixed cells and72 consumed
outputs across the faulted answer and independent retained-root control. Retry and
saved-answer replay left the entire native trace unchanged. File writes totalled
384,906,336 bytes; checkpoint file loads were0. COW copies totalled150,994,944
bytes; peak native snapshot payload capacity was238,116,226 bytes. These are
instrumented native counters, not total process/VRAM peak measurements. No speed
benchmark or independent numerical oracle ran; maximum logit error is unmeasured.

Server154718 and native154724 exited0 and are absent. All private mounts were
removed and the exact loop device detached; no forced process cleanup was needed.
Profile1fcece65 and frozen cow-root-cancel-green-source-v1 remain unchanged.
Full fingerprints, commands, failures and measurements are in the adjacent JSON.

Private backup: box-git:/srv/data/git/private_context_cow.evidence/
20261017-vision-canonical-enospc-v1/evidence.tar.gz. The archive contains1,369 regular members:1,368 inventoried files were individually
hash-verified, and the inventory manifest was parsed and its exact member coverage
checked. The archive hash also covers the manifest. Archive18,524,350 bytes, SHA256
2ae5d3c99b493c67deac18ca5856eda7b3cf58e7a3515c65cb654146e0f3bf9b.
Optional numeric checkpoint payloads are inventoried but excluded; canonical
metadata is retained in the loop-image artifacts. No Windows or GitHub writes.

## Reproduce

Use run_vision_canonical_enospc.py with the pinned runtime/profile, BranchWrite
client and dependency paths. run_vision_enospc_mount.py owns private mounting and
cleanup; run_vision_enospc_user.py delegates to the ordinary user with the profile's
unchanged execution environment. The archived fixture config records all arguments,
image UUID/inode and launch paths. For a new run, create a new owned64MiB ext4 image,
empty mountpoint, separate checkpoint directory, output directory and config;
never reuse an already-populated fixture. Mount through sudo unshare --mount --fork;
serialize the GPU with the established flock. First use --preflight-only, then the
actual campaign under run-command.sh. Infer only as the ordinary user. Do not fill
host storage or relax mount/device guards.

This completes the bounded canonical-full-filesystem subcase within VC08. It does
not complete VC08, VC09 or original G1-G5. Next: integrated corrupt-payload/position
and regeneration-mismatch rejection, publication-stage crashes, allocation failures
and pressure, followed by the declared Q4 8K/restart/performance campaign. Q8 stays
deferred and chat stays off.
