# Accidental GitHub branch publication and containment

The owner authorized private PC/box-git work only. A worker violated that boundary
by running `git push origin HEAD:refs/heads/work/luna-cli-machine-result-20261016`
from `fleet/context-cow-cli-machine-result-luna`. Its origin was
`git@github.com:CC-David-CC/Strata-a5500.git`, not private box-git.

Root independently observed that public branch at
`26a0a7cbfce358c7fcd7a8d12c9ecb1ef0302a01`. The tip changes CLI machine-result
output and its tests, and its parent is the private seed/adoption checkpoint
`863bb447093ddd07b77fa377cba567759bf2eb64`. Reachability therefore included ancestor
history; the exposure was not limited to the tip's two changed files. The worker's
retained second push output also reports large historical blobs. The exact newly
uploaded object set and whether anyone fetched it are not established.

Root contained the ref as follows:

1. Verified the intended private remote URL was
   `box-git:/srv/data/git/private_context_cow.git`.
2. Pushed the exact tip to that private repository under the same branch name.
3. Verified private `ls-remote` returned the exact full commit.
4. Rechecked the public URL and exact public ref value.
5. Deleted only that public branch, using an exact expected-value lease so a
   concurrent change would reject deletion.
6. Verified public `ls-remote` returned no matching branch.

Both backup and deletion commands completed successfully. This withdraws the
branch ref; it does not prove deletion of retained GitHub objects, caches or
copies. No claim that confidentiality was fully restored is made.

The original local incident note is retained at
`fleet/context-cow-artifacts/accidental-public-push-20261016.txt`, SHA-256
`899c0558a7c82a274289d788c72c5a77972dc1479278a15632980e082e1c6c60`.
It preserves the reported push command, retry output and observed public ref.

Operational correction: workers now commit locally and hand off exact commits.
Root alone performs Git pushes, after checking the actual destination URL and
explicit destination ref. Authorized Linux test/artifact writes continue; no
further publication is authorized. The owner was informed during containment.
