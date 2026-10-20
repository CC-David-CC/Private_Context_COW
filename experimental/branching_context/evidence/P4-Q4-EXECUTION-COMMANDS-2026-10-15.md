# P4 Q4 phased execution commands — prepared, not run

These are staged commands for the reviewed P4 runner, gated on root
authorization, restoration of all three mapped root payloads, and fresh
resource preflights. **Do not run while the machine-readable prep status is
blocked.** Each arm starts one coordinator and never repeats the 65,536-token
prefix construction. Archive the disk arm and reclaim only its verified local
checkpoint payloads before running resident; retain the complete disk-arm
archive.

Root imported worker checkpoint c3ad1da4 and corrected the helper invocation
to use `bash`: the verified remote run-command.sh is mode0644 and is not
directly executable. No run is implied by this command preparation.

The fixture hashes identify two different stages: raw source
`fixtures/q4_64k.json` is `9fb327…`; the pinned pure normalizer produces the
P5 campaign’s normalized `fixture.json` at `c66108…`. All normalized
components match the archived/live campaign fixture. The commands bind the
normalized file used by the runner and also verify the raw fixture and
normalizer source.

The seed `p4-q4-arm-order-draw-2026-10-15-01` hashes to
`92c46e55…`; its first hex digit is `9`, so the recorded parity draw chose
`disk,resident`. Both arm invocations receive that same order and seed. The
disk arm is a controlled exact-key pool removal followed by an observed disk
preload; it is not a natural-pressure eviction claim.

Before restoring the source payloads, run this read-only disk-capacity check
and record its output with the restore receipt. It gates restoration plus the
disk arm, which runs only after the verified restore.

```bash
set -euo pipefail
PRE_RESTORE_REQUIRED_FREE_BYTES=43688258416
free_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d '[:space:]')
printf 'pre_restore_free_bytes=%s required=%s\n' "$free_bytes" "$PRE_RESTORE_REQUIRED_FREE_BYTES"
test "$free_bytes" -ge "$PRE_RESTORE_REQUIRED_FREE_BYTES"
```

```bash
set -euo pipefail
umask 077

STAGE=/home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context
Q4=/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure
RAW_FIXTURE=/home/dflanag3/branching-g3/multi-code-contexts-20261014/fixtures/q4_64k.json
NORMALIZER=/home/dflanag3/branching-g3/p5-q4-root24b4-source-20261015/experimental/branching_context/scripts/build_codebase_fixture.py
RUN_ROOT=/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01
RUN_COMMAND=/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
REQUIRED_FREE_BYTES_AFTER_RESTORE=38196451720
MIN_AVAILABLE_BYTES=17179869184
export PYTHONPATH="$STAGE/src:$STAGE/scripts"

# This phase runs the disk arm only, after root-authorized, hash-verified
# payload restoration. RUN_ROOT is separate from output and receipt paths.
test ! -e "$RUN_ROOT"
test -f "$RAW_FIXTURE"
test -f "$NORMALIZER"
test -d "$Q4/store/checkpoints"

# Bind the exact raw-to-normalized fixture derivation, source receipt, profile,
# selected reference outputs/schedules, and historical runner bytes.
sha256sum -c - <<'SHA256'
9fb32731440a794234eae1f622b34415d6a80c433df79f30aea6c37d0f9cb5c2  /home/dflanag3/branching-g3/multi-code-contexts-20261014/fixtures/q4_64k.json
3d4403bc03d539c46dc4db3a50b778b26e1a0769ae3f5884f1558825afc914a8  /home/dflanag3/branching-g3/p5-q4-root24b4-source-20261015/experimental/branching_context/scripts/build_codebase_fixture.py
40965f747ebda3fc87eb61cd274c96e2847d46dba65d346eef99e6d25c3d1cfa  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/result.json
784978eb9e290b1ba02c2bbdd852e495189619b0ddb7811f17559d339522fc5a  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/prepare.json
f3c69ad908786548cdd325f33af03a623553134d9406483e7622edd62135ea4b  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/profile.json
c661084b390720c1cbb1ca2465734cd5a099d366f2811704b4661251f5ca1903  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fixture.json
47aa637d0bfc996406b04d8e00744e50f35501ee5e98f975a50207cec5af1d6c  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-0.stdout
947e692e37a056be63ac8ae24a6101a376b4f67239b295b2f2b05c794e2ce0a1  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-2.stdout
7280b2fe26543464122c546b6d07f9e1541adfeefd5512fe09943fe7df99cf98  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-4.stdout
f00617384145e752143c7acb3c6e11c2a786b9f9db8ffcf0a17665fbdd90cddc  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-0.f32.schedule.json
1c54909490f8268378ee0c7b283b2485e5efdefb08e6df302c334acfca1e127e  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-2.f32.schedule.json
07bb49a8941ac6d77c2801e1f2fed373d70b4fbd5642cfb7180a3f58d58039bd  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-4.f32.schedule.json
c3bdb0d34f88cfd778b0cbd58ee0a4d07354da18d9001ab8e397f3cf19205314  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-0.tokens
946f4b0ebf4cb045d5386e21bf3dd54ef6f844f8b3c97647f2dcbd30b9d41841  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-2.tokens
7d686165e4e99ec0d766d71db227d2f47da88bd97d2412b1f351429944547043  /home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure/fresh-tree-4.tokens
f3e1898ab70fb9125216fb477d756a0ca7cd8a543c55cdab7597ae816b4f8fc9  /home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context/scripts/p4_return_latency.py
33f341c5dc48097cc714c5b4ef1e38b8111509f78a3e69ab553ea0d459c55698  /home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context/scripts/test_p4_return_latency.py
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
SHA256

# The three source checkpoint payloads must first be restored separately from
# the verified archive. Fail closed here if any payload is missing or changed.
sha256sum -c - <<SHA256
cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca  $Q4/store/checkpoints/cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca.state
9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7  $Q4/store/checkpoints/9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7.state
43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7  $Q4/store/checkpoints/43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7.state
07a8d31e1d61e71e25f15322ea5ddd8608e9244a242863ad49e24ce389849a69  $Q4/store/checkpoints/e3ecceb2ade807fdf33d87e66d9f5f4c4e901ae6a5c14ec49b36ca11ee1e5286.json
e897d52f34c2cd445635e7c7002455ac8d93eeef7b581881d1b0de43b93cc9f2  $Q4/store/checkpoints/c4386e8b4c4ffe8793ab0dfc34e899b52e7185b74114c455268b22b9bf84b316.json
5c725a23cafa398c40c24e202cf6b8d23cee900d9f77bccb50b85b2e0605e8e4  $Q4/store/checkpoints/bb974fb5a77c22a4c4ee3e9067a404fa937b70fee53cc469c54273aa02fc584a.json
SHA256
test "$(stat -c %s "$Q4/store/checkpoints/cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca.state")" -eq 1830602232
test "$(stat -c %s "$Q4/store/checkpoints/9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7.state")" -eq 1830602232
test "$(stat -c %s "$Q4/store/checkpoints/43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7.state")" -eq 1830602232

check_sequence_resources() {
  # Gate the current phase's single-arm envelope before its coordinator.
  free_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d '[:space:]')
  test "$free_bytes" -ge "$REQUIRED_FREE_BYTES_AFTER_RESTORE"
  available_kb=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
  test "$((available_kb * 1024))" -ge "$MIN_AVAILABLE_BYTES"
  gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
  test "$gpu_uuid" = "GPU-2d2b1358-9abf-ef75-fc3b-892135ea6e58"
  nvidia-smi -i 0 --query-gpu=uuid,memory.free,memory.total --format=csv,noheader
}

mkdir -m 700 -p "$RUN_ROOT/disk-parent"
test ! -e "$RUN_ROOT/command-receipts/disk"
test ! -e "$RUN_ROOT/command-receipts/resident"

# Seeded order draw: disk first. The verified command helper writes exact argv
# and numeric exits under a receipt directory separate from --output.
check_sequence_resources
bash "$RUN_COMMAND" "$STAGE" "$RUN_ROOT/command-receipts/disk" \
  python3 "$STAGE/scripts/p4_return_latency.py" \
  --source-store "$Q4/store" \
  --source-result "$Q4/result.json" \
  --prepare "$Q4/prepare.json" \
  --profile "$Q4/profile.json" \
  --fixture "$Q4/fixture.json" \
  --reference-dir "$Q4" \
  --output "$RUN_ROOT/disk-parent/output" \
  --arm disk \
  --arm-order disk,resident \
  --order-seed p4-q4-arm-order-draw-2026-10-15-01 \
  --ram-bytes 103079215104 \
  --disk-bytes 137438953472 \
  --min-available-bytes 17179869184

# Verify frozen source payload bytes again after the disk arm.
sha256sum -c - <<SHA256
cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca  $Q4/store/checkpoints/cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca.state
9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7  $Q4/store/checkpoints/9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7.state
43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7  $Q4/store/checkpoints/43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7.state
07a8d31e1d61e71e25f15322ea5ddd8608e9244a242863ad49e24ce389849a69  $Q4/store/checkpoints/e3ecceb2ade807fdf33d87e66d9f5f4c4e901ae6a5c14ec49b36ca11ee1e5286.json
e897d52f34c2cd445635e7c7002455ac8d93eeef7b581881d1b0de43b93cc9f2  $Q4/store/checkpoints/c4386e8b4c4ffe8793ab0dfc34e899b52e7185b74114c455268b22b9bf84b316.json
5c725a23cafa398c40c24e202cf6b8d23cee900d9f77bccb50b85b2e0605e8e4  $Q4/store/checkpoints/bb974fb5a77c22a4c4ee3e9067a404fa937b70fee53cc469c54273aa02fc584a.json
SHA256
```

## Archive disk-arm evidence and reclaim its local payloads

Do not start the resident arm yet. Archive the complete
`$RUN_ROOT/disk-parent/output` tree to the private box-git target below.
Generate a sorted SHA-256 manifest for every file in the output tree, include
it in the archive, and record the archive SHA-256 and member listing. Verify
the archived files against that manifest on box-git before reclaiming anything.
The archive is on a different filesystem with more than 680 GB free. Archive
transfer and verification are outside both latency arms.

Private archive target:
`box-git:/srv/data/git/private_context_cow.evidence/20261015-p4-q4-disk-return/p4-q4-disk-return-20261015.tar`

After successful archive verification, reclaim only the exact local checkpoint
payload paths listed in the verified archive manifest under
`$RUN_ROOT/disk-parent/output/store/checkpoints/`. Keep the archive,
`result.json`, all checkpoint manifests, both command receipts, and every file
under `$Q4/store` intact. Record the exact reclaimed paths and byte count. Do
not use a broad recursive delete. If archive/member verification or exact path
identification fails, stop without reclaiming and do not run resident.

## Resident arm after verified archive and reclamation

Run this second phase only after the disk archive and payload manifest verify,
the exact listed local checkpoint payloads have been reclaimed, and the three
source payloads still match their frozen hashes. It rechecks the single-arm
disk envelope, host memory, and GPU identity. The external archive and disk
arm command receipt preserve the complete first-arm evidence.

```bash
set -euo pipefail
umask 077
STAGE=/home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context
Q4=/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure
RUN_ROOT=/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01
RUN_COMMAND=/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
REQUIRED_FREE_BYTES=38196451720
MIN_AVAILABLE_BYTES=17179869184
export PYTHONPATH="$STAGE/src:$STAGE/scripts"
test -f "$RUN_ROOT/disk-parent/output/result.json"
test -f "$RUN_ROOT/command-receipts/disk/terminal-status.txt"
test ! -e "$RUN_ROOT/command-receipts/resident"
free_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d '[:space:]')
test "$free_bytes" -ge "$REQUIRED_FREE_BYTES"
available_kb=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
test "$((available_kb * 1024))" -ge "$MIN_AVAILABLE_BYTES"
gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "$gpu_uuid" = "GPU-2d2b1358-9abf-ef75-fc3b-892135ea6e58"
sha256sum -c - <<SHA256
cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca  $Q4/store/checkpoints/cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca.state
9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7  $Q4/store/checkpoints/9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7.state
43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7  $Q4/store/checkpoints/43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7.state
SHA256
nvidia-smi -i 0 --query-gpu=uuid,memory.free,memory.total --format=csv,noheader
mkdir -m 700 -p "$RUN_ROOT/resident-parent"
bash "$RUN_COMMAND" "$STAGE" "$RUN_ROOT/command-receipts/resident" \
  python3 "$STAGE/scripts/p4_return_latency.py" \
  --source-store "$Q4/store" \
  --source-result "$Q4/result.json" \
  --prepare "$Q4/prepare.json" \
  --profile "$Q4/profile.json" \
  --fixture "$Q4/fixture.json" \
  --reference-dir "$Q4" \
  --output "$RUN_ROOT/resident-parent/output" \
  --arm resident \
  --arm-order disk,resident \
  --order-seed p4-q4-arm-order-draw-2026-10-15-01 \
  --ram-bytes 103079215104 \
  --disk-bytes 137438953472 \
  --min-available-bytes 17179869184
```

At preparation time, the three payloads were absent and `/` had 13,687,603,200
bytes free. The pre-restore gate for the disk phase is 43,688,258,416 bytes: the
three 1,830,602,232-byte source payloads, one 15-checkpoint working envelope,
and a 10 GiB floor. After restoration, each arm needs 38,196,451,720 bytes
free. Retaining both 15-checkpoint arm trees locally would require
65,655,485,200 bytes after restore, above the current capacity projection; the
phased archive and exact-payload-reclaim process avoids that combined local
footprint while retaining the complete disk-arm evidence externally. The
verified run-command helper stores exact argv, numeric command/wrapper exits,
stdout, stderr, and terminal status outside each arm's `--output` directory.
The root controls restore, archive verification, reclamation, and launch.
