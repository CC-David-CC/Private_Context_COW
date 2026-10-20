#!/usr/bin/env bash
set -euo pipefail
umask 077

STAGE=/home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red/experimental/branching_context
Q4=/home/dflanag3/spatial-artifacts/p5-q4-full96-run-20261015-v1/q4-64k-pressure
RAW_FIXTURE=/home/dflanag3/branching-g3/multi-code-contexts-20261014/fixtures/q4_64k.json
NORMALIZER=/home/dflanag3/branching-g3/p5-q4-root24b4-source-20261015/experimental/branching_context/scripts/build_codebase_fixture.py
PRIOR_RUN_ROOT=/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01
RUN_ROOT=/home/dflanag3/spatial-artifacts/p4-q4-lifecycle-attribution-20261015-v1
RUN_COMMAND=/home/dflanag3/spatial-artifacts/p4-return-runner/run-command.sh/run-command.sh
TOOL=/home/dflanag3/branching-g3/checkpoint-lifecycle-diagnostics-luna-20261015-v9/experimental/branching_context/scripts/checkpoint_lifecycle_diagnostics.py
GPU_LOCK=/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
DISK_RESULT=$PRIOR_RUN_ROOT/disk-parent/output/result.json
RESIDENT_OUTPUT=$RUN_ROOT/output/resident
RESIDENT_RECEIPT=$RUN_ROOT/command-receipts/resident
RECLAIM_RECEIPT=/home/dflanag3/spatial-artifacts/p4-q4-disk-reclaim-20261015-01/unlink-receipt.json
RECLAIM_MAP=/home/dflanag3/spatial-artifacts/p4-q4-disk-reclaim-20261015-01/reclaim-map.json
PIN_CHECKER=/home/dflanag3/spatial-artifacts/p4-q4-resident-source-check-20261015-01/input/p4-q4-resident-verify-build-pins-v2.py
REQUIRED_FREE_BYTES=41512139080
MIN_AVAILABLE_BYTES=17179869184
RAM_BYTES=103079215104
DISK_BYTES=137438953472
export PYTHONPATH="$STAGE/src:$STAGE/scripts"

# Hold the designated GPU-0 lock across preflight and the entire runner process.
test -f "$GPU_LOCK"
test ! -L "$GPU_LOCK"
exec 9>>"$GPU_LOCK"
flock -n 9 || { echo "P4_RESIDENT_PREFLIGHT_FAIL: GPU-0 lock is held" >&2; exit 2; }

test -f "$RAW_FIXTURE"
test -f "$NORMALIZER"
test -d "$Q4/store/checkpoints"
test ! -e "$RUN_ROOT"

test ! -e "$RESIDENT_OUTPUT"
test ! -e "$RESIDENT_RECEIPT"
test "$(cat "$PRIOR_RUN_ROOT/command-receipts/disk/terminal-status.txt")" = COMPLETE
test "$(cat "$PRIOR_RUN_ROOT/command-receipts/disk/command.exit")" = 0
test "$(cat "$PRIOR_RUN_ROOT/command-receipts/disk/wrapper.exit")" = 0
test "$(cat /home/dflanag3/spatial-artifacts/p4-q4-disk-reclaim-unlink-command-20261015-01/terminal-status.txt)" = COMPLETE
test "$(cat /home/dflanag3/spatial-artifacts/p4-q4-disk-reclaim-unlink-command-20261015-01/command.exit)" = 0
test "$(cat /home/dflanag3/spatial-artifacts/p4-q4-disk-reclaim-unlink-command-20261015-01/wrapper.exit)" = 0

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
7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537  /home/dflanag3/spatial-artifacts/p4-q4-disk-archive-20261015-01/run-command.sh
0afcc23a2d30889586341b70f5efe443714a541e1fb932a5b32d467f2ff07bf9  /home/dflanag3/spatial-artifacts/p4-q4-disk-reclaim-20261015-01/unlink-receipt.json
6fdfdbd3e6ee8eae2926d0fb316bfa7eb047eb9d15f34bf974c491ac838f2a56  /home/dflanag3/spatial-artifacts/p4-q4-disk-reclaim-20261015-01/reclaim-map.json
SHA256
printf '%s  %s\n' 4fd182338e4f9613c26b15b0d48aa18a18efb9b0479bd60e31726ba9bc071906 "$TOOL" | sha256sum --check --status
printf '%s  %s\n' 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537 "$RUN_COMMAND" | sha256sum --check --status

python3 - "$RECLAIM_RECEIPT" "$RECLAIM_MAP" "$PRIOR_RUN_ROOT/disk-parent/output/store/checkpoints" <<'PY'
import json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
mapping = json.loads(pathlib.Path(sys.argv[2]).read_text())
checkpoint_dir = pathlib.Path(sys.argv[3])
expected = [row["relative_path"] for row in mapping["candidates"]]
if receipt.get("status") != "PASS" or receipt.get("unlink_performed") is not True:
    raise SystemExit("disk-arm reclaim receipt is not PASS")
if receipt.get("candidate_map_sha256") != "6fdfdbd3e6ee8eae2926d0fb316bfa7eb047eb9d15f34bf974c491ac838f2a56":
    raise SystemExit("reclaim receipt map hash changed")
if receipt.get("removed") != expected or receipt.get("remaining") != []:
    raise SystemExit("reclaim receipt does not list exactly all mapped candidates")
if receipt.get("candidate_count") != 9 or receipt.get("candidate_bytes") != 17789976604:
    raise SystemExit("reclaim receipt candidate count/bytes changed")
if receipt.get("protected_metadata_fingerprint_before") != "5ee8f5859e02ff3366de427744a272cd6f3c78cbd61da9dbe579cc54a17ebbe8" or receipt.get("protected_metadata_fingerprint_after") != receipt.get("protected_metadata_fingerprint_before"):
    raise SystemExit("protected metadata fingerprint changed across reclaim")
for rel in expected:
    path = checkpoint_dir / pathlib.PurePosixPath(rel).name
    if path.exists() or path.is_symlink():
        raise SystemExit(f"reclaimed payload unexpectedly still exists: {path}")
print("RECLAIM_RECEIPT_AND_EXACT_PATHS_PASS candidates=9 protected_files=90")
PY

# The frozen result is explicitly pinned and the verifier merges duplicate
# source/artifact paths only when their recorded digests agree.
sha256sum -c - <<'SHA256'
76438074db2af7117f79764de759cb71363d08185bf24353225783ccc5ccd9f0  /home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01/disk-parent/output/result.json
646becefff5af8a3c46e7ab5332718b0a5ff69b7f036832ff5794a0aa469beed  /home/dflanag3/spatial-artifacts/p4-q4-resident-source-check-20261015-01/input/p4-q4-resident-verify-build-pins-v2.py
SHA256
python3 "$PIN_CHECKER" "$DISK_RESULT"

sha256sum -c - <<SHA256
cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca  $Q4/store/checkpoints/cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca.state
9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7  $Q4/store/checkpoints/9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7.state
43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7  $Q4/store/checkpoints/43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7.state
07a8d31e1d61e71e25f15322ea5ddd8608e9244a242863ad49e24ce389849a69  $Q4/store/checkpoints/e3ecceb2ade807fdf33d87e66d9f5f4c4e901ae6a5c14ec49b36ca11ee1e5286.json
e897d52f34c2cd445635e7c7002455ac8d93eeef7b581881d1b0de43b93cc9f2  $Q4/store/checkpoints/c4386e8b4c4ffe8793ab0dfc34e899b52e7185b74114c455268b22b9bf84b316.json
5c725a23cafa398c40c24e202cf6b8d23cee900d9f77bccb50b85b2e0605e8e4  $Q4/store/checkpoints/bb974fb5a77c22a4c4ee3e9067a404fa937b70fee53cc469c54273aa02fc584a.json
SHA256

for file in \
  "$Q4/store/checkpoints/cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca.state" \
  "$Q4/store/checkpoints/9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7.state" \
  "$Q4/store/checkpoints/43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7.state"; do
  test "$(stat -c %s "$file")" -eq 1830602232
  test "$(stat -c %a "$file")" = 600
  test "$(stat -c %h "$file")" -eq 1
done

free_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d '[:space:]')
if (( free_bytes < REQUIRED_FREE_BYTES )); then
  echo "P4_RESIDENT_PREFLIGHT_FAIL: disk free=$free_bytes required=$REQUIRED_FREE_BYTES" >&2
  exit 2
fi
available_kb=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
available_bytes=$((available_kb * 1024))
if (( available_bytes < MIN_AVAILABLE_BYTES )); then
  echo "P4_RESIDENT_PREFLIGHT_FAIL: MemAvailable=$available_bytes required=$MIN_AVAILABLE_BYTES" >&2
  exit 2
fi
if (( free_bytes < REQUIRED_FREE_BYTES )); then
  echo "LIFECYCLE_PREFLIGHT_FAIL: external disk gate free=$free_bytes required=$REQUIRED_FREE_BYTES" >&2
  exit 2
fi
gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
test "$gpu_uuid" = "GPU-2d2b1358-9abf-ef75-fc3b-892135ea6e58"
gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')
test -z "$gpu_apps"
gpu_util=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
test "$gpu_util" -eq 0
nvidia-smi -i 0 --query-gpu=uuid,memory.free,memory.total,utilization.gpu --format=csv,noheader
printf 'P4_RESIDENT_PREFLIGHT_PASS free_bytes=%s required_free_bytes=%s MemAvailable_bytes=%s min_available_bytes=%s profile=9b23df373eb42687a2eca77fb99a22409ada0ab4c3edac732da2674bd23d3763 output_limit=128 arm_order=disk,resident\n' \
  "$free_bytes" "$REQUIRED_FREE_BYTES" "$available_bytes" "$MIN_AVAILABLE_BYTES"

if [[ "${1:-}" == "--preflight-only" ]]; then
  printf 'LIFECYCLE_PREFLIGHT_PASS disk_free=%s required_disk_free=%s memavailable=%s gpu_uuid=%s\n' "$free_bytes" "$REQUIRED_FREE_BYTES" "$available_bytes" "$gpu_uuid"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 64
fi
mkdir -m 700 -p "$RUN_ROOT/output" "$RUN_ROOT/diagnostics" "$RUN_ROOT/command-receipts"
bash "$RUN_COMMAND" "$STAGE" "$RESIDENT_RECEIPT" \
  python3 "$TOOL" \
  --source-root "/home/dflanag3/spatial-artifacts/p4-return-luna-20261015-red" \
  --diagnostics "$RUN_ROOT/diagnostics/resident.json" \
  --storage-check-path "$RUN_ROOT" --required-free-bytes "$REQUIRED_FREE_BYTES" \
  --source-store "$Q4/store" \
  --source-result "$Q4/result.json" \
  --prepare "$Q4/prepare.json" \
  --profile "$Q4/profile.json" \
  --fixture "$Q4/fixture.json" \
  --reference-dir "$Q4" \
  --output "$RESIDENT_OUTPUT" \
  --arm resident \
  --arm-order disk,resident \
  --order-seed p4-q4-arm-order-draw-2026-10-15-01 \
  --ram-bytes "$RAM_BYTES" \
  --disk-bytes "$DISK_BYTES" \
  --min-available-bytes "$MIN_AVAILABLE_BYTES"
