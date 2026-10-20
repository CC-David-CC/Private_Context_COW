#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1
readonly TOOLS=/home/dflanag3/spatial-artifacts/q4-enospc-full-launch-v5
readonly INPUT_STORE="$ROOT/input-store-v1"
readonly ORIGINAL_STORE=/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/input-store-v1
readonly IMAGE="$ROOT/mount-preflight-v1/bounded-ext4.img"
readonly MOUNT="$INPUT_STORE/checkpoints"
readonly PROFILE=/home/dflanag3/spatial-artifacts/lazy-prefix-20261015-v1/profiles-v2/q4/profile.json
readonly PROFILE_SHA=f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c
readonly NATIVE_SHA=4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451
readonly SNAPSHOT=808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85
readonly MANIFEST_SHA=f97f7fcb8834d0dd3e69e6a3dc705e131a1fe2b62bf62f5506766b2f86f3e177
readonly PAYLOAD=423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810.state
readonly PAYLOAD_SHA=423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810
readonly RUNTIME=/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/runtime-v4
readonly API_SRC="$RUNTIME/experimental/branching_context/src"
readonly API_SCRIPTS="$RUNTIME/experimental/branching_context/scripts"
readonly PYTHON=/home/dflanag3/src/Strata/.venv/bin/python
readonly DRIVER="/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/repair-v3-green/source/experimental/branching_context/scripts/run_q4_enospc.py"
readonly DRIVER_SCRIPTS="/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1/repair-v3-green/source/experimental/branching_context/scripts"
readonly PROBE_RUN="$ROOT/gpu-q4-enospc-v3/probe-result"


mode=${1:?mode required}
expected_image_sha=${2:?image hash required}
parent_namespace=${3:?parent mount namespace required}
if [[ "$mode" == preflight ]]; then
  RECEIPT_DIR="$ROOT/private-mount-preflight-v5"
else
  RECEIPT_DIR="$ROOT/private-mount-run-v5"
fi
if [[ $(id -u) -ne 0 || ( "$mode" != preflight && "$mode" != run ) ]]; then
  echo 'Private namespace setup must run as root in preflight/run mode' >&2
  exit 64
fi
if [[ ! "$parent_namespace" =~ ^mnt:\[[0-9]+\]$ || "$parent_namespace" != "$COW_PARENT_MOUNT_NAMESPACE" ]]; then
  echo 'Parent mount namespace argument/environment mismatch' >&2
  exit 65
fi
if [[ -L "$INPUT_STORE" || -L "$MOUNT" || ! -d "$INPUT_STORE" || ! -d "$MOUNT" ]]; then
  echo 'Disposable Store or checkpoint mountpoint is missing or redirected' >&2
  exit 66
fi
if [[ -L "$ORIGINAL_STORE" || ! -d "$ORIGINAL_STORE" ]]; then
  echo 'Original source Store identity path is invalid' >&2
  exit 66
fi
if [[ -L "$IMAGE" || ! -f "$IMAGE" || $(stat -c '%i' "$IMAGE") != 5252779 \
   || $(blkid -p -s UUID -o value "$IMAGE") != f068766f-7ce6-4584-9063-2f80024a0e09 \
   || $(sha256sum "$IMAGE" | cut -d ' ' -f1) != "$expected_image_sha" ]]; then
  echo 'Private image changed after outer preflight' >&2
  exit 67
fi
if [[ $(sha256sum "$INPUT_STORE/checkpoints/$SNAPSHOT.json" | cut -d ' ' -f1) != "$MANIFEST_SHA" \
   || $(sha256sum "$ORIGINAL_STORE/checkpoints/$SNAPSHOT.json" | cut -d ' ' -f1) != "$MANIFEST_SHA" \
   || $(sha256sum "$ORIGINAL_STORE/checkpoints/$PAYLOAD" | cut -d ' ' -f1) != "$PAYLOAD_SHA" \
   || $(sha256sum "$INPUT_STORE/objects/$SNAPSHOT.json" | cut -d ' ' -f1) != "$SNAPSHOT" ]]; then
  echo 'Source snapshot/manifest/payload/object pins do not match' >&2
  exit 68
fi
if [[ -e "$PROBE_RUN" || -L "$PROBE_RUN" ]]; then
  echo 'Probe result directory must be fresh' >&2
  exit 69
fi
if [[ -e "$RECEIPT_DIR" || -L "$RECEIPT_DIR" ]]; then
  echo 'Mount receipt directory must be fresh' >&2
  exit 69
fi
mkdir -m 700 "$RECEIPT_DIR"
chown dflanag3:dflanag3 "$RECEIPT_DIR"
chmod 700 "$RECEIPT_DIR"
if [[ -n $(losetup -j "$IMAGE") ]] || mountpoint -q "$MOUNT"; then
  echo 'Refusing already attached/mounted bounded image' >&2
  exit 70
fi
mount --make-rprivate /
loop_device=
loop_used=
mounted=0
probe_status=0
cleanup_status=0
finish() {
  status=$?
  trap - EXIT
  if [[ "$mounted" -eq 1 ]]; then
    if umount "$MOUNT"; then mounted=0; else cleanup_status=91; fi
  fi
  if [[ -n "$loop_device" ]]; then
    if losetup -d "$loop_device"; then loop_device=; else cleanup_status=92; fi
  fi
  image_after=$(sha256sum "$IMAGE" | cut -d ' ' -f1)
  inode_after=$(stat -c '%i' "$IMAGE")
  printf '{"mode":"%s","probe_exit":%d,"cleanup_exit":%d,"mount_removed":%s,"loop":"%s","loop_detached":%s,"image_inode_after":"%s","image_sha_before":"%s","image_sha_after":"%s"}\n' \
    "$mode" "$status" "$cleanup_status" "$([[ "$mounted" -eq 0 ]] && echo true || echo false)" \
    "$loop_used" "$([[ -z "$loop_device" ]] && echo true || echo false)" "$inode_after" "$expected_image_sha" "$image_after" \
    | tee "$RECEIPT_DIR/terminal-receipt.json"
  chown dflanag3:dflanag3 "$RECEIPT_DIR/terminal-receipt.json"
chmod 600 "$RECEIPT_DIR/terminal-receipt.json"
  if [[ "$cleanup_status" -ne 0 ]]; then exit "$cleanup_status"; fi
  exit "$status"
}
trap finish EXIT
loop_device=$(losetup --find --show "$IMAGE")
loop_used=$loop_device
mount -o rw,nosuid,nodev,noexec "$loop_device" "$MOUNT"
mounted=1
chown dflanag3:dflanag3 "$MOUNT"
chmod 700 "$MOUNT"
source_manifest="$ORIGINAL_STORE/checkpoints/$SNAPSHOT.json"
source_payload="$ORIGINAL_STORE/checkpoints/$PAYLOAD"
if [[ ! -d "$MOUNT/lost+found" || -L "$MOUNT/lost+found" \
   || $(stat -c '%u:%g' "$MOUNT/lost+found") != 0:0 \
   || -n $(find "$MOUNT/lost+found" -mindepth 1 -print -quit) ]]; then
  echo 'Ext4 lost+found must remain the empty root-owned mkfs directory' >&2
  exit 71
fi
if [[ -e "$MOUNT/$SNAPSHOT.json" && -e "$MOUNT/$PAYLOAD" ]]; then
  [[ ! -L "$MOUNT/$SNAPSHOT.json" && ! -L "$MOUNT/$PAYLOAD" ]]
  [[ $(sha256sum "$MOUNT/$SNAPSHOT.json" | cut -d ' ' -f1) == "$MANIFEST_SHA" ]]
  [[ $(sha256sum "$MOUNT/$PAYLOAD" | cut -d ' ' -f1) == "$PAYLOAD_SHA" ]]
else
  unexpected_entry=$(find "$MOUNT" -mindepth 1 -maxdepth 1 ! -name lost+found -print -quit)
  if [[ -n "$unexpected_entry" ]]; then
    echo 'Mounted image contains unexpected entries before source checkpoint installation' >&2
    exit 71
  fi
  install -m 600 "$source_manifest" "$MOUNT/$SNAPSHOT.json"
  install -m 600 "$source_payload" "$MOUNT/$PAYLOAD"
  chown dflanag3:dflanag3 "$MOUNT/$SNAPSHOT.json" "$MOUNT/$PAYLOAD"
  sync -f "$MOUNT/$SNAPSHOT.json"
  sync -f "$MOUNT/$PAYLOAD"
  sync -f "$MOUNT"
fi
unexpected=$(find "$MOUNT" -mindepth 1 -maxdepth 1 ! -name lost+found \
  ! -name "$SNAPSHOT.json" ! -name "$PAYLOAD" -print -quit)
if [[ -n "$unexpected" || $(find "$MOUNT" -mindepth 1 -maxdepth 1 -type f | wc -l) -ne 2 ]]; then
  echo 'Mounted image must contain only lost+found and the pinned source manifest/payload' >&2
  exit 72
fi
"$PYTHON" -B - "$MOUNT" <<'PY'
import os, sys
root=sys.argv[1]
fd=os.open(root, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
try: os.fsync(fd)
finally: os.close(fd)
PY
chown dflanag3:dflanag3 "$MOUNT"
chmod 700 "$MOUNT"
if [[ $(sha256sum "$MOUNT/$SNAPSHOT.json" | cut -d ' ' -f1) != "$MANIFEST_SHA" \
   || $(sha256sum "$MOUNT/$PAYLOAD" | cut -d ' ' -f1) != "$PAYLOAD_SHA" ]]; then
  echo 'Mounted source checkpoint did not verify after copy' >&2
  exit 73
fi
if [[ ! -d "$API_SRC" || ! -d "$API_SCRIPTS" || ! -f "$DRIVER" || ! -f "$PROFILE" ]]; then
  echo 'Pinned API/runtime/profile/driver paths are missing' >&2
  exit 74
fi
if [[ "$mode" == preflight ]]; then
  probe_run=$ROOT/preflight-only-must-not-be-created
  preflight_flag=--preflight-only
else
  probe_run=$PROBE_RUN
  preflight_flag=
fi
set +e
runuser -u dflanag3 -- env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$DRIVER_SCRIPTS:$RUNTIME:$API_SRC:$API_SCRIPTS" \
  "$PYTHON" -B "$DRIVER" \
  --runtime-root "$RUNTIME" --api-src "$API_SRC" --api-scripts "$API_SCRIPTS" \
  --profile "$PROFILE" --store "$INPUT_STORE" --original-store "$ORIGINAL_STORE" \
  --branch CliA --run-dir "$probe_run" --loop-image "$IMAGE" --checkpoint-mount "$MOUNT" \
  --parent-mount-namespace "$parent_namespace" \
  --expected-profile-sha256 "$PROFILE_SHA" --expected-native-sha256 "$NATIVE_SHA" \
  --expected-source-snapshot "$SNAPSHOT" --ram-gib 96 --disk-gib 128 \
  --max-fill-mib 1400 --max-tokens 128 $preflight_flag
probe_status=$?
set -e
exit "$probe_status"
