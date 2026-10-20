#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1
IMAGE=$ROOT/mount-preflight-v1/bounded-ext4.img
AUDIT=$ROOT/readonly-audit-v3
MOUNT=$AUDIT/mount
SELF=/home/dflanag3/spatial-artifacts/run_audit_q4_enospc_v3_20261016.sh
if [[ ${1:-} != --inner ]]; then
  [[ $(id -u) == 1000 && ! -e "$AUDIT" ]]
  mkdir -m 700 "$AUDIT" "$MOUNT"
  parent=$(readlink /proc/self/ns/mnt)
  exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
    sudo -n unshare --mount --propagation private bash "$SELF" --inner "$parent"
fi
[[ $(id -u) == 0 && $(readlink /proc/self/ns/mnt) != "$2" ]]
[[ ! -L "$IMAGE" && $(stat -c '%i' "$IMAGE") == 5252779 ]]
[[ $(blkid -p -s UUID -o value "$IMAGE") == f068766f-7ce6-4584-9063-2f80024a0e09 ]]
[[ $(sha256sum "$IMAGE" | cut -d ' ' -f1) == e437fe3e17fe3246d6016b4cf9928870943c82ddbdd679ca125e4d2db29f99d8 ]]
[[ -z $(losetup -j "$IMAGE") ]]
mount --make-rprivate /
loop=
mounted=0
cleanup() {
  status=$?
  trap - EXIT
  if [[ $mounted == 1 ]]; then umount "$MOUNT"; mounted=0; fi
  if [[ -n $loop ]]; then losetup -d "$loop"; loop=; fi
  after=$(sha256sum "$IMAGE" | cut -d ' ' -f1)
  printf '{"audit_exit":%d,"mount_removed":true,"loop_detached":true,"image_sha256_after":"%s"}\n' "$status" "$after" > "$AUDIT/cleanup.json"
  chown dflanag3:dflanag3 "$AUDIT/cleanup.json"
  chmod 600 "$AUDIT/cleanup.json"
  exit "$status"
}
trap cleanup EXIT
loop=$(losetup --read-only --find --show "$IMAGE")
mount -o ro,noload,nosuid,nodev,noexec "$loop" "$MOUNT"
mounted=1
runuser -u dflanag3 -- /home/dflanag3/src/Strata/.venv/bin/python -B \
  /home/dflanag3/spatial-artifacts/audit_q4_enospc_v3_20261016.py "$MOUNT"
