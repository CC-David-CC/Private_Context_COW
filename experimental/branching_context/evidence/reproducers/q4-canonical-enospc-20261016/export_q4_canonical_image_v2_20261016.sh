#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1
IMAGE=$ROOT/canonical-ext4-64m.img
AUDIT=$ROOT/readonly-audit-v2
SELF=/home/dflanag3/spatial-artifacts/export_q4_canonical_image_v2_20261016.sh
PY=/home/dflanag3/src/Strata/.venv/bin/python
if [[ ${1:-} != --inner ]]; then
  expected=${1:?terminal image SHA-256}
  [[ $expected =~ ^[0-9a-f]{64}$ && $(id -u) == 1000 && ! -e $AUDIT ]]
  mkdir -m 700 "$AUDIT" "$AUDIT/mount"
  parent=$(readlink /proc/self/ns/mnt)
  exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
    sudo -n unshare --mount --propagation private bash "$SELF" --inner "$expected" "$parent"
fi
expected=${2:?image SHA-256}
parent=${3:?parent mount namespace}
[[ $(id -u) == 0 && $(readlink /proc/self/ns/mnt) != "$parent" ]]
[[ ! -L $IMAGE && $(stat -c '%i:%s' "$IMAGE") == 5254728:67108864 ]]
[[ $(blkid -p -s UUID -o value "$IMAGE") == 4df1febe-d3f8-401e-908a-81da3824b712 ]]
[[ $(sha256sum "$IMAGE" | cut -d ' ' -f1) == "$expected" ]]
[[ -z $(losetup -j "$IMAGE") ]]
loop=
used_loop=
mounted=0
cleanup_status=0
finish() {
  status=$?
  trap - EXIT
  if [[ $mounted == 1 ]]; then
    if umount "$AUDIT/mount"; then mounted=0; else cleanup_status=91; fi
  fi
  if [[ -n $loop && $mounted == 0 ]]; then
    if losetup -d "$loop"; then loop=; else cleanup_status=92; fi
  fi
  after=$(sha256sum "$IMAGE" | cut -d ' ' -f1)
  if [[ $after != "$expected" ]]; then cleanup_status=93; fi
  printf '{"export_exit":%d,"cleanup_exit":%d,"mount_removed":%s,"loop_detached":%s,"loop":"%s","image_sha256_before":"%s","image_sha256_after":"%s"}\n' \
    "$status" "$cleanup_status" "$([[ $mounted == 0 ]] && echo true || echo false)" \
    "$([[ -z $loop ]] && echo true || echo false)" "$used_loop" "$expected" "$after" > "$AUDIT/cleanup.json"
  chown dflanag3:dflanag3 "$AUDIT/cleanup.json"
  chmod 600 "$AUDIT/cleanup.json"
  cat "$AUDIT/cleanup.json"
  if [[ $cleanup_status != 0 ]]; then exit "$cleanup_status"; fi
  exit "$status"
}
trap finish EXIT
mount --make-rprivate /
loop=$(losetup --read-only --find --show "$IMAGE")
used_loop=$loop
mount -o ro,noload,nosuid,nodev,noexec "$loop" "$AUDIT/mount"
mounted=1
"$PY" -B /home/dflanag3/spatial-artifacts/export_canonical_enospc_metadata_20261016.py \
  "$AUDIT/mount" "$AUDIT/canonical-export" "$AUDIT/export.json"
