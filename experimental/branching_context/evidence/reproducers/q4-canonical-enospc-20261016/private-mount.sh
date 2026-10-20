#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1
IMAGE=$ROOT/canonical-ext4-64m.img
STORE=$ROOT/store-mount
META=$ROOT/input-metadata-v1
CACHE=$ROOT/checkpoint-volume-v1
ORIGINAL=/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/input-store-v1
RUNTIME=/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/runtime-v4
PROFILE=/home/dflanag3/spatial-artifacts/lazy-prefix-20261015-v1/profiles-v2/q4/profile.json
PY=/home/dflanag3/src/Strata/.venv/bin/python
mode=${1:?mode required}
driver=${2:?driver required}
driver_sha=${3:?driver hash required}
image_sha=${4:?image hash required}
parent_namespace=${5:?parent namespace required}
[[ $(id -u) == 0 && ( $mode == preflight || $mode == filler || $mode == run ) ]]
[[ $parent_namespace =~ ^mnt:\[[0-9]+\]$ && $(readlink /proc/self/ns/mnt) != "$parent_namespace" ]]
[[ ! -L $IMAGE && $(stat -c '%i:%s' "$IMAGE") == 5254728:67108864 ]]
[[ $(sha256sum "$IMAGE" | cut -d ' ' -f1) == "$image_sha" ]]
[[ $(sha256sum "$driver" | cut -d ' ' -f1) == "$driver_sha" ]]
[[ $(blkid -p -s UUID -o value "$IMAGE") == 4df1febe-d3f8-401e-908a-81da3824b712 ]]
for path in "$STORE" "$META" "$CACHE" "$ORIGINAL"; do [[ -d $path && ! -L $path ]]; done
[[ -z $(losetup -j "$IMAGE") ]]
! mountpoint -q "$STORE"
RECEIPTS=$ROOT/mount-$mode-v3
[[ ! -e $RECEIPTS ]]
mkdir -m 700 "$RECEIPTS"
chown dflanag3:dflanag3 "$RECEIPTS"
loop=
loop_used=
mounted=0
bound=0
cleanup_status=0
finish() {
  status=$?
  trap - EXIT
  if [[ $bound == 1 ]]; then
    if umount "$STORE/checkpoints"; then bound=0; else cleanup_status=91; fi
  fi
  if [[ $mounted == 1 && $bound == 0 ]]; then
    if umount "$STORE"; then mounted=0; else cleanup_status=92; fi
  fi
  if [[ -n $loop && $mounted == 0 ]]; then
    if losetup -d "$loop"; then loop=; else cleanup_status=93; fi
  fi
  after=$(sha256sum "$IMAGE" | cut -d ' ' -f1)
  printf '{"mode":"%s","probe_exit":%d,"cleanup_exit":%d,"cache_unmounted":%s,"canonical_unmounted":%s,"loop":"%s","loop_detached":%s,"image_sha256_before":"%s","image_sha256_after":"%s"}\n' \
    "$mode" "$status" "$cleanup_status" "$([[ $bound == 0 ]] && echo true || echo false)" \
    "$([[ $mounted == 0 ]] && echo true || echo false)" "$loop_used" \
    "$([[ -z $loop ]] && echo true || echo false)" "$image_sha" "$after" > "$RECEIPTS/terminal-receipt.json"
  chown dflanag3:dflanag3 "$RECEIPTS/terminal-receipt.json"
  chmod 600 "$RECEIPTS/terminal-receipt.json"
  cat "$RECEIPTS/terminal-receipt.json"
  if [[ $cleanup_status != 0 ]]; then exit "$cleanup_status"; fi
  exit "$status"
}
trap finish EXIT
mount --make-rprivate /
loop=$(losetup --find --show "$IMAGE")
loop_used=$loop
mount -o rw,nosuid,nodev,noexec "$loop" "$STORE"
mounted=1
[[ -d $STORE/lost+found && ! -L $STORE/lost+found ]]
[[ -z $(find "$STORE/lost+found" -mindepth 1 -print -quit) ]]
if [[ ! -e $STORE/refs.json ]]; then
  [[ -z $(find "$STORE" -mindepth 1 -maxdepth 1 ! -name lost+found -print -quit) ]]
  cp -a "$META/." "$STORE/"
fi
chown dflanag3:dflanag3 "$STORE"
chmod 700 "$STORE"
[[ -d $STORE/checkpoints && ! -L $STORE/checkpoints ]]
[[ -z $(find "$STORE/checkpoints" -mindepth 1 -print -quit) ]]
mount --bind "$CACHE" "$STORE/checkpoints"
bound=1
[[ $(stat -c '%d' "$STORE") != $(stat -c '%d' "$STORE/checkpoints") ]]
[[ $(sha256sum "$STORE/objects/808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85.json" | cut -d ' ' -f1) == 808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85 ]]
[[ $(sha256sum "$STORE/checkpoints/423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810.state" | cut -d ' ' -f1) == 423c332989865bff62838ebed5a444f1d7ae402c1db59a69f0310b4c78734810 ]]
sync -f "$STORE"
if [[ $mode == filler ]]; then
  runuser -u dflanag3 -- env PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B /home/dflanag3/spatial-artifacts/probe_canonical_filler_20261016.py \
    "$driver" "$STORE" "$ROOT/filler-only-v1"
  exit 0
fi
preflight=()
if [[ $mode == preflight ]]; then preflight=(--preflight-only); fi
runuser -u dflanag3 -- env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$(dirname "$driver"):$RUNTIME:$RUNTIME/experimental/branching_context/src:$RUNTIME/experimental/branching_context/scripts" \
  "$PY" -B "$driver" \
  --runtime "$RUNTIME" --api-src "$RUNTIME/experimental/branching_context/src" \
  --profile "$PROFILE" --store "$STORE" --original-store "$ORIGINAL" \
  --branch CliA --run-dir "$ROOT/gpu-q4-v2/campaign" \
  --canonical-mount "$STORE" --checkpoint-directory "$STORE/checkpoints" \
  --loop-image "$IMAGE" --parent-mount-namespace "$parent_namespace" \
  --expected-image-uuid 4df1febe-d3f8-401e-908a-81da3824b712 --expected-image-inode 5254728 \
  --expected-profile-sha256 f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c \
  --expected-native-sha256 4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451 \
  --expected-source-snapshot 808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85 \
  --ram-gib 96 --disk-gib 128 --max-fill-mib 64 --max-tokens 16 "${preflight[@]}"
