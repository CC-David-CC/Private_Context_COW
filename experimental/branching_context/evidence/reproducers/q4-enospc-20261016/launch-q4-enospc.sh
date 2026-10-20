#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1
readonly TOOLS=/home/dflanag3/spatial-artifacts/q4-enospc-full-launch-v5
readonly IMAGE="$ROOT/mount-preflight-v1/bounded-ext4.img"
readonly EXPECTED_INODE=5252779
readonly EXPECTED_UUID=f068766f-7ce6-4584-9063-2f80024a0e09
readonly EXPECTED_IMAGE_BYTES=1610612736
readonly LOCK=/home/dflanag3/branching-g3/native-client-speed.gpu0.lock
# Only unallocated image growth can consume additional host space.
# Keep the same 16 GiB reserve after the image reaches its maximum size.
allocated_image_bytes=$(( $(stat -c '%b' "$IMAGE") * 512 ))
remaining_image_bytes=$((EXPECTED_IMAGE_BYTES - allocated_image_bytes))
if (( remaining_image_bytes < 0 )); then remaining_image_bytes=0; fi
readonly MIN_FREE_BYTES=$((remaining_image_bytes + 16 * 1024 * 1024 * 1024))

mode=${1:?mode must be preflight or run}
expected_image_sha=${2:?expected image sha256 required}
if [[ "$mode" != preflight && "$mode" != run ]]; then
  echo 'mode must be preflight or run' >&2
  exit 64
fi
if [[ $(id -u) -ne 1000 || $(id -un) != dflanag3 ]]; then
  echo 'Launcher must run as the ordinary dflanag3 user' >&2
  exit 77
fi
if [[ -L "$IMAGE" || ! -f "$IMAGE" || $(stat -c '%i' "$IMAGE") != "$EXPECTED_INODE" \
   || $(stat -c '%s' "$IMAGE") != "$EXPECTED_IMAGE_BYTES" \
   || $(blkid -p -s UUID -o value "$IMAGE") != "$EXPECTED_UUID" ]]; then
  echo 'Bounded image inode/size/UUID changed' >&2
  exit 78
fi
image_sha=$(sha256sum "$IMAGE" | cut -d ' ' -f1)
if [[ "$image_sha" != "$expected_image_sha" ]]; then
  echo "Bounded image SHA mismatch: $image_sha" >&2
  exit 78
fi
available=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')
if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < MIN_FREE_BYTES )); then
  echo "Refusing private image use below image+16GiB free-space floor: $available" >&2
  exit 79
fi
parent_namespace=$(readlink /proc/self/ns/mnt)
if [[ ! "$parent_namespace" =~ ^mnt:\[[0-9]+\]$ ]]; then
  echo 'Cannot capture parent mount namespace' >&2
  exit 80
fi
exec flock -n "$LOCK" sudo -n unshare --mount --propagation private env \
  COW_PARENT_MOUNT_NAMESPACE="$parent_namespace" \
  bash "$TOOLS/private-mount-run.sh" "$mode" "$expected_image_sha" "$parent_namespace"
