#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-canonical-enospc-20261016-v1
IMAGE=$ROOT/canonical-ext4-64m.img
INNER=$ROOT/launch-v4/private-mount.sh
mode=${1:?mode required}
driver=${2:?driver path required}
driver_sha=${3:?driver hash required}
image_sha=${4:?image hash required}
[[ $(id -u) == 1000 && ( $mode == preflight || $mode == filler || $mode == run ) ]]
[[ $driver == "$ROOT"/*/source/experimental/branching_context/scripts/run_q4_canonical_enospc.py ]]
[[ ! -L $driver && -f $driver && $(sha256sum "$driver" | cut -d ' ' -f1) == "$driver_sha" ]]
[[ ! -L $IMAGE && -f $IMAGE && $(stat -c '%i:%s' "$IMAGE") == 5254728:67108864 ]]
[[ $(blkid -p -s UUID -o value "$IMAGE") == 4df1febe-d3f8-401e-908a-81da3824b712 ]]
[[ $(sha256sum "$IMAGE" | cut -d ' ' -f1) == "$image_sha" ]]
available=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')
# Reserve for two new checkpoint payloads and the complete fault image.
(( available >= 16*1024*1024*1024 + 1024*1024*1024 + 64*1024*1024 ))
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
parent_namespace=$(readlink /proc/self/ns/mnt)
[[ $parent_namespace =~ ^mnt:\[[0-9]+\]$ ]]
exec flock -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
  sudo -n unshare --mount --propagation private bash "$INNER" \
  "$mode" "$driver" "$driver_sha" "$image_sha" "$parent_namespace"
