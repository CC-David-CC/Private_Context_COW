#!/usr/bin/env bash
set -euo pipefail
root=/home/dflanag3/spatial-artifacts/q4-enospc-harness-20261016-v1
tools=/home/dflanag3/spatial-artifacts/q4-enospc-full-launch-v5
source_dir="$root/repair-v3-green/source"
runner=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
test "$(sha256sum "$runner" | cut -d ' ' -f1)" = 7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537
test "$(sha256sum "$tools/launch-q4-enospc.sh" | cut -d ' ' -f1)" = e700c283d73825ce268a1679ca667a58d4161236be288d7d89a75a03017c0045
test "$(sha256sum "$tools/private-mount-run.sh" | cut -d ' ' -f1)" = 0ea9f737c5076006006d978312a7b4d5c018fa57fffc067a88690337aff8287a
test "$(sha256sum "$source_dir/experimental/branching_context/scripts/run_q4_enospc.py" | cut -d ' ' -f1)" = 2a68537ec177f273fdaa5ce19838a2c6dd29285c8de3ae27fb770d003901da52
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test ! -e "$tools/gpu-command-v3"
chmod -R a-w "$source_dir"
exec bash "$runner" "$source_dir" "$tools/gpu-command-v3" \
  bash "$tools/launch-q4-enospc.sh" run a32d86371637926c1818db53fe21862f3ee7a420590a63e1019760b8359d58f8
