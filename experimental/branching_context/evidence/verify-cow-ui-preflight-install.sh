#!/usr/bin/env bash
set -euo pipefail

: "$PRE"
: "$DEST"
test -d "$PRE"
if [[ -e "$DEST" || -L "$DEST" ]]; then
  echo "install-check destination already exists" >&2
  exit 73
fi
mkdir -m 700 "$DEST"
files=(
  cow_http_keyfile.py
  run-cow-http-pid.sh
  run-cow-ui-smoke.sh
  start-cow-service.sh
  signal-cow-server.py
  cow_fresh_reference.py
  scan_cow_ui_artifacts.py
  cow_native_smoke.py
)
for name in "${files[@]}"; do
  install -m 700 "$PRE/$name" "$DEST/$name"
done
(cd "$PRE" && sha256sum --check SHA256SUMS)
(cd "$DEST" && sha256sum --check "$PRE/SHA256SUMS")
bash -n "$DEST/start-cow-service.sh" "$DEST/run-cow-http-pid.sh" "$DEST/run-cow-ui-smoke.sh"
python3 -m py_compile \
  "$DEST/cow_http_keyfile.py" "$DEST/signal-cow-server.py" \
  "$DEST/cow_fresh_reference.py" "$DEST/scan_cow_ui_artifacts.py" \
  "$DEST/cow_native_smoke.py"
test ! -e "$DEST/browser"
echo "preflight install source and installed hashes match; syntax checks pass; browser output remains absent"
