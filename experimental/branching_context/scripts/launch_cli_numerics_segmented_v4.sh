#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/cli-workflow-harness-20261016-v1
STAGE="$ROOT/audit-root-preflight-v3/source"
RUNTIME="$ROOT/credential-green-v1/source"
PROFILE="$ROOT/credential-green-v1/profile.json"
CAMPAIGN="$ROOT/runs/g4-cli-q4-tiny-redaction-v1/campaign/result.json"
ORACLE="$ROOT/audit-oracle-compile-v1/strata-batched-fresh-segmented-audit"
ORIGINAL_BUILD="/home/dflanag3/spatial-artifacts/cow-seal-stream-candidate-20261015-v1/native-probe-v3/build.json"
SEGMENTED_BUILD="$ROOT/audit-oracle-compile-v1/compile-link.json"
INPUT_PROOF="$ROOT/audit-oracle-compile-v1/input-verification-final.json"
RUNNER=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh
RUNNER_SHA=7053d5ffe05e7050b035d6b715b794dd742b68ee29b9198a703979391323f537
PROFILE_SHA=9acd2231483bda8d04eb81c19514ba1b06eb31f8aa7a4fbc825e45f967c21328
CAMPAIGN_SHA=a5f02ae8868dde74dfb842256145dc1302f1779a03bcc6d6527c1d5eb169d9dc
AUDIT_OUTPUT="$ROOT/runs/g4-cli-q4-tiny-redaction-v1-numerics-segmented-v3"
RUN_DIR="$ROOT/runs/g4-cli-q4-tiny-redaction-v1-numerics-command-v3"
PYTHON=/home/dflanag3/src/Strata/.venv/bin/python
sha_check() { test "$(sha256sum "$1" | awk '{print $1}')" = "$2"; }
sha_check "$RUNNER" "$RUNNER_SHA"
sha_check "$PROFILE" "$PROFILE_SHA"
sha_check "$CAMPAIGN" "$CAMPAIGN_SHA"
sha_check "$STAGE/native/fresh_prefill_segmented.cpp" 85f9c1942f697ede539f977674db4ffe9af9f0800a30ace4dfe7baf718223b5c
sha_check "$STAGE/scripts/verify_cli_workflow_numerics.py" d8bf1906b9237d475b1022ac414ffb3e8aa7e99b4db6ce6f170d21ac44564007
sha_check "$STAGE/tests/implementation/test_cli_workflow_numerics.py" 04ebd2b33260377cf9069d29a242afa2f5345a8a92820efd56916404025935be
sha_check "$ORACLE" 130866087ea5c19e1a6b2c556e77a9537ffa4854d19bb637696af8c612ffad5a
sha_check "$ORIGINAL_BUILD" d251cd11267d2e32a57144d25588a7f6450bd466df72f1cd828b21b36a2c7b0c
sha_check "$SEGMENTED_BUILD" 7b26945aef71ed8ad46e3c8c030b9eebe3381a0b5fe6dab9f152667c6ac6aa30
sha_check "$INPUT_PROOF" 9ec1bb03b22e8b18449dc62775c57b84658b84181adb9f8c2787ef75ce8af6e1
for path in "$AUDIT_OUTPUT" "$RUN_DIR"; do
  if [ -e "$path" ]; then echo "refusing existing output leaf: $path" >&2; exit 2; fi
done
mkdir -p "$ROOT/runs"
PYTHONPATH="$STAGE/scripts:$STAGE/tests/implementation:$RUNTIME/tools:$RUNTIME/experimental/branching_context/src:$RUNTIME/experimental/branching_context/scripts" \
  exec bash "$RUNNER" "$STAGE" "$RUN_DIR" /usr/bin/env \
  PYTHONPATH="$STAGE/scripts:$STAGE/tests/implementation:$RUNTIME/tools:$RUNTIME/experimental/branching_context/src:$RUNTIME/experimental/branching_context/scripts" \
  "$PYTHON" -B "$STAGE/scripts/verify_cli_workflow_numerics.py" \
  --runtime-root "$RUNTIME" --profile "$PROFILE" --campaign "$CAMPAIGN" \
  --oracle-binary "$ORACLE" --original-build-receipt "$ORIGINAL_BUILD" \
  --build-result "$SEGMENTED_BUILD" --input-verification "$INPUT_PROOF" \
  --output "$AUDIT_OUTPUT" --outputs 8 --continue-outputs 8