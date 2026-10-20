#!/usr/bin/env bash
set -u

if [[ -z "${RUN:-}" ]]; then
  RUN="$(cd "$(dirname "$0")" && pwd)"
  export RUN
fi
: "${COW_RUNTIME_API:=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/api}"
: "${COW_FIXTURE:=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/fixture.json}"
: "${COW_CONTEXT_FILE:=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/cow-code-context.txt}"
if [[ -e "$RUN/browser" || -L "$RUN/browser" ]]; then
  echo "smoke output directory must not exist before launch" >&2
  exit 64
fi

cd "$COW_RUNTIME_API" || exit 90
IFS= read -r COW_API_KEY < "$RUN/api-key"
export COW_API_KEY
set +e
python3 "$RUN/cow_native_smoke.py" \
  --base-url http://127.0.0.1:8099 \
  --output-dir "$RUN/browser" \
  --tmp-root "$RUN/chromium-tmp" \
  --browser /snap/bin/chromium \
  --reusable-context \
  --reusable-fixture "$COW_FIXTURE" \
  --server-profile-id f39df09d619ee55032a199c0a089fa964ef1b27f48929bca11f6e70aff9a07c7 \
  --context-file "$COW_CONTEXT_FILE" \
  --context-name cow-code-context.txt \
  --output-limit 128 --max-resumes 0 --inference-kind real \
  >"$RUN/browser.stdout.log" 2>"$RUN/browser.stderr.log"
rc=$?
unset COW_API_KEY
printf '%s\n' "$rc" > "$RUN/browser.exit"
exit "$rc"
