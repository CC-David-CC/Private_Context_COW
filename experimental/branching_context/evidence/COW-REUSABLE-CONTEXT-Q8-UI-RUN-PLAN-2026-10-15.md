# Q8 reusable-context browser run: preparation checkpoint

This is a preparation receipt, not a qualification. No model inference, native worker, COW HTTP service, ordinary chat service, or GPU job was started during this preparation. Root review is required before the proposed browser run.

## Prepared inputs

The immutable Linux runtime overlay is `/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/api`, extracted from source commit `981aaeaaabba0b49897b1e4256e3d8ee2931583d`. Its 593-file manifest is `runtime-file-manifest.json` (SHA-256 `9bacb3d814e8622d58197b76e9729143510cfc35ccf1037d9ce94f803df65bd7`); extracted source bytes total 7,802,674, with zero source-hash mismatches. The runtime source manifest is `b7de2843ba96ab2d0eecde3d253814d0843c29bf2aca44bdf09db0828e625066`.

The derived profile is `/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/derived-profile/profile.json`, SHA-256 `11eeda86f34fa2b53b824f273a37ad743bbf6c0b88e668f3b9c03073b23a13b6`. Its execution profile is `f39df09d619ee55032a199c0a089fa964ef1b27f48929bca11f6e70aff9a07c7`, model contract `fc6d0d643e4b9ae020964f8c4e5006c0da7199b19cce5285c573023dbe82ca46`, capacity 98,304, prefill chunk 8,192, FP16. `f39…` is the proposed caller profile; the earlier `b62b4e1842ae4c9b5a0b9c6ea372cb9dcaa2333206adf4b2dd9b7a747f7323d1` is only its base and must not be used for this run. The preflight checked model-file stats for 11 files and reused recorded hashes; it did not hash model weight bytes. The profile has 283 native-build hash entries, including 110 verified runtime Python entries.

The UI fixture is `/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/fixture.json`, SHA-256 `384984f5ef1f30a22edc3a1a80ed2830e59b8a56331e8c397181f482f8ecd16b`; its source attachment is `cow-code-context.txt` (36,264 bytes; SHA-256 `239869c177abc706a8e6a4141cb91b696e9fafe75d28d568fc2671304e37143b`). It records a user-role, code-only root of 8,189 tokens and two independent question suffixes of 594 and 596 tokens. The renderer identity is `555ee456490815ec5703830a444b5cf1bc8a0ac1fed7cba23b027471e9c95837`. Fixture UI hashes for `app.js`, `workspace.js`, and `workspace-graph.js` match the staged source. A separate CPU/headless Chromium check executed the actual file-framing helper for the full body and a Unicode/multiline/backtick adversarial case; it did not open an HTTP server or use a GPU.

G0 on the exact staged source passed 51 tests, with 0 failures, errors, skips, or xfails. The verified command wrapper exited 0 and the test command exited 0. The Q8 capacity campaign terminal receipt reports PASS, all six decode files complete, no projected decode bytes remaining, and 13,703,184,384 free bytes versus a 10,737,418,240-byte floor. Recheck free space immediately before any later run.

The machine-readable receipt beside this file contains exact paths and hashes. The full preparation record is `/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/preparation-handoff.json` (SHA-256 `925c4d31a78a4714daff317150b4343d1caf126e6cf67f0fa93a9ed2d9222598`).

## Proposed reviewed run

Use a new private run directory, store, namespace, key file, and Chromium temporary profile. Bind the service to `127.0.0.1` only. Create the key file with mode `0600`; never put its contents in shell history, a command argument, a report, or a receipt. `cow_http.py` currently accepts `--api-key` but has no key-file option. To preserve the no-inline-key requirement without editing runtime source, use a short-lived Python launcher that reads the key file and injects the value in-process; its OS command line contains only the launcher code and key-file path. Root should review this wrapper before execution. `cow_native_smoke.py` reads the same key through `COW_API_KEY` and omits it from its report.

The following is the proposed launcher shape, not an executed command. Use an unused port and fresh paths after a live process and disk-space preflight. Generate the key file under the private run directory with restrictive umask; verify `stat` reports mode `600`. Keep the service process in a named private tmux session so its ready record and shutdown status are retained.

```sh
umask 077
RUN=/home/dflanag3/spatial-artifacts/cow-ui-q8-8189-20261015
mkdir -m 700 "$RUN"
mkdir -m 700 "$RUN/chromium-tmp"
export KEY_PATH="$RUN/api-key"
python3 -c 'import os,secrets; p=os.environ["KEY_PATH"]; fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(fd,secrets.token_hex(32).encode()+b"\n"); os.close(fd)'
stat -c '%a %n' "$KEY_PATH"
```

Create this small launcher in the private run directory. It contains no secret; it reads the mode-600 file and invokes the existing `cow_http.main()` interface. Its process command line contains only the key-file path, never the key value.

```sh
cat > "$RUN/cow_http_keyfile.py" <<'PY'
import sys
from pathlib import Path

arguments = sys.argv[1:]
key = Path(arguments.pop(0)).read_text(encoding="utf-8").strip()
if not key:
    raise SystemExit("empty key file")
sys.argv = ["cow_http.py", *arguments, "--api-key", key]
from serve.cow_http import main
main()
PY
chmod 700 "$RUN/cow_http_keyfile.py"
API=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/api
tmux new-session -d -s cow-ui-q8-8189 \
  "umask 077 && cd '$API' && PYTHONPATH='$API/src:$API' python3 '$RUN/cow_http_keyfile.py' '$RUN/api-key' --profile '/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/derived-profile/profile.json' --store '$RUN/store' --host 127.0.0.1 --port 8099 --text"
```

Before starting the process, root should confirm no conflicting COW/ordinary chat listener or GPU job, verify the runtime manifest again, and check `df -B1 /`; require the projected durable checkpoint and outputs to leave at least the agreed 10 GiB floor. Current Q8 terminal free space was 13,703,184,384 bytes with zero decode bytes remaining; that is historical and must be rechecked. The tmux command must be validated by root with a harmless `--help`/wrapper preflight before a GPU launch. Do not continue unless `/health` is loopback-only and the printed ready record carries the exact `f39…` ID. Check the service's actual durable receipts after the run as well; the smoke's `--server-profile-id` flag is caller-supplied and is not proof of identity.

After root approves the profile and free-space preflight, the intended service entry point is `serve/cow_http.py --profile <derived-profile>/profile.json --store <new-run>/store --host 127.0.0.1 --port <unused-loopback-port> --text`, with the reviewed key-file launcher supplying `--api-key`. Require its ready receipt to report the exact `f39…` execution profile. Do not use the normal Strata chat server.

Once that service is ready, run the existing real-HTTP/CDP smoke against it, with a fresh output directory and temporary browser profile. The secret is read from the file into the smoke process environment; it is not included in argv:

```sh
IFS= read -r COW_API_KEY < "$RUN/api-key"
cd "$API"
COW_API_KEY="$COW_API_KEY" python3 serve/web/cow_native_smoke.py \
  --base-url http://127.0.0.1:8099 \
  --output-dir "$RUN/browser" \
  --tmp-root "$RUN/chromium-tmp" \
  --browser /snap/bin/chromium \
  --reusable-context \
  --reusable-fixture /home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/fixture.json \
  --server-profile-id f39df09d619ee55032a199c0a089fa964ef1b27f48929bca11f6e70aff9a07c7 \
  --context-file /home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/cow-code-context.txt \
  --context-name cow-code-context.txt \
  --output-limit 128 --inference-kind real
unset COW_API_KEY
```

The reusable-context smoke opts into the actual Save-as-context control, captures the browser's `/cow/text/context` request body, checks the durable code-only root, asks A, forks the whole saved root from the graph, and asks B. It captures matching request/response exchanges and checks branch isolation and native forward-work counters. Its reusable-context path sets zero resumes and never forces an answer complete; each ask is bounded to 128 output tokens, and an active controller must be reported honestly. Its report's caller profile label is not sufficient: require both durable receipt profile/model identities to equal the fixture and reviewed profile.

After the smoke closes Chromium and the service/coordinator are confirmed stopped, run the read-only disk audit (no model load/native work) against the exact store and report:

```sh
cd "$API"
python3 experimental/branching_context/scripts/audit_cow_browser_context_store.py \
  --repo-root "$API" \
  --store-dir "$RUN/store" \
  --profile /home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/derived-profile/profile.json \
  --browser-report "$RUN/browser/report.json" \
  --fixture /home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/fixture.json \
  --output "$RUN/offline-audit.json"
```

Retain the raw ready record, smoke report, captured request bodies, offline audit, exact stdout/exit receipts, and before/after source/store hashes. The offline audit is separate from the browser's own evidence and runs only after service shutdown. Do not claim production numerical gates, full backend prefix reuse, or performance from this UI workflow. Do not launch until root reviews this checkpoint and confirms the final disk margin and process state.

## Not run

No GPU/native HTTP/UI process was launched. There is no CDP network capture, real receipt, fresh-namespace result, actual durable store audit, branch-isolation runtime result, prefix-work result, or screenshot from a live server yet. The framing check was local-file-only. This is a ready-to-review plan, not a PASS for the proposed workflow.
