# Q8 reusable-context GPU retry plan

Status: prepared for review only. No model service, ordinary chat service, GPU job, or native backend was started for this plan. The resident latency arm has scheduling priority; wait for root to schedule this retry.

The earlier retry directory ending in retry-20261015 is preserved as a setup failure: its browser output directory already existed, so the harness exited before browser startup or any HTTP/model request. The server had been ready and then exited 143 during cleanup; the native child disappeared, but no numeric native exit receipt was captured. This is not a browser or model result. The new retry directory ending in retry-20261015-v2 is fresh.

## Pinned inputs and scope

The tested application source is root code HEAD `30e52e894156723e0cdc16a0641d0f50afe1dc15`; root HEAD `c4ca754206df6af20a636118af2fd15121b20489` adds evidence-only changes. The 593-file staged runtime at `/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/api` derives from `981aaeaaabba0b49897b1e4256e3d8ee2931583d`, manifest SHA-256 `9bacb3d814e8622d58197b76e9729143510cfc35ccf1037d9ce94f803df65bd7`. Runtime source manifest: `b7de2843ba96ab2d0eecde3d253814d0843c29bf2aca44bdf09db0828e625066`.

Use the derived profile `/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/derived-profile/profile.json` (SHA-256 `11eeda86f34fa2b53b824f273a37ad743bbf6c0b88e668f3b9c03073b23a13b6`), execution profile `f39df09d619ee55032a199c0a089fa964ef1b27f48929bca11f6e70aff9a07c7`, model contract `fc6d0d643e4b9ae020964f8c4e5006c0da7199b19cce5285c573023dbe82ca46`, capacity 98,304, prefill chunk 8,192, FP16. This is not the prior base profile. It validates model file stats and reuses recorded hashes; it does not rehash model weights.

Fixture `/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/fixture.json` SHA-256 `384984f5ef1f30a22edc3a1a80ed2830e59b8a56331e8c397181f482f8ecd16b`; attachment is 36,264 bytes with SHA-256 `239869c177abc706a8e6a4141cb91b696e9fafe75d28d568fc2671304e37143b`. The user-role code-only context is 8,189 tokens, with independent A/B suffixes of 594/596 tokens. UI renderer identity is `555ee456490815ec5703830a444b5cf1bc8a0ac1fed7cba23b027471e9c95837`. Run with output limit 128 and max resumes 0, against GPU 0, RAM pool 96 GiB, disk pool 128 GiB. The browser workflow is correctness-only, not a performance measurement.

Application/UI hashes must match the execution evidence JSON. Keep the ordinary chat service off and use a fresh private store, namespace, Chromium profile, and run directory.

## Preflight and service start

Immediately before starting, confirm the resident arm is finished and root has scheduled this retry. Check that GPU 0 is unowned, port 8099 is unused, and filesystem free space is above the 10 GiB floor after projecting the run's checkpoint/reference writes. Current observations are historical only; repeat them live.

The first retry directory contains the preserved preflight failure. Use a new directory:

```text
/home/dflanag3/spatial-artifacts/cow-ui-q8-8189-retry-20261015-v2
```

Create it and its store and Chromium temporary-profile directories with mode 0700; do not create the browser output directory because the smoke harness requires that path to be absent at startup. Create the API key file with mode 0600. Keep the secret out of argv, shell history, reports, and receipts:

```sh
RUN=/home/dflanag3/spatial-artifacts/cow-ui-q8-8189-retry-20261015-v2
umask 077
mkdir -m 700 "$RUN"
mkdir -m 700 "$RUN/store" "$RUN/chromium-tmp"
KEY_PATH="$RUN/api-key" python3 -c 'import os,secrets; p=os.environ["KEY_PATH"]; fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(fd,secrets.token_hex(32).encode()+b"\n"); os.close(fd)'
stat -c '%a %n' "$RUN" "$RUN/api-key"
```

Install the reviewed helpers from the preflight directory. The browser driver is the separately tested external harness, not the older copy in the immutable API tree. Verify its digest before installing it:

```sh
PRE=/home/dflanag3/spatial-artifacts/cow-ui-q8-8189-20261015-preflight-v2/reviewed
SMOKE=/home/dflanag3/spatial-artifacts/cow-ui-reusable-review-20261015/combined-serving/serve/web/cow_native_smoke.py
test "$(sha256sum "$SMOKE" | cut -d ' ' -f 1)" = 45bf00c4ed8ec95d414664e00c112ab510699c13d5147141d79da724c83e82b3
(cd "$PRE" && sha256sum --check SHA256SUMS)
install -m 700 "$PRE/cow_http_keyfile.py" "$RUN/"
install -m 700 "$PRE/run-cow-http-pid.sh" "$RUN/"
install -m 700 "$PRE/start-cow-service.sh" "$RUN/start-service.sh"
install -m 700 "$PRE/signal-cow-server.py" "$RUN/"
install -m 700 "$PRE/cow_fresh_reference.py" "$RUN/"
install -m 700 "$PRE/scan_cow_ui_artifacts.py" "$RUN/"
install -m 700 "$SMOKE" "$RUN/cow_native_smoke.py"
install -m 700 "$PRE/run-cow-ui-smoke.sh" "$RUN/"
test ! -e "$RUN/browser"
bash -n "$RUN/start-service.sh" "$RUN/run-cow-http-pid.sh" "$RUN/run-cow-ui-smoke.sh"
python3 -c 'from pathlib import Path; import sys; paths=[Path(sys.argv[1])/n for n in ("start-service.sh","run-cow-http-pid.sh","run-cow-ui-smoke.sh","cow_http_keyfile.py")]; assert all(b"\r\n" not in p.read_bytes() for p in paths), "CRLF in installed launcher"' "$RUN"
```

Do not change the immutable API tree.

The launch bundle is a new directory at /home/dflanag3/spatial-artifacts/cow-ui-q8-8189-20261015-preflight-v2/reviewed; the older preflight directory was left untouched. Its SHA256SUMS is 10559c7fd1453b1fe6057a279c2459bb668572319ac4bc4bb910bd0e6eb86fce. The prepared bundle was installed from that exact directory into a fresh install-check directory by verify-cow-ui-preflight-install.sh; source and destination hashes matched, shell syntax/Python compilation passed, and the browser output remained absent. The verified run-command-v2 receipts are at /home/dflanag3/spatial-artifacts/cow-ui-launcher-tests-20261015/install-receipt-v2/ (command.exit 0, wrapper.exit 0, terminal COMPLETE); stdout is 13ad388da691dd803f94cb92c836b990435e5184bfcaf877d2fbcaeb0daebc8c.

Start with a direct argv invocation, without nested shell command text:

```sh
tmux new-session -d -s cow-ui-q8-8189-retry-v2 bash /home/dflanag3/spatial-artifacts/cow-ui-q8-8189-retry-20261015-v2/start-service.sh
```

The start script fixes `RUN` to that fresh path and executes the PID-recording wrapper. The wrapper writes the exact Python service PID to `server.pid`, runs the service on loopback only, and records the real Python exit in `server.exit`. It enables the mode-0600 key-file launcher and writes a native exit receipt from the original backend close path.

The PID launcher is covered by a CPU subprocess test that starts with SIGINT ignored, proves readiness, signals only the recorded Python PID, and observes a clean native-child return code of 0. The Python key-file entrypoint explicitly restores Python's default SIGINT handler before importing the service. Shell files are checked for LF line endings and with bash -n after transfer.

After health/profile checks, run the real browser smoke with this exact argument set (the shell variable contains the key, but the process argv does not):

```sh
RUN=/home/dflanag3/spatial-artifacts/cow-ui-q8-8189-retry-20261015-v2
API=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/api
RUN="$RUN" bash "$RUN/run-cow-ui-smoke.sh"
```

Keep the browser and request evidence under the fresh run directory. Require each client-consumed complete NDJSON acknowledgement, actual context/ask request-body capture, durable receipts, branch isolation, and truthful output counts.

Measure actual checkpoint growth and free space after the root, after A, and before B. Stop if the remaining projected checkpoint plus independent-reference bytes would cross the floor. The prior estimate is planning information, not a storage guarantee.

## Exact shutdown and post-shutdown checks

Do not send Ctrl-C to tmux, signal the process group, or use SIGTERM. Invoke the helper with the run path; it reads the recorded PID, verifies that the PID's argv contains this run's `cow_http_keyfile.py` and key-file path, then sends SIGINT to that Python PID only:

```sh
python3 /home/dflanag3/spatial-artifacts/cow-ui-q8-8189-retry-20261015-v2/signal-cow-server.py /home/dflanag3/spatial-artifacts/cow-ui-q8-8189-retry-20261015-v2
```

Wait for `server.exit` to contain 0, require a valid `server.pid`, require all `native-exits.jsonl` rows to contain return code 0, and verify the recorded Python/native PIDs are absent. The fresh-reference helper fails closed if the server PID receipt is missing, invalid, or still alive. It validates each checkpoint manifest checksum, snapshot/profile/model identity, payload length and SHA-256, and finite saved frontier before comparing the root and A/B final frontiers with the independent fresh oracle. It compares exact token IDs and saved frontier bytes; browser streaming does not expose full logits.

Run the read-only checkpoint audit only after native/service shutdown:

```sh
API=/home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/api
PYTHONPATH="/home/dflanag3/issue-2026-20261012/deps:$API/scripts:$API/src:$API" \
  python3 "$API/scripts/audit_cow_browser_context_store.py" \
  --repo-root "$API" --store-dir "$RUN/store" \
  --profile /home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/derived-profile/profile.json \
  --browser-report "$RUN/browser/report.json" \
  --fixture /home/dflanag3/branching-g3/cow-reusable-context-runtime-20261015/fixtures/attempt-5/fixture.json \
  --output "$RUN/offline-audit.json"
```

Then run the key scanner over retained reports, requests, browser traces, reference outputs, audit receipts, and logs; require zero matches without printing the key. The scanner excludes the key file and checkpoint store payloads, and includes run-root JSON/NDJSON/text receipts. Preserve raw exit receipts, report, requests, audit, disk measurements, source/profile hashes, and all failures. Do not describe this workflow as a G0/model qualification or a performance result.
