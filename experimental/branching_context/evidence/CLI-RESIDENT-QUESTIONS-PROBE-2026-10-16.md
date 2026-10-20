# CLI resident code-context probe

This harness is a bounded real-service CLI measurement, not a numerical or quality oracle. It loads one context-only root, forks three independent question branches from that same snapshot, then asks each question with the pinned COW CLI. It verifies exact answer token IDs and text from the machine-result envelope against stored lineage, complete controller state, source/sibling immutability, native forward/prefill deltas, saved-prefix reuse, and the root native capture/restore trace.

The service lifecycle uses the pinned `_Campaign` helper so startup import witnesses and shutdown receipts come from the same process owner as the other CLI probes. The helper-generated API key is passed to CLI child processes through `COW_API_KEY`; it is not put in CLI arguments or retained in artifacts. Nested service stop/cleanup records and any error text are redacted before the result file is written.

The current target is the frozen v16 runtime/profile pair and the existing `serving_tooling_template` context fixture, whose retained CPU tokenization has 8,087 tokens. The harness requires the measured root token count to match the explicit `--expected-root-tokens` value (default 8,087). Each answer uses `--max-tokens` as an honest ceiling (default 128); EOS may stop earlier and the receipt reports actual output counts.

Per question, the output records native `generation_seconds` and `request_seconds`, full CLI process duration, native counter deltas, and client-observed request-to-first-token/request-to-durable timings from the existing NDJSON observer. The observer timestamps lines at the CLI response-reader boundary; it does not provide a model-internal TTFT. Root-load CLI duration includes tokenization, prefill, publication, and capture; the native trace exposes capture/restore events and handle reuse but no capture-only duration.

Example llm-60 invocation, with a fresh external run directory:

```sh
R=/home/dflanag3/spatial-artifacts/seed-stop-combined-20261016-v1/combined-cpu-v16/source
P=/home/dflanag3/spatial-artifacts/seed-stop-combined-20261016-v1/combined-cpu-v16/profile-v4/profile.json
C=/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1/fixtures-v1/2-serving_tooling_template/context.txt
H=/home/dflanag3/spatial-artifacts/cow-cli-resident-probe-20261016-v1/source/experimental/branching_context/scripts/run_cli_resident_questions_probe.py
W=/home/dflanag3/spatial-artifacts/cow-cli-resident-probe-20261016-v1/wrapper-v1
RUN=/home/dflanag3/spatial-artifacts/cow-cli-resident-probe-20261016-v1/campaign-v1
O=20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28
CH=256fc858336f316b8084d9eded1ab278fd5f87510056dc2047df44b2a42b1d8d
bash /home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/run-command.sh \
  "$R" "$W" \
  /usr/bin/env PYTHONPATH="$R:$R/tools:$R/experimental/branching_context/src:$R/experimental/branching_context/scripts" \
  /home/dflanag3/src/Strata/.venv/bin/python -B "$H" \
  --runtime-root "$R" --profile "$P" --context-file "$C" \
  --run-dir "$RUN" \
  --expected-root-tokens 8087 --context-sha256 "$CH" \
  --max-tokens 128 --observer-sha256 "$O"
```

The actual launch should use unique wrapper `NEW_RUN_DIR` (`$W`) separate from `--run-dir` (`$RUN`), and root must perform GPU-idle/resource preflight and launch. No GPU run is included in this source checkpoint.

The frozen helper import is `run_cli_workflow_probe.py` from the same v16 scripts directory on `PYTHONPATH`; the harness records that helper’s path and SHA-256 in `result.json`. Use a fresh `XDG_STATE_HOME` for every attempt. The helper starts the HTTP service with an owned random key, and the CLI subprocess receives it only through `COW_API_KEY`.
