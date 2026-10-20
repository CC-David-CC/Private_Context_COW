# Run the native Context COW browser service

This guide describes the already-qualified, private llm60 Q8 setup. It is a
repeatable launch of the frozen source and executed profile on that host; it is
not a recipe for compiling the native drivers or creating a new qualified
profile. The build/profile pipeline still needs a maintained, reproducible
recipe. Do not substitute a path-redacted profile from the repository results.

The qualification used llm60 GPU 0 (RTX PRO 6000 Blackwell), the Qwen3.8
Flash-Next Q8_0 model and pack, 98,304-token capacity, 8,192-token prefill
chunks, FP16 KV, and the `consumed-greedy-ack-v1` native request protocol. Its
profile explicitly selects `GPU=[0]` and disables template thinking with
`enable_thinking=false`; the native UI uses deterministic greedy generation.
The frozen source commit is
`ac6b8e6d465d866a0ae341d9d37a28400aea582f`; its
executed profile identity is
`b62b4e1842ae4c9b5a0b9c6ea372cb9dcaa2333206adf4b2dd9b7a747f7323d1`, and the
profile file SHA-256 is
`14b3d16b9ba65e84e42c16352ffcd88304e4ff3ea2302a235d75ef1ecb164e59`.
The profile points to the existing model, Q8 pack and expert-profile files on
llm60; those large inputs and native executables are outside Git.

The browser run completed eight grouped checks against the real same-origin COW
HTTP server: health, authentication, empty namespace, prepare/answer, fork
isolation, graph, math settings and trace. The coordinator observed one active
request maximum on GPU 0. This is a bounded serving/browser qualification, not
a speed measurement, independent numerical comparison, long-output test, or
completion of all G1–G5 gates. See the
[executed Q8 browser report](../../experimental/branching_context/evidence/NATIVE-Q8-BROWSER-2026-10-15.md)
and its [machine-readable report](../../experimental/branching_context/evidence/NATIVE-Q8-BROWSER-2026-10-15.json).

## Start the frozen service

From Windows PowerShell, enter the llm60 shell:

```powershell
ssh -t llm-60
```

Run the following in that remote shell. They use the frozen checkout, the
already executed profile, and the installed dependency directory verified on
that host. The remote interpreter is Python 3.12.3. The local Windows machine
is only the SSH/browser driver; use `python`, not `python3`, for any local
Python commands.

```bash
cd /home/dflanag3/branching-g3/namespace-q8-browser-ac6-20261014/source-ac6/tree
export PYTHONPATH="$PWD/experimental/branching_context/src:$PWD:/home/dflanag3/issue-2026-20261012/deps${PYTHONPATH:+:$PYTHONPATH}"

umask 077
RUN_DIR="$HOME/cow-native-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$RUN_DIR"
printf 'Keep this run directory for later launches: %s\n' "$RUN_DIR"
read -rsp 'COW API key: ' COW_API_KEY
printf '\n'
printf 'SSH shell PID for owner-scoped exit checks: %s\n' "$$"

python3 -m serve.cow_http \
  --profile /home/dflanag3/spatial-artifacts/namespace-q8-http-luna-20261014/gpu-8k-512-128-v1/result/profile.json \
  --store "$RUN_DIR/store" \
  --host 127.0.0.1 --port 8099 \
  --ram-gib 96 --disk-gib 128 \
  --api-key "$COW_API_KEY" --text
```

The `read -s` prompt keeps the key out of shell history. The server CLI accepts
the key as an argument, so it may be visible in process inspection on the host
while the process runs, subject to host `/proc` policy. The browser stores the
entered key in that browser's local storage. `/health` is public; other COW
routes require the bearer key.
Keep the service bound to loopback. Create a new private Store directory on
the first launch; the native UI creates canonical COW namespaces and snapshots
there, not in the ordinary SQLite chat database. Keep the printed `RUN_DIR`
path: after stopping the server, a later launch should set `RUN_DIR` to that
same recorded path, verify that `"$RUN_DIR/store"` exists, and reuse it with
the unchanged profile. Do not rerun the timestamped `mkdir` command for a
retained Store.

For a later SSH session, enter the exact directory printed by the first run
and verify the Store before rerunning the same service command:

```bash
read -rp 'Existing COW run directory: ' RUN_DIR
test -d "$RUN_DIR/store" && printf 'Existing Store found\n'
read -rsp 'COW API key: ' COW_API_KEY
printf '\n'
printf 'SSH shell PID for owner-scoped exit checks: %s\n' "$$"
```

Continue only if the check prints `Existing Store found`; correct the path
before proceeding if it does not. Then run the same `python3 -m serve.cow_http`
command shown above. This keeps
canonical history at the same path; it does not create a new Store.

The pool settings above match the qualified run. They cap the derived serialized
checkpoint pool at 96 GiB RAM and 128 GiB disk, with a 16 GiB host-available
memory floor for admission. Pool accounting covers checkpoint payloads and
frontier extents, not Python overhead, model weights, the native process, or GPU
allocations. In the browser run, peak accounted RAM was 9,186,322,784 bytes and
disk checkpoint files used 1,050,826,714 bytes; no pressure eviction occurred.
Those figures are observations from that run, not a promise for another session.

When the command prints its `ready` JSON, check that it reports the expected
execution-profile ID. In a second local Windows PowerShell window, open the
loopback tunnel:

```powershell
ssh -N -L 8099:127.0.0.1:8099 llm-60
```

Keep both terminals open. Open `http://127.0.0.1:8099/` in the local browser.
In **About → Settings**, enter the same API key and save it. The UI polls the
authenticated `/cow/status` endpoint and reports Ready/Working/Queued from
actual coordinator metrics. A response with `service: context-cow` from
`/health` confirms that this is the native COW UI, not ordinary Strata chat.
Make one request at a time; the single GPU worker serializes admitted work and
reports any waiting request as queued.

For a direct check, open a second SSH shell on llm60 while the foreground
service runs. Shell variables do not carry into that new terminal, so enter
the same key again at its prompt. `/health` needs no key; `/cow/status` does:

```bash
curl --fail --silent --show-error http://127.0.0.1:8099/health
read -rsp 'COW API key: ' COW_API_KEY
printf '\n'
curl --fail --silent --show-error \
  -H "Authorization: Bearer $COW_API_KEY" \
  http://127.0.0.1:8099/cow/status
unset COW_API_KEY
```

For an owner-scoped process exit check, use a second llm60 shell while the
service is running. Substitute the SSH shell PID printed before startup. The
first command shows its direct child (the Python service); the second shows
that service's direct child (the native worker). Read the PIDs from each
output and keep these variables in the second shell:

```bash
read -rp 'SSH shell PID printed before startup: ' SSH_SHELL_PID
ps -o pid=,ppid=,comm= --ppid "$SSH_SHELL_PID"
read -rp 'Python server PID shown above: ' SERVER_PID
ps -o pid=,ppid=,comm= --ppid "$SERVER_PID"
read -rp 'Native worker PID shown above: ' NATIVE_PID
```

After Ctrl+C returns in the first terminal, run this in the second shell:

```bash
ps -p "$SERVER_PID,$NATIVE_PID" -o pid=,ppid=,comm=
```

The final command should print no rows after both processes exit. This check
does not search for or stop unrelated processes.

To stop, press Ctrl+C in the remote foreground terminal and wait for the
process to return, then run `unset COW_API_KEY`. Startup and shutdown close the
coordinator and its native worker. The qualification checked that the server,
native worker and browser driver processes had exited. Do not kill unrelated
model/server processes or restart the ordinary chat service as part of this
workflow. The private Store remains at `$RUN_DIR/store` after shutdown; keep or
remove that exact run directory only when its history is no longer needed.
Reusing it is the continuity path; the real-browser qualification above did
not exercise a browser reload or server restart. Separate native restart
evidence covers a different 65,536-token prefix, 8,335-token suffix and
2,048-output disk/process
resume profile, with 13 sampled frontier comparisons; it is not a restart
qualification of this browser workflow and has a different profile identity
(`2cd8857a454d5645e8caf5a0551213bcfbf99ee3b57622ef738f60b62f4f25ee`). See the
[native Q8 restart summary](results/native-resident-20261014/digest-reuse-q8-restart-summary.json).

## What this does not reproduce

This command reuses the frozen source, executed Q8 profile and existing native
executables. `serve.cow_http` validates the profile's build/model fingerprints
at startup, but the repository does not yet provide one maintained recipe that
rebuilds the native request driver binaries and receipt, prepares the matching
98,304/8,192/FP16 profile and text-renderer metadata, and qualifies that new
profile. The comparison guide's `prepare_speed_profile.py` creates a different
12,288-token unqualified comparison profile; it is not interchangeable.
Likewise, the older replay-mode steps and screenshots in the
[chat workspace guide](CHAT_WORKSPACE_GUIDE.md) describe the SQLite-backed
ordinary UI, not this native COW service.

Use the frozen profile as-is for this bounded llm60 launch. A different model,
pack, binary, source revision, environment, execution setting or GPU requires
its own profile and verification; this guide does not claim those are
qualified.
