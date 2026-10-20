# Run the private CLI trial on llm 60

This is the server-owner recipe for the current pinned Linux installation.
Friends run only the terminal client. They do not download the model. The
ordinary Strata chat server remains off; this starts the dedicated COW service.

## Start the service in a retained terminal

Connect with `ssh llm-60`. Check `nvidia-smi` and the current handoff before
starting: another job must not already own GPU 0. Run the following in a new
tmux session named `cow-friend`. Tmux keeps this terminal available after SSH
disconnects; reconnect and attach to the same session rather than starting a
second server.

```bash
tmux new-session -s cow-friend
```

In that session, enter:

```bash
umask 077
BASE=/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1
REPO=$BASE/cold-cache-v1/repo
PROFILE=$BASE/profile-q8-cold-cache-v1/profile.json
COWPY=/home/dflanag3/src/Strata/.venv/bin/python
TRIAL=/home/dflanag3/spatial-artifacts/cow-friend-cli-session
install -d -m 700 "$TRIAL"
cd "$REPO"
flock -F -n /home/dflanag3/branching-g3/native-client-speed.gpu0.lock \
  env PYTHONPATH="$REPO:$REPO/experimental/branching_context/src" \
  "$COWPY" -m serve.cow_http --profile "$PROFILE" --store "$TRIAL/store" \
  --host 127.0.0.1 --port 8099 --text --ram-gib 96 --disk-gib 128
```

Wait for the JSON `ready` record. Model startup can take a while. If the lock
is busy, the command exits; do not bypass it. The profile validates this
machine's pinned source, engine, GPU, model and packs. It is not a portable
profile for another machine. Select `profile-q4-cold-cache-v1/profile.json`
instead only when intentionally running the separately qualified Q4 model.

The Store is separate from all qualification Stores. Restart with the same
Store path to retain the trial's histories and durable numerical checkpoints.
Do not copy changing campaign Stores into this location. No database server is
required; the service uses the existing canonical filesystem store.

This launch omits the optional audit trace. It does not disable checkpoint
hashes or durability. RAM96GiB/disk128GiB are pool limits, not preallocated
storage or a promise of available disk capacity. At the last inspection
llm-60 had about12GiB filesystem space free. Check `df -h` before adding large
codebases: a many-tree long-context trial needs additional space or a separately
reviewed retention/cleanup action. A small initial trial can use the current
space; do not advertise128GiB of free storage.

## Give the friend a connection

Provide an authorized SSH destination and COW port8099. `llm-60` is a private
SSH alias on this controller, so a friend's computer needs the actual
authorized host/address or its own SSH alias. The client quick start shows
the tunnel and commands. Do not send private SSH keys.

```powershell
ssh -N -L 18080:127.0.0.1:8099 YOUR_AUTHORIZED_SSH_HOST
```

In a separate terminal, the friend uses the extracted package:

```powershell
python tools/cow_cli.py --url http://127.0.0.1:18080 status
```

The dedicated service stays bound to Linux loopback. This trial uses a shared
workspace; namespaces organize trees but are not separate user accounts. Use
contexts intended for this shared trial. A public bind requires an API key,
but exposing a public service is not part of this handoff.

## Stop without interrupting the native worker first

Use a second SSH terminal to inspect the exact Python HTTP process with the
trial Store path in its command:

```bash
ps -u dflanag3 -o pid,ppid,args
```

Send SIGINT to that HTTP PID only:

```bash
kill -INT HTTP_PID
```

Replace `HTTP_PID` with the number you inspected. Do not signal every Strata
process or the whole process group. The HTTP process closes its own native
worker, drains publication and releases the GPU lock. Wait for its terminal
to return, then check `nvidia-smi`. A terminal-wide Ctrl+C can also signal a
native child; use the leader-only stop above for this prototype.

Attach to the retained terminal with `tmux attach-session -t cow-friend` when
checking startup or shutdown. The normal CLI test harness uses the same
service module and has separately tested this leader-only shutdown behavior.
