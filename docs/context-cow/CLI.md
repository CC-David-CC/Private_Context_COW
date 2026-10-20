# Context COW from a terminal

`tools/cow_cli.py` is a small Python standard-library client for an already
running COW service. It runs on your Windows machine; it does not install or
start the Linux model engine, and it does not use the ordinary OpenAI-compatible
`/v1/chat/completions` route. The service and its native model remain on Linux.
Use the Windows client-Python setup in
[the client quick start](CLI-CLIENT-README.md#install-python-on-windows) and run it with
`python` from the repository root.

## Connect to an existing service

If the service listens on Linux loopback port 8099, open a PowerShell window and
keep this SSH tunnel running:

```powershell
ssh -N -L 18080:127.0.0.1:8099 llm-60
```

In a second PowerShell window, set the client URL and run a read-only check:

```powershell
$env:COW_URL = "http://127.0.0.1:18080"
python tools/cow_cli.py --url $env:COW_URL status
python tools/cow_cli.py --url $env:COW_URL namespaces
```

When the service requires an API key, set it in the environment without putting
it in the command line or shell history:

```powershell
$env:COW_API_KEY = Read-Host "COW API key"
python tools/cow_cli.py --url $env:COW_URL status
```

The key is sent as an HTTP Bearer credential through the SSH tunnel. For a
direct remote URL, use HTTPS. The CLI refuses non-loopback plain HTTP URLs.

## Start a context and ask a question

Create one namespace, load a base context, then fork it before asking separate
questions. Each fork starts from the same idle context snapshot, so the first
answer on one branch does not become part of the other branch. Choose branch
names beginning with a letter. The example assumes the namespace receipt
returned `0123456789abcdef0123456789abcdef`.

```powershell
python tools/cow_cli.py --url $env:COW_URL namespace --title "renderer notes"
python tools/cow_cli.py --url $env:COW_URL load `
  --namespace-id 0123456789abcdef0123456789abcdef `
  --branch b-renderer-base --file .\renderer-context.txt
python tools/cow_cli.py --url $env:COW_URL fork `
  --source-branch b-renderer-base --name b-renderer-a
python tools/cow_cli.py --url $env:COW_URL fork `
  --source-branch b-renderer-base --name b-renderer-b
python tools/cow_cli.py --url $env:COW_URL ask `
  --branch b-renderer-a --text "Summarize the main constraint." --max-tokens 128
python tools/cow_cli.py --url $env:COW_URL ask `
  --branch b-renderer-b --text "List the main edge case." --max-tokens 128
python tools/cow_cli.py --url $env:COW_URL history --branch b-renderer-a
```

If the namespace response is lost, use `retry-request` with its printed
operation ID. The client replays the saved creation request. Do not issue a new
namespace command or use a new ID for an uncertain creation.

`load` reads UTF-8 text and creates a closed context root. `ask` appends the
first user question to that idle context. The client prints generated text as
provisional output. It prints the durable receipt to stderr only after the
server sends a valid terminal `complete` event. Operation IDs also go to
stderr before a mutation; stdout is reserved for model text and read-only JSON.
The CLI configures stdout as UTF-8 and stderr as UTF-8 with escaped fallback,
so redirected output remains readable when Windows starts Python with a legacy
code page. Programs capturing either stream must decode it as UTF-8.

For a new chat that should start with a prepared user prompt, create the
namespace first, then use `prepare` with a new branch name. `answer` completes
that prepared prompt:

```powershell
python tools/cow_cli.py --url $env:COW_URL prepare `
  --namespace-id 0123456789abcdef0123456789abcdef `
  --branch b-first-question --text "Explain the ownership rule."
python tools/cow_cli.py --url $env:COW_URL answer --branch b-first-question --max-tokens 128
```

`prepare` is itself a durable mutation. If the process loses its response, use
the printed operation ID with `operation` before deciding what to do next.

## Continue, fork, and rewind

A controller marked active means the current assistant turn has not reached its
stopping boundary. A request can still durably publish a token-limited partial
answer and leave that controller active. After a successful completion receipt,
`continue` resumes that active turn; it does not add a new user message. While a
request is still streaming, its text remains provisional until the receipt
arrives. Ctrl+C closes the local response and exits 130 without claiming a
commit; check the printed operation ID before choosing a next action. A
completed stopped turn accepts another question through `append-user`:

```powershell
python tools/cow_cli.py --url $env:COW_URL continue --branch b-renderer-a --max-tokens 128
python tools/cow_cli.py --url $env:COW_URL append-user `
  --branch b-renderer-a --text "Now show the edge case." --max-tokens 128
```

To explore a saved point, fork from the current head or provide an explicit
snapshot ID from history:

```powershell
python tools/cow_cli.py --url $env:COW_URL branches
python tools/cow_cli.py --url $env:COW_URL fork `
  --source-branch b-renderer-a --name b-renderer-alt
python tools/cow_cli.py --url $env:COW_URL fork `
  --source-branch b-renderer-a --snapshot SNAPSHOT_ID --name b-renderer-old
```

The fork receipt identifies the created branch. Rewind moves one existing
branch to an explicit snapshot. The CLI reads the current revision immediately
before the mutation, and the service checks that revision atomically:

```powershell
python tools/cow_cli.py --url $env:COW_URL rewind `
  --branch b-renderer-a --snapshot SNAPSHOT_ID
```

If another writer changes the branch first, the service rejects the stale
revision. Re-read history and make a deliberate new command; the CLI never
silently retries the mutation.

## Recover an uncertain operation

Before sending a mutation, the client saves its exact request body, operation
ID and original source revision in a private local request journal. It then
prints the operation ID to stderr. Keep that ID if the connection drops.
You can query the current server process:

```powershell
python tools/cow_cli.py --url $env:COW_URL operation --operation-id op-PASTE_ID
```

Operation lookup is scoped to the coordinator process that accepted the request.
After a client restart, explicitly replay the saved original request:

```powershell
python tools/cow_cli.py --url $env:COW_URL retry-request --operation-id op-PASTE_ID
```

This uses the original body and revision. It does not fetch a newer branch head
or make a new request ID. If a checked completion receipt is already in the
local journal, it returns that receipt without contacting the service. Otherwise,
the server decides whether the original operation committed, is still running,
or cannot be completed. A new operation ID means new work. A stale revision is
an error, not permission to overwrite a newer branch.

After a server restart, `operation` may report unknown even when a durable Store
receipt exists. `retry-request` can still submit the original operation for
receipt recovery. Numerical-checkpoint eviction and interrupted, uncommitted
work have separate recovery limits; a retry is not a guarantee that the server
retains an unfinished answer. Read the actual returned error before deciding on
a new operation. No automatic inference retry is enabled.

Only when the current service reports a retained publication that can be
finished, retry publication using that same operation ID:

```powershell
python tools/cow_cli.py --url $env:COW_URL retry-publication --operation-id op-PASTE_ID
```

`retry-publication` only asks the current service to finish an already-retained
publication; it does not run inference again. It is different from replaying
the original request. Do not use a new operation ID to recover an uncertain
acknowledgment.

The journal contains submitted prompts or source code when these occur in a
request body. It contains no API key or authorization header. On Windows its
default folder is `%LOCALAPPDATA%\Strata\ContextCOW\requests`. On Linux it is
`$XDG_STATE_HOME/strata-context-cow/requests`, or
`~/.local/state/strata-context-cow/requests` if that variable is unset. Use
`--state-dir PATH` before the command, or `COW_CLI_STATE_DIR`, to choose another
private folder. Keep that folder across client restarts. Removing it removes
the client's exact retry intent, not the server's canonical history. POSIX
files use private modes; Windows ACL and machine power-loss durability have not
been qualified.

## Commands and limits

```text
status | contexts (alias: namespaces) | branches
history --branch BRANCH
namespace --title TITLE [--operation-id ID]
load --namespace-id ID --branch BRANCH --file UTF8_FILE [--operation-id ID]
prepare --namespace-id ID --branch NEW_BRANCH --text TEXT
ask --branch BRANCH --text TEXT [--max-tokens N]
answer | continue --branch BRANCH [--max-tokens N]
append-user --branch BRANCH --text TEXT [--max-tokens N]
fork --source-branch BRANCH [--snapshot ID] --name NEW_BRANCH
rewind --branch BRANCH --snapshot ID [--revision N]
operation --operation-id ID
retry-publication --operation-id ID
retry-request --operation-id ID
```

The command list above shows the basic text workflow. Per-generation sampling
controls and named server-owned [GBNF states](GRAMMAR.md) are also available on
capable workers. The opt-in [image CLI guide](vision-cli.md) describes mixed
text/image contexts, branching, restart and the tested sampler scope. These
controls do not execute proposed tools. The request journal is retry metadata and can include
submitted text; it is not a second canonical transcript or branch history.
`history`, `branches`, and `operation` read the service's canonical projection
and receipts. Save command output only if you want a local note; it is not a
server-side COW record.

## Executed validation

The terminal client passed the focused eight-case Linux socket/fixture suite.
Integrated helper and multi-client controls passed29 cases and G0 passed51.
The actual CLI subprocess entrypoint then passed Q4 and Q8 native service
campaigns on llm-60:7231-token file context, different512/488-token questions,
four128-output calls, fork, rewind and a clean process restart. Restarted tokens
and final canonical snapshot matched; continuations evaluated no saved prefix.
This campaign did not run a fresh-logit numerical oracle. See the Q4/Q8 CLI
machine results linked by the human and agent guide.

The campaign runner is
`experimental/branching_context/scripts/run_cow_cli_campaign.py`. It requires
`COW_RUNTIME_REPO` pointing to the exact source directory pinned by the supplied
profile, its API/scripts directories on `PYTHONPATH`, a compatible `--profile`,
the matching `--fixture-set` and a new private `--run-dir`. It starts/stops its
own services. Run it only while holding the designated GPU lock. It is an audit
tool; ordinary users use `tools/cow_cli.py` against an already-running service.
