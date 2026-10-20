# Context COW terminal client private draft

This package connects your terminal to an existing Context COW model service.
The model runs on the server. This package has no model files, GPU engine,
browser or installer for an independent inference server.

## Before you start

Ask the server owner for an authorized SSH address and the COW service port.
The examples use `llm-60` and port `8099`; these are examples, not a public
service address. Your computer must be able to reach the supplied SSH host.
The owner must start the dedicated COW service, not ordinary Strata chat.
The owner recipe is [CLI server setup](CLI-SERVER.md). Your first trial connects
to that existing server; it does not require a model download on your computer.

## Install Python on Windows

If `python --version` already reports Python 3.12 or newer, skip installation.
The client needs no extra Python packages or virtual environment. A virtual
environment is a separate folder for a program's Python packages; this small
client uses only packages included with Python.

1. Open the [Python 3.13.15 download page](https://www.python.org/downloads/release/python-31315/).
   This version is verified on the project's Windows controller.
2. Under **Files**, select **Windows installer (64-bit)**.
3. Run the installer. Select **Add python.exe to PATH**, then **Install Now**.
4. After installation, close and reopen PowerShell.
5. Type `python --version`. Use `python`, not `python3`, on Windows.

Extract the private client ZIP. Open its `context-cow-cli` folder in File
Explorer, type `powershell` in the address bar and press Enter. Keep this
PowerShell window for client commands. Do not install the model on this PC.

## Open the connection

In another terminal, open an SSH tunnel using the address and port the owner
provided. Keep that terminal open:

```powershell
ssh -N -L 18080:127.0.0.1:8099 llm-60
```

In the client terminal, run:

```powershell
$env:COW_URL = "http://127.0.0.1:18080"
python tools/cow_cli.py --url $env:COW_URL status
```

If the owner enabled an API key, enter it without putting its value in your
command history:

```powershell
$env:COW_API_KEY = Read-Host "COW API key"
```

The loopback service connection travels through SSH. A direct remote URL must
use HTTPS. The CLI does not configure TLS or open a public server port.

## Try two independent questions

Create a UTF-8 text file named `codebase.txt` in the client folder. Put a few
source files or project notes in it, with filenames above their contents. Start
with a modest real context. Keep that file unchanged during this first trial.

```powershell
python tools/cow_cli.py --url $env:COW_URL namespace --title "My project"
```

The returned receipt contains a `namespace_id`. Copy its value and replace
`PASTE_NAMESPACE_ID` in the next command. The example word is not a valid ID.

```powershell
python tools/cow_cli.py --url $env:COW_URL load --namespace-id PASTE_NAMESPACE_ID --branch MyBase --file codebase.txt
python tools/cow_cli.py --url $env:COW_URL fork --source-branch MyBase --name MyQuestionA
python tools/cow_cli.py --url $env:COW_URL fork --source-branch MyBase --name MyQuestionB
python tools/cow_cli.py --url $env:COW_URL ask --branch MyQuestionA --text "Explain the main data flow." --max-tokens 128
python tools/cow_cli.py --url $env:COW_URL ask --branch MyQuestionB --text "Which boundaries need tests?" --max-tokens 128
python tools/cow_cli.py --url $env:COW_URL history --branch MyQuestionA
```

Both questions start from `MyBase`. Answer A is not part of question B. Select
another branch by supplying its name in the next command; there is no separate
local active-chat database.

A 128-token limit can save a partial answer while its controller remains active.
Use `continue` for that answer. Use `append-user` for another user message only
after the model's controller is stopped. The history command reports this state.

```powershell
python tools/cow_cli.py --url $env:COW_URL continue --branch MyQuestionA --max-tokens 128
```

For more projects, create more namespaces and base branches. The server queues
their requests and admits one request per configured GPU worker. Idle trees
remain saved without each requiring an active GPU request.

## Know when the answer is saved

Generated text goes to standard output. Operation IDs and durable receipts go
to standard error, so redirecting standard output saves just the displayed text.
The text is provisional until the CLI prints a durable completion receipt.

Ctrl+C exits the client. It does not prove that the server cancelled before
committing. Save the printed operation ID and inspect it before retrying:

```powershell
python tools/cow_cli.py --url $env:COW_URL operation --operation-id PASTE_OPERATION_ID
```

Operation lookup is local to the running coordinator process. After a server
restart, inspect canonical history and branch heads as well. The current source
client also keeps a private request journal. To explicitly retry the original
request after a lost response, use:

```powershell
python tools/cow_cli.py --url $env:COW_URL retry-request --operation-id PASTE_OPERATION_ID
```

This keeps the same request body, operation ID and original branch revision.
It does not read a newer head or create a new action. A completed local receipt
can be returned without sending another request. The server can return an error
if unfinished computation is no longer available. Do not blindly resubmit with
a new operation ID. The CLI performs no automatic mutation retry.

The journal is saved under `%LOCALAPPDATA%\Strata\ContextCOW\requests` by
default. It can contain prompts and source code from your requests. Keep it
private and keep it across client restarts. It does not contain your API key.
See [the recovery walkthrough](CLI.md#recover-an-uncertain-operation) for a
custom journal folder and the separate `retry-publication` command.

The earlier private v3 ZIP predates the request journal. It does not have
`retry-request`. These instructions describe the current source client; use a
newly qualified package before relying on that command in an extracted ZIP.

## Read more and report a problem

[CLI command walkthrough](CLI.md) covers fork, rewind and publication recovery.
[Human and agent guide](CONTEXT-COW-GUIDE.md) explains storage, performance and
future coding-client integration. In the extracted ZIP, find both files under
`docs/context-cow`.

This draft supports plain-text requests under the service's greedy profile.
It is not an OpenCode, Hermes or Codex provider bridge. Tool calling, attachments,
arbitrary sampling and independent-machine engine installation are separate work.

For a useful report, keep the command, operation ID, branch, error, actual
output count and whether the service restarted. Include timing and context
size if known. Keep credentials out of reports. Preserve the project's executed
test evidence separately from the client package; an early trial is not full
G1 through G5 certification.
