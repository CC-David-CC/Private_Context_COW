# Start here: PCC and BW on the computer you have

**PCC runs the AI and keeps its place. BW lets you ask it to work on a project.**
You can run both on one suitable Linux computer, or keep BW on your own computer
and connect to a server. Owning several GPUs or several computers is optional.

This guide describes the current source and the tested routes as of 23 October
2026. Some hardware routes are separate previews or still need work. The table
below tells you before you download a model.

## What are you installing?

**Private Context COW (PCC)** contains two parts:

- The **modified Strata core** runs supported models using graphics and system memory.
- The **COW layer** manages recorded model context, independent continuations,
  checkpoints, memory/disk caches and placement on configured workers.

The optional dense-model route uses a pinned native llama.cpp adapter under the
same COW layer. It does not require pretending dense Wen and large Flash use an
identical engine. The model itself is a separate downloaded file or set of files.

**BranchWrite (BW)** is the separate client. It takes your request, manages
permitted file/tool operations, runs checks and saves project versions and task
progress. COW keeps model context; BW keeps the task and files coordinated with it.

PCC is built on [Niko's Strata](https://github.com/Niko1221/Strata).
Niko and the upstream Strata contributors are welcome to review these changes
and take useful improvements back into Strata.

## Choose your hardware route

All routes below assume Linux. The BW installer currently supports 64-bit
Intel/AMD Linux. These are CPU names; AMD Radeon graphics need the separate HIP
route. Existing Windows viewing/SSH use does not establish a native Windows BW
installation.

| Your computer | What to use | What the evidence actually covers |
| --- | --- | --- |
| **RTX PRO 6000 Blackwell, 96 GB VRAM** | The guided large Flash Q4 server on current main, or a separately configured dense model. | The reference large-model machine had 128 GB system RAM. Wen/Qwen BW tasks, optional tokenizer workflows and returned native cache execution passed bounded checks. |
| **RTX 4090, 24 GB VRAM** | For the smaller-model route, follow the Qwen3.5-9B dense-adapter guide. Large Flash preparations use the separate guided setup and need substantial RAM/disk. | Dense state/restore checks and a BW website/follow-up ran. On a 32 GB RAM host, large Flash trials were slow and some coding/recovery runs failed. A smaller download alone does not fix that. |
| **RX 7900 XTX, 24 GB VRAM** | The separate `feat/qwen35-9b-hip` dense-model preview, with its pinned ROCm build and model/profile. | Native text/mixed-state comparisons and HTTP image continuation across restart passed. Interactive BW UX and the latest complete cache/placement stack were not rerun on AMD. This branch is not part of main's Flash installer. |
| **Dell R730 with two Tesla P4s** | Current main can be used for appropriate CPU/client or cache-storage experiments; combined GPU inference still needs a qualified backend. | The LAN test used the R730 as checkpoint storage, then executed the returned state on the original model host. It did not execute inference on either P4. Two cards are not automatically one combined pool. |
| **Rented/shared A100 server** | The tested native dense/Wen route with your own model/build/profile and allocated GPU(s); run BW on the server or connect through SSH. | Small Wen BW tasks, peer restore, compatible cross-worker handoff and input/continuation roles passed. This does not reserve the whole host or establish WAN transfer throughput. |

If your only machine is the R730/P4 setup, there is no established current
end-to-end PCC/BW GPU installation recipe to promise yet. Use it as a BW client
to an available server, or treat qualifying a P4 inference backend as remaining
engineering work. The rest of this guide does not silently turn that into a pass.

### Route A: large Flash model on NVIDIA

Use the [existing guided installation](TWO-COMPUTER-SETUP.md) for driver/CUDA checks,
system tools and model preparation. On the server:

```sh
mkdir -p ~/Apps
cd ~/Apps
git clone https://github.com/CC-David-CC/Private_Context_COW.git ContextCOW
cd ContextCOW
./cow-setup --check
./cow-setup
./cow-start
```

`--check` previews setup without installing or downloading. Review the reported
memory and disk budget before accepting the download. The default model is about
111 GB; the original guide asks for 200 GB free disk. The server needs a working
NVIDIA driver and CUDA compiler before setup. Leave `cow-start` running.

For IQ3/IQ2, use the [separate model-folder commands and measured limitations](TWO-COMPUTER-SETUP.md#optional-choose-a-smaller-model).
The smaller-model trials do not establish a successful full coding workload on a
32 GB RAM / 4090 machine. Keep different models in separate data folders.

### Route B: smaller dense model on NVIDIA

Follow [Qwen3.5-9B and research-model setup](QWEN35-9B-BACKEND.md). This route builds
a pinned native adapter, uses the actual model file and creates a new server
configuration. The tested Qwen Q8 file is about 9.5 GB, rather than the guided
111 GB Flash download. Model-file size is not total memory usage.

That guide provides the build commands, model hashes, configuration and launch
steps. It is currently an operator setup, not the automatic Flash downloader.
A compatible model and pinned native build must be supplied; changing a model,
GPU arrangement or build needs a distinct compatible profile and its own checks.

For an already configured dense server, its operator starts it with the Python
environment used during setup:

```sh
"$COW_PYTHON" "$COW_ROOT/tools/cow_server.py" "$SERVER_DATA"
```

Here `COW_PYTHON`, `COW_ROOT` and `SERVER_DATA` are the installation paths defined
in that guide. Do not run setup again just to restart an existing server.

### Route C: RX 7900 XTX dense preview

Use a separate checkout of the [published HIP branch](https://github.com/CC-David-CC/Private_Context_COW/tree/feat/qwen35-9b-hip):

```sh
git clone --branch feat/qwen35-9b-hip https://github.com/CC-David-CC/Private_Context_COW.git ContextCOW-AMD
cd ContextCOW-AMD
```

Follow that branch's [single-card AMD build instructions](https://github.com/CC-David-CC/Private_Context_COW/blob/feat/qwen35-9b-hip/docs/QWEN35-9B-BACKEND.md#single-card-amd-build-preview)
with an installed ROCm development environment. The executed target was gfx1100
with the pinned llama.cpp revision. Read its [runtime and restart evidence](https://github.com/CC-David-CC/Private_Context_COW/blob/feat/qwen35-9b-hip/validation/qwen35-hip-20261021/README.md).

Do not run the NVIDIA Flash installer and expect it to configure this route.
The earlier [blocked AMD check](../validation/amd-cow-preflight-20261021/README.md)
describes the Flash path; the later dense preview is a different backend.
CUDA and HIP checkpoints cannot be interchanged. The new main cache/role features
are not qualified merely because this earlier AMD HTTP run passed.

### Route D: rented or shared A100

Use the dense/native setup guide with the model and build prepared for the
server. The tested BW client ran on a separate Linux host. Running BW on the
shared server itself additionally requires its tool/sandbox admission to pass;
the tested A100 host's system-tool ownership prevented that route. Do not weaken
those checks to make a shared-host installation appear to work.

Check actual GPU use and follow the provider's allocation rules before
choosing physical device IDs. A COW launcher lock coordinates cooperating COW
processes under your account; it does not reserve a GPU against other users.

Start with one allocated GPU and ordinary combined serving. Only try separate
input/continuation roles if you have suitable independently allocated workers.
The historical test's GPU numbers are examples, not a permanent reservation.
Budget host RAM, active model memory, checkpoints and disk staging separately.

On a rented server, keep the API bound to an appropriate private boundary. One
way to use a localhost-bound PCC server from your Linux client is SSH forwarding:

```sh
ssh -N -L 18099:127.0.0.1:8099 YOUR_ACCOUNT@YOUR_SERVER
```

Replace the account/host and the server's port with your real values. Leave this
terminal open. In BW on your client, enter `http://127.0.0.1:18099` and the server's
API key. This exposes the forwarded service only on your client's local address;
it does not publish an unauthenticated model server on the Internet.

## Install BW on that same computer, or on your client

Run these on the computer where you want the project files. On Ubuntu/Debian:

```sh
sudo apt update
sudo apt install -y curl git ca-certificates bubblewrap util-linux
curl -fsSL https://raw.githubusercontent.com/CC-David-CC/BranchWrite/main/install.sh -o "$HOME/branchwrite-install.sh"
BW_REF=main sh "$HOME/branchwrite-install.sh"
```

Run the installer as your normal user. `BW_REF=main` selects current source;
omitting it selects the older frozen preview. An operator can pin a reviewed
commit instead. Keep old runtime versions while saved tasks still need them.
This documentation change does not constitute a fresh installation test.

Open your first project:

```sh
mkdir -p ~/my-first-website
cd ~/my-first-website
~/.local/bin/bw
```

At the connection prompts:

- **PCC and BW on this computer:** use PCC's configured local URL, commonly
  `http://127.0.0.1:8099`.
- **PCC on another trusted-LAN computer:** use its actual LAN address and port.
- **SSH forwarding as above:** use `http://127.0.0.1:18099`.

The API key is the service's password, created during server setup. The guided
default stores it at `$HOME/ContextCOW-data/server/api-key.txt`; custom setups
store it in their configured server directory. Give it privately to the BW user.
BW keeps it in memory unless explicitly asked to save it.

## Make something, fork it, and return

Start with a small task:

> Make a one-page website about dogs. Use one HTML file. Make it readable on a
> phone and add a button that shows a friendly greeting. Do not download anything.

When it finishes, type `/preview` and open the printed link. Try the button and
review the page. Ask for a follow-up such as "Make the dog names bold."

Use `/history dogs` to find saved Work turns. For an example saved turn 1:

- `/explore 1 Make the site more playful.` uses the model context after that turn
  and the current files, in a separate alternative folder.
- `/edit 1 Make a formal dog-training website.` starts before that request with
  its corresponding file version and admits a changed request.
- `/switch 1` restores that selected turn's matching file version. Conflicting
  manual edits cause refusal rather than silent overwrite.

Ordinary follow-ups, alternatives and restored versions have different meanings.
The [BW interactive guide](https://github.com/CC-David-CC/BranchWrite/blob/main/docs/manual/09_INTERACTIVE_SESSION.md)
explains the precise behaviour. No GPU choice is needed for each message.

Enter steers active work at a safe boundary; Tab queues a note; Esc asks it to
stop. `/plan` discusses without editing, `/work` permits changes, and `/aside`
asks a side question. `/exit` settles and leaves. Reopen BW in the same folder to
return; `/continue` resumes saved unfinished work when required. Wait for the
settlement message instead of resubmitting an uncertain operation with a new ID.

## What the DeepInfra-inspired features do

The new features are in **PCC's COW layer**. BW uses the same server requests
and benefits when the relevant optional settings are enabled. Its current client
also checks advertised server capabilities and model/execution identities.
Completed BW results reopen from its own saved records without rediscovering a
server. BW does not maintain a competing model-cache database.

| Feature | What the operator enables | What the user gains |
| --- | --- | --- |
| Cache inventory and recovery | `--cache-inventory` | PCC can observe available saved computation and recover after missed location updates. |
| Cost-aware local placement | `--scheduling transfer-aware` | PCC can compare measured restore and computation costs when choosing an available local worker. |
| Checkpoint export and configured peers | `--cache-transfer` and a private `--cache-peers` configuration, with inventory as required | Compatible saved computation can be checked and recovered from another location. |
| Separate input/continuation roles | Appropriate `--gpu`, `--prefill-gpus` and `--decode-gpus` sets | Allocated workers can have different roles; suffix input still runs with its continuation worker. |
| Optional tokenizer | `--tokenizer-backend fastokens`, with the qualified dependency installed | Faster token preparation on the measured long-piece cases, while keeping the normal short-piece path. |

These are configuration arguments to the normal setup tools, not commands to
paste as a complete installation. Use a new server configuration where required,
with the actual model/native-build arguments from your route.
[The full settings, examples, dependency pins and bounds are here](CACHE-FABRIC.md).

On one GPU, ordinary context reuse remains useful without any peer or split-role
configuration. Two P4s, one AMD card, a 4090 and an A100 are not interchangeable
checkpoint destinations. Model and execution identities, integrity and actual
native restore support must match. Peer transfer moves saved data, not an
inference job. The test storage listeners are not a packaged cache-server product.

Credit goes to the published **DeepInfra, NVIDIA Dynamo, KVCR and fastokens** work
that inspired this integration. [Read the original design links and attribution](CACHE-FABRIC.md#inspiration-and-thanks).
This is COW's implementation, not a claim that those entire products are bundled.

## Why we used a research model

The immediate aim was to exercise the mechanics while bringing the components
together. Wen/Wen 1.1 supplied a real research model with its own weights,
vocabulary and template. It was run through the pinned dense native adapter;
original Qwen dense and Flash models provided additional checks through the same
COW/BW boundaries. Research-model results are not claims about every model.

The programme checked specific questions:

1. Can two continuations share a beginning without changing the saved parent or
   leaking one continuation into the other?
2. Can a checked native checkpoint restore the expected complete response and
   controller, with counters showing no repetition of saved-prefix computation?
3. Can a compatible checkpoint survive bounded transport, duplicate requests and
   selected interruption/recovery cases without becoming new canonical history?
4. Can BW read, edit, run an actual test and finish a small task with the optional
   features off and on? Can it pause/resume and reopen a completed result offline?
5. Does optional tokenization return the same token IDs before timing its work?

Real Wen/Qwen native workflows passed bounded feature-off/on cases. A100 peer
restore and separate worker roles passed their selected checks. Two Wen payloads
made LAN storage round trips through the 4090, AMD and R730 hosts, then produced
matching native responses on their original host. This demonstrates storage
transfer on those peers, not inference on their GPUs. Fastokens passed 5,180
token-ID checks. CPU campaign/fault tests exercise additional boundaries but do
not substitute for GPU execution on untested hardware.

The recorded final CPU campaign passed 2,077 tests, G0 passed 51, BW's focused
integration selection passed 158, and a related HTTP/recovery selection passed
53. Some selections overlap. [The detailed record](STACK-CAPABILITIES.md) separates
source/runtime revisions, model identities, earlier failures and excluded work.
Small successful coding tasks do not establish broad model competence; other
model/task trials retained failures and timeouts.

### What this did not establish

**We have not run this integrated stack through a full real-world day of varied
inference load. Feature unification and testing are still in progress.**

Remaining work includes long sessions and unfamiliar projects, multiple users,
long queues, sustained GPU/RAM/disk pressure, repeated network/server failures,
fairness and tail latency, realistic cache churn, and useful throughput versus a
same-hardware baseline. The full G1-G5 correctness/stress programme and every
listed hardware/backend combination remain incomplete. The planned upstream
Strata refresh and the separate AMD branch are not automatically merged by these
documentation changes.

A mechanics pass establishes its measured invariant. A useful answer establishes
success for its chosen task. A day-long load test would establish a different set
of operational properties. The project needs all of those; one cannot stand in
for the others. The current public claim is **working research integration with
bounded evidence and explicit remaining work**.

## Stop and return later

Exit BW cleanly and stop your own PCC server with Ctrl+C when you want its model
allocation released. Closing BW alone leaves the server available for other
clients. On a shared machine, never stop another user's processes.

Restart the same configured server directory and reopen the same BW project for
saved work. Keep its pinned runtime versions. If an optional checkpoint was
evicted, retained history may require explicit recomputation. If the run used an
ephemeral server, its session histories/caches are temporary; do not promise
durable server recovery after removing that session. Preserve durable data and
original operation identities when continuing unsettled work.
