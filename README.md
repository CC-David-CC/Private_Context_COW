# Private Context COW (PCC)

## 4.8x faster on small forward-work probes

S3's evaluator can now use existing COW context reuse through the same native
consumed-request code used by BW. Two shared-A100 probes measured 4.8x and 7.1x
speedups; **4.8x is the lower observed ratio**, not a universal lower bound.
Other loads were active on the server. These timings cover small forward-work
and restore checks; full-assessment and complete training-loop timings are
separate measurements.

Read the [implementation and numerical evidence](docs/COW-EVALUATION.md) and
[S3's detailed evaluation guide](https://github.com/CC-David-CC/sss/blob/feat/minicpm-baseline-20261024/docs/COW-EVALUATION.md).

[What we are discovering](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/FINDINGS.md): readable findings about context
cost, tool differences, training, forgetting and practical next steps.

[Wen's full-weight update pilot](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/WEN-FULL-UPDATE-PILOT.md)
completed ordinary native backprop and FP32 save/reload. Protected checks moved
from 17/54 to 19/54, with three gains and one regression; the candidate was not
promoted. [Harness comparisons](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/HARNESS-COMPARISON.md) hold weights
fixed and measure a separate intervention. The
[larger paired assessment](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/WEN-PAIRED-ASSESSMENT.md)
found place value 23/96 to 29/96, seven gains and one loss, with unchanged
outcomes on 96 review questions; the candidate remains unpromoted.
[Real-client CPU checks](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/REAL-HARNESS-SETUP.md)
verify PCC's existing Responses formatter with Codex and real file/tool loops
in Codex/OpenCode/pi. These mock-reply checks do not measure Wen quality.
[Joint learning and harness hill-climbing](https://github.com/CC-David-CC/sss/blob/feat/foundations-20261023/docs/JOINT-LEARNING-LOOP.md)
keeps separate per-harness improvements and mixed-review training measurable;
the full live coupled cycle remains to be established.
Development checkpoint: [foundations and workflow validation](validation/foundations-20261023/README.md). Live-model qualification remains pending.


[S3 foundations, optional thinking and multi-turn lesson mechanics](docs/SSS-FOUNDATIONS.md). The operator guide separates working reference machinery from model qualifications.

**Run an AI model, keep its place, and explore different directions from the same starting point.**

Private Context COW is built on a **modified [Strata core](https://github.com/Niko1221/Strata)**.
It combines the engine that runs the model with the COW layer that manages saved
model context. This repository contains both parts.

Use **PCC** for the whole project, **Strata core** for its model engine, and **COW**
for its context-management layer. The GitHub repository is named
`Private_Context_COW`; "Private" is part of its name, not a statement that this
public repository is hidden.

**Want the AI to work on your files? Use [BranchWrite (BW)](https://github.com/CC-David-CC/BranchWrite).**
BW is the app you talk to; PCC is the server it connects to.

## One system, three responsibilities

| Part | What it does | Why it matters |
| --- | --- | --- |
| **Modified Strata core, inside PCC** | Runs the supported AI model using the graphics card and system memory. | Provides the engine that produces the model's answers. |
| **COW layer, inside PCC** | Keeps model context, independent branches and reusable saved computation. Manages memory, disk caches and placement on configured workers. | Lets related requests share a beginning, try separate futures and continue from saved work. |
| **BranchWrite, a separate app** | Takes your requests, reads and changes permitted project files, runs checks and keeps saved project versions. | Turns model answers into work you can inspect and continue. |

The AI model is the downloaded set of learned weights. Strata runs it; COW manages
its context; BW manages the task and tools. Choosing a model and giving it useful
tools still affect what it can accomplish.

A single suitable Linux computer can run both PCC and BW. A separate laptop is
optional. With a remote server, your project files stay on the BW computer and
the content needed for the task is sent to your chosen PCC server. That client
computer needs no model download or graphics card.

## What does "keeping the AI's place" mean?

Suppose the AI has read your project and you want to try two repairs.

1. Keep that shared starting point.
2. Fork it: try repair A and repair B as independent continuations.
3. Inspect the results and continue the direction you choose.

A **fork** starts another path from a saved point. It does not make the two paths
share subsequent changes. The original saved history stays intact.

COW keeps two different things:

- **History:** the recorded inputs and continuation settings that define a path.
- **Saved computation, or a checkpoint:** work the engine can reuse so it may
  continue without processing the same beginning again.

Checkpoints can live in graphics memory, system RAM or on disk. Removing an
optional checkpoint does not remove retained history. Continuing later may need
to rebuild that computation. Reuse requires a compatible model and execution
setup; a saved state for one model cannot simply become another model's state.

BW adds the matching project-file versions and user controls. See
[its saved-version walkthrough](https://github.com/CC-David-CC/BranchWrite#try-another-direction-and-come-back).
Saving model context and saving your files have different jobs; the two work together.

## What the newer cache features add

These features extend the same COW layer. They are **optional server settings**;
a BW user keeps making ordinary requests.

| Feature | In ordinary words | Why it is useful |
| --- | --- | --- |
| **Recoverable cache inventory** | Tracks where reusable saved computation is available and can recover its view after missed updates or a restart. | The server can look for existing work before rebuilding it. |
| **Placement that considers reuse and cost** | Considers cached work, measured restore/computation costs and worker availability when choosing a configured local worker. | A warm worker is useful when reusing its work is actually worthwhile. |
| **Configured cache peers** | Transfers compatible, checked checkpoints between explicitly configured servers. | Saved computation can be recovered from another location. This transfers cache data; it does not send inference jobs to a remote server. |
| **Separate prefill and decode roles** | Can use one GPU to process a request's starting input and another for continuation requests. | Allows those stages to be assigned separately. Input added during continuation still runs on the continuation worker. |
| **Optional fastokens** | Uses an additional tokenizer for qualified long text pieces while keeping the existing short-piece path. | Reduces text-to-token preparation time on the measured long-piece examples. |

These settings are off by default. They do not automatically combine every
computer's graphics cards into one model or make incompatible checkpoints portable.
The measured handoffs and workflows passed; a general whole-task speedup has not
been established. See [settings and examples](docs/CACHE-FABRIC.md) and
[how the integrated system was checked](docs/STACK-CAPABILITIES.md).

### Inspiration and credit

The COW inventory, placement and transfer work was inspired by published designs
from **DeepInfra, NVIDIA Dynamo and KVCR**. The optional tokenizer uses
**fastokens**. The COW integration was written for this project; those complete
systems are not bundled as additional services.

Thank you to their contributors. [The design references, upstream links and
dependency details are here](docs/CACHE-FABRIC.md#inspiration-and-thanks).

## Start using it

**[Start here with one computer or a remote server](docs/START-HERE.md).**
The guide covers the machines below, first use, forks, optional features and the
research-model test programme. You do not need to own the whole collection.

| Your setup | Current route and evidence |
| --- | --- |
| **RTX PRO 6000 Blackwell, 96 GB** | Main reference for large Qwen Flash Q4 serving; Wen/Qwen native BW workflows and the new feature controls have bounded tests. |
| **RTX 4090, 24 GB** | The smaller Qwen3.5-9B dense adapter has native state and BW website checks. Large Flash Q4/IQ3/IQ2 trials have important memory, latency and coding/recovery limits. |
| **RX 7900 XTX, 24 GB** | A separate dense-model HIP branch has native continuation and HTTP image/restart checks. It is not merged into this main Flash installer or qualified for the new complete feature stack. |
| **R730 with two Tesla P4s** | Current evidence covers CPU/cache-storage work; the combined PCC/BW inference path on those GPUs remains unqualified. Earlier Strata P4 experiments do not establish it. |
| **Rented/shared A100** | Real Wen workflows, compatible checkpoint handoff and separate input/continuation roles were checked. Use allocated GPUs and connect BW from a trusted Linux client, as tested. |

[Choose the appropriate route](docs/START-HERE.md#choose-your-hardware-route).
For the existing large-model installation steps, use the
[desktop/server walkthrough](docs/TWO-COMPUTER-SETUP.md).

On a suitable Linux/NVIDIA server with its driver and CUDA compiler ready:

```sh
git clone https://github.com/CC-David-CC/Private_Context_COW.git ContextCOW
cd ContextCOW
./cow-setup
./cow-start
```

Setup prepares the engine, model and server configuration. Start keeps the server
running. Give the BW user its server address and API key privately. An API key is
the password for the service. Keep this setup on a trusted network.

### Model downloads and measured results

| Download | Weight size | Start here |
| --- | --- | --- |
| **Qwen3.8 Flash Q4**, default | **111.33 GB** (103.69 GiB) | `./cow-setup`; [Flash installation](docs/TWO-COMPUTER-SETUP.md). |
| **Wen 1.1 BF16**, research model | **16.24 GB** (15.13 GiB) | [Download GGUF](https://huggingface.co/davidflanagancomm/Wen/resolve/7328e9fab25b207523b5236aa29c81be24eed924/Wen-1.1-BF16.gguf?download=true); [download, verify and dense setup](docs/MODELS-AND-RESULTS.md). |
| **Original Wen 1 BF16**, comparison model | **15.94 GB** (14.85 GiB) | [Historical GGUF](https://huggingface.co/davidflanagancomm/Wen/resolve/674907f2b65baa576e678b9c9a06bf07c3db9076/Wen-BF16.gguf?download=true); same separate dense route. |

These are download sizes, not VRAM requirements. The tested large-Q4 reference
server has **128 GB RAM and 96 GB graphics memory**. Wen does not use the Flash
one-click downloader. [The model guide](docs/MODELS-AND-RESULTS.md) gives pinned
CLI downloads, exact bytes/hashes and historical Q4/Q8/lower-quant results.

[Skills³ (S³)](https://github.com/CC-David-CC/sss) owns the detailed skill tree,
model comparisons and learning/regression reports. PCC summarizes the model and
hardware story here; BW summarizes the practical task story. The current Wen 1
and 1.1 comparison uses the same S³ tool fixtures. Historical Flash throughput
is a separate campaign, not a score on those tasks.

The first matched A100 trial finished at **Wen 1: 5/9; Wen 1.1: 4/9**. These
are strict mock-tool completion scores, including correct answers that omitted
the required finish action; newline failures have unresolved parser/model
attribution. [Read the full results and limitations](https://github.com/CC-David-CC/sss/blob/main/docs/MODEL-COMPARISON.md).

[Smaller Q4/IQ3/IQ2 trials](validation/lower-quants-llm49-20261021/) have their own
results and limitations. Ordinary upstream Strata hardware support does not imply
that every COW adapter supports the same hardware. The guided Flash/COW setup
currently targets NVIDIA/Linux; see the [AMD check](validation/amd-cow-preflight-20261021/).

For supported dense models, including the inspected Wen naming support, see the
[separate native-adapter guide](docs/QWEN35-9B-BACKEND.md).
[Other compatible CLI clients](docs/COW-CLI-COMPATIBILITY.md) can also use the
configured server. BW supplies the additional project, tool and workflow controls.

The current integration source is on this repository's `main` branch. The
[new user guide](docs/START-HERE.md) and linked installation walkthrough explicitly
select BW's current `main`. Omitting `BW_REF=main` still makes the installer choose
its older frozen preview. Existing installations do not update automatically.
Keep matching client/server versions and the versions needed by saved work.

## Why BW benefits without a second cache system

The new inventory, placement and transfer features belong to the **COW layer**.
BW keeps sending task/model requests to PCC, and PCC manages available saved work
and local execution. BW does not select GPUs or copy checkpoint files for each task.
The integrated BW client checks PCC's advertised capabilities and identities at
connection and admission of unfinished work. Completed results can reopen offline.

No BW-specific cache feature must be enabled for every server optimization.
An older installed BW build still needs updating to gain the new client-side
discovery/replay changes. Ordinary single-GPU reuse works without cache peers;
peer transfers and separate GPU roles need the corresponding extra resources.

## Research model, real mechanics, unfinished validation

Wen/Wen 1.1 was used as a research model to exercise the machinery: independent
continuations from a shared beginning, native state capture and restore,
checked cache transfer, pause/resume, tokenizer equality and small BW coding
workflows. Original Qwen models provided additional checks. Dense Wen/Qwen
execution uses the optional pinned native adapter; the large Flash model uses
the modified Strata path. Both are managed through the COW layer and BW.

These were real executions with controlled inputs, complete-state/output checks
and bounded tasks. **They were not a full day of real-world inference load.**
They do not establish many-user operation, sustained queue/memory/disk pressure,
general model quality, every hardware combination or a general speed advantage.

The aim was to bring the features together and check specific mechanics.
**Working integration increments have passed their tests; the complete feature
unification and validation programme is still unfinished.** Read
[how the research model was used and what remains](docs/START-HERE.md#why-we-used-a-research-model).

### Native training work in progress

PCC is also gaining an **offline native training experiment**, separate from its
context cache. The goal is to take an independently checked correction and use
it to update either a small **LoRA adapter** (the model's reusable "hat") or the
model's own weights through **ordinary full backpropagation**. Direct weight
updates do not require merging an adapter into the model.

The training code uses C++/GGML without PyTorch. Ordinary end-to-end backprop is
the default, including when the graph spans GPUs. Passing gradients between
independently executed sections is a separate experimental path.

**Optional BW and CLI lesson controls are now connected.** After a failed check
and verified correction, write one checked lesson and train it now, or queue
lessons for a later batch. Continue a completed LoRA, or request direct full-weight
updates. The CPU wiring test uses real native updates, private records and both
CLIs; its tiny generated model is not evidence of useful language learning.

Training produces separate, versioned candidates. It does not automatically
learn from every conversation or replace the model serving existing requests.
The owner must enable training and configure resources. Read the
[lesson workflow and setup](docs/LEARNING-LOOP.md) and
[native training status and limits](experimental/native_training/README.md).

The [first 1,000-variation Wen learning study](validation/wen-learning-quality-20261023/README.md)
learned the selection pattern (2/48 to 48/48), but lost five of sixteen unrelated
checks and reached only 3/16 on differently worded blind cases. It failed the
declared quality gate and remains unpromoted. Working training controls do not
yet establish a model that reliably improves with each lesson.

## What has been checked?

The skill tree, evaluation suites and mixed-practice policy now belong to the
independent **[Skills³ (S³)](https://github.com/CC-David-CC/sss)** package. PCC
retains native computation/training, resource leases, immutable candidates and
context identity. Its optional owner-configured S³ adapter can evaluate all
active skills between bounded training rounds and prioritize a regression's
separate practice examples. It does not turn test answers into training labels.
See [S³ integration](https://github.com/CC-David-CC/sss/blob/main/docs/INTEGRATION.md)
for the current five-skill arithmetic scope and broader unmeasured admin/code map.

Real Wen 1.1 and Qwen Flash Q4 runs completed small BW read/edit/test/finish tasks
with the optional features off and on. Separate checks covered pause/resume,
offline reopening of completed BW work, A100 checkpoint handoff, separate GPU
roles and LAN cache round trips restored on the original native profile.

The final recorded CPU campaign passed **2,077 tests**; the G0 reference checks
passed **51**. BW passed **158 focused integration tests**, and the related COW
HTTP/recovery selection passed **53**. Some selections overlap; these counts
should not be added. [Read the results and their exact scope](docs/STACK-CAPABILITIES.md).

This is a research preview. Broader unfamiliar-project testing and the complete
G1-G5 correctness/stress programme remain open. The storage-only listeners used
in the LAN trial are test tools, not a packaged storage service. Running-request
migration and cross-profile checkpoint portability are not provided by this increment.

## Strata credit and upstream invitation

PCC builds on Niko's work and the work of the
[upstream Strata contributors](https://github.com/Niko1221/Strata).
**Niko and the upstream Strata contributors are welcome to review this project
and take useful improvements back into Strata.**

This is an independently maintained project. Changes here are not automatically
changes to Niko's upstream `main`. This integration uses its documented pinned
engine; the planned upstream Strata refresh is separate.

For the current combined server, start here. [Public_Context_COW](https://github.com/CC-David-CC/Public_Context_COW)
is an earlier public publication, and [Strata-a5500](https://github.com/CC-David-CC/Strata-a5500)
is a separate Strata fork. Their repository names do not identify extra components
you must install alongside PCC and BW.

## Technical documentation

- [Server setup and operating scope](docs/CONTEXT-COW-SERVER.md)
- [Cache settings, transfers, GPU roles and tokenizer](docs/CACHE-FABRIC.md)
- [Integrated capabilities and recorded tests](docs/STACK-CAPABILITIES.md)
- [Branching-context API and tests](experimental/branching_context/README.md)
- [Source publication and provenance](PUBLIC_RELEASE.md)
- [Earlier inherited README and interface history](UPSTREAM_README.md)

Server/API source is in `serve/cow_http.py`, `serve/cow_adapter.py` and
`experimental/branching_context/src/context_time_travel/`. Native engine and adapter
work is in `src/`, `include/` and `experimental/branching_context/native/`.

MiniCPM5-2B ordinary dense contexts now have CPU/CUDA and tokenizer integration evidence: [setup and current qualification limits](experimental/branching_context/MINICPM-NATIVE.md). MiniCPM training is not yet qualified.

## MiniCPM training development checkpoint

[Native ordinary-Llama mechanics](experimental/native_training/MINICPM-QUALIFICATION.md)
now pass independently checked tiny full/LoRA CPU graphs; an earlier revision
also passed on two A100s. The exact final CUDA guard, real MiniCPM forward and
learning remain unqualified. [Muon comparison](docs/OPTIMIZER-COMPARISON.md) is
planned at a fixed compute budget; only SGD exists in the trainer today.

[Native AdamW/Muon checkpoint](experimental/native_training/OPTIMIZER-QUALIFICATION.md):
CPU/CUDA math and exact completed-state continuation pass. Actual MiniCPM short
forward agrees with FP64; the earlier 0.011 discrepancy is isolated to inference
output selection/execution shape. No MiniCPM weights have been trained yet.
