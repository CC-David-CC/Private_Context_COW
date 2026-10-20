# AGENTS.md

## Context COW publication boundary

- Routine commits and backups go to private LAN box-git:
  `box-git:/srv/data/git/private_context_cow.git` and its private evidence area.
- Workers commit locally. The primary agent reviews and pushes explicit private
  refs; never infer a publication destination from `origin`.
- Any GitHub push, pull request, release or other publication requires the
  owner's explicit approval and the applicable review/checks first. Existing
  public repositories and working GitHub credentials are not authorization.
- Strata upstream publication additionally waits for Niko's approval, confirmed
  by the owner. The owner handles that conversation; do not message Niko.

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

## Execution placement and thinking policy (owner, 2026-10-17)

Windows is the steering wheel: use it only to drive SSH/read status/transport.
Do source editing, builds, tests, packaging, logs, checkpoint storage and evidence
collection on the authorized Linux hosts. Do not stage generated artifacts or
archives on the Windows controller. Existing Windows repositories, workloads and
curated LoRA examples are preserved; do not refresh them with bulk outputs.
Private box-git is the backup destination, never GitHub without fresh approval.

New model task admissions use4096 thinking tokens (owner update2026-10-18).
The owner's newer thinking ladder advances on a verified task failure,including
natural early closure;tool progress,infrastructure faults and unresolved effects
do not trigger retries. Standard4K/8K/16K/32K and opt-in Lite starting256 have
definitions;live Lite testing is deferred. Keep saved controllers unchanged.
Do not run no-thinking arms unless explicitly requested. Earlier experiments remain historical evidence. Harness defaults for
reproducing those old experiments do not override this operational instruction.


- Capture generalizable model failures in the sibling
  `branchwrite-lora-examples/candidates/` collection, even when more examples,
  clearer prompts, grammar or harness improvements work around them. Follow
  that collection's README.md and COLLECTION-RULE.md. Preserve original evidence,
  separate mitigation outcomes and a generator brief. Do not infer a systematic
  defect from one observation or export protected test answers as training data.
  Collection does not authorize training or public publication.

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

Keep existing512-token successes, frozen registries and persisted controller budgets unchanged.
Natural-cutoff research and broader end-to-end data collection are deferred.
