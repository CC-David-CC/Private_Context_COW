# AGENTS.md

## Publication

This is a public source project. Publish reviewed changes only to the explicitly
authorized repository and branch. Preserve existing remotes and unpublished
history. Keep credentials, protected diagnostic answers and machine-specific
operational handoffs out of public commits. Contributions to other projects
require separate owner authorization; do not contact maintainers on the owner's
behalf.

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Context COW for a user

Use [docs/TWO-COMPUTER-SETUP.md](docs/TWO-COMPUTER-SETUP.md) for the Linux desktop
server and BranchWrite laptop client. The pinned Q4 resources are prepared by
`tools/cow_prepare_model.py`; the server is configured and started with
`tools/cow_server_setup.py` and `tools/cow_server.py`. Ordinary Strata serving
setup is a separate path and downloads additional resources not used by this
COW installation. Keep the model server on an authenticated trusted-LAN boundary.

The original engine's separate setup documentation remains in
[docs/AI_SETUP.md](docs/AI_SETUP.md) and [UPSTREAM_README.md](UPSTREAM_README.md).
Do not substitute that server for the COW API expected by BranchWrite.

## Working on the code

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
