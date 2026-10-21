<!-- Guided setup uses the existing COW runtime, not ordinary Strata serving. -->
**New here? [Set up a desktop server and a laptop client](docs/TWO-COMPUTER-SETUP.md).**
This step-by-step Linux guide includes automatic Q4 model download, server setup
and the BranchWrite client. No AI assistant is needed to follow it.

# Context COW

**Linux server:** clone this repository, then run `./cow-setup`. It detects
the server and prepares its engine, model and settings. Run `./cow-start` afterward.
[Desktop + laptop instructions](docs/TWO-COMPUTER-SETUP.md).

Optional smaller downloads use the pinned GSQ-RCO models:

```bash
./cow-setup --model IQ2_XS --data-dir "$HOME/ContextCOW-IQ2"
./cow-start "$HOME/ContextCOW-IQ2"
```

Use `--model IQ3_XXS` for the 3-bit configuration. Q4 remains the default.
Use a separate data folder for each model. Models cannot share computational
checkpoints. These options do not add AMD COW inference support.


Experimental source preview,2026-10-20. Context COW keeps immutable model
histories with independent continuations and optional RAM/disk checkpoints.
It uses a modified [Strata](https://github.com/Niko1221/Strata) inference engine.
[BranchWrite](https://github.com/CC-David-CC/BranchWrite) is the separate coding
client and workflow/tool harness using this server through authenticated HTTP.

- Current server/API source: `serve/cow_http.py`, `serve/cow_adapter.py` and
  `experimental/branching_context/src/context_time_travel/`.
- Native implementation and probe builders: `src/`, `include/` and
  `experimental/branching_context/native/` plus `scripts/`.
- [API contract and tests](experimental/branching_context/README.md).
- [Preview scope and provenance](PUBLIC_RELEASE.md).
- [Earlier public publication](https://github.com/CC-David-CC/Public_Context_COW).
- [Original Strata setup/readme](UPSTREAM_README.md).

The Linux guide builds the server from source and prepares the pinned Q4 model.
It requires a working NVIDIA driver and CUDA compiler. **AMD GPUs cannot run
this COW server yet:** the current HIP build has no `strata-branch` inference
driver or COW snapshot-session integration. Ordinary Strata AMD support is
separate. [Executed AMD check](validation/amd-cow-preflight-20261021/README.md).
Other hardware and model
configurations need their own checks. Do not import checkpoints across incompatible profiles. Cross-batch
arrangement equivalence is an accepted prototype limitation; cumulative G1-G5
certification is incomplete. See the preserved mathematical specification for
the broader original contract. An HTTP/client smoke pass is not certification.

