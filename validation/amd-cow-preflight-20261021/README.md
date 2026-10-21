# AMD server check — 21 October 2026

**Result: BLOCKED for COW inference.** This is not a numerical or performance qualification.

The RX 7900 XTX was idle with about 24 GB VRAM free. The host had about 57 GiB available RAM. The pinned Q4 shard filenames and sizes were present, but their hashes were not checked and no model was loaded. Only about 7 GB was free on the system disk.

On the published source `086271b0f12e9c9d97ce80be7aa168dcfaf254af`, the HIP/gfx1100 configuration succeeded. Configure plus the target check took 22.16 seconds. Building the required COW target then returned:

```text
gmake: *** No rule to make target 'strata-branch'.  Stop.
```

CMake includes COW snapshot-session code and the inference driver only under `STRATA_ENABLE_CUDA`. The HIP build includes some page/digest support; that does not establish a working COW inference server. Ordinary Strata HIP inference is a separate path and was not tested here.

The installer previously called this a missing `nvidia-smi` tool. It now identifies AMD display hardware and explains the missing COW backend before downloads, builds or data-directory creation. A mixed-vendor unit fixture preserves selection of the existing NVIDIA path; that fixture is not a live mixed-GPU trial.

## Executed checks

| Check | Result |
|---|---|
| Actual AMD preflight | Expected rejection; no data directory created |
| New backend diagnostic tests | 6/6 PASS, after six expected test-first errors |
| Existing setup tests | 17/17 PASS |
| Existing model-preparation tests | 9/9 PASS in the pinned setup environment |
| G0 reference semantics | 51/51 PASS |
| AMD COW inference / CLI coding | BLOCKED; zero requests executed |

Two regression launches used the wrong entry point or an incomplete environment. Their logs were retained. The model-helper suite was rerun with the documented module entry point in the existing pinned setup environment; all nine passed. No assertions or tolerances changed.

There are no new logits, generated tokens, model hashes, inference counters, timing comparisons or G1–G5 passes. Numerical error is null, not zero. The completed diagnostic's 653 MB generated build tree was removed; source and logs were retained privately. Model files and running services were preserved.

## What AMD support still needs

Wire the COW session implementation and inference driver into the HIP build, resolve actual compilation/runtime differences, and validate capture, isolation, integrity, disk restore, samplers and the separate-client workflow on real AMD hardware. Provide enough disk capacity for the configured cache and build. CPU setup tests and earlier memory-only GPU tests cannot replace those checks.

See [result.json](result.json) for measurements, test names, fingerprints and limitations.
