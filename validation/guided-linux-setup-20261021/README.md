# Guided Linux installation and a real CLI website trial

The installer detects GPU memory, available system RAM and CPUs, then builds
and configures the existing native COW path. It verifies pinned model files and
keeps API keys outside the source checkout.

- Final automatic setup on RTX 4090 /32 GB system RAM: PASS, 375 seconds with existing pinned model files; native CUDA13.3 build, real packing, full hashes and new settings/key. Model transfer took1006.6 seconds separately. No full111 GB Internet download was claimed.
- Repeat setup: PASS, settings and key unchanged.
- Final wrapper startup: PASS, authenticated info ready in25.0 seconds.
- Setup policy17/17, model resources9/9, G0 51/51, HTTP fixture regressions16/16, BW connection16/16 and interactive6/6. Fixture tests are not GPU gate qualification.
- Small-host coding: TIMEOUT at600.66 seconds; two model steps completed and the CLI saved a safe resumable boundary. This is not a coding PASS.
- The same fresh GitHub-installed dev66 CLI in a separate VM, using the real Q4 server on the larger reference host: website PASS in292.10 seconds; follow-up bold names PASS in238.16 seconds; clean exit. These are complete workflow times, not decode throughput.
- Independent sandboxed Chromium: PASS at 390 x 844 and 1200 x 800. Clicked the actual button, checked visible greeting, four visible Buddy occurrences at font weight 900, and no horizontal overflow. Screenshot/source hashes are in result.json.

## Failures retained

8K context rejected the prefix; guided setup now starts at 32K. Small prefill
batches caused repeated SSD reads. The previous 14 GiB expert allocation and
hard-coded 16 GiB free-RAM guard were incompatible on the small host. Guided setup
now retains a positive 8 GiB guard there, budgets native overhead and selects 8 GiB
expert RAM plus 2 GiB context RAM. Legacy HTTP startup still defaults to 16 GiB.
Missing emoji fonts in the minimal client VM were fixed before final rendering.
The initial inspection fixture returned the wrong action-receipt shape; its
formatting was fixed, with assertions preserved.

## Scope

This is bounded installation and workflow evidence. No fresh numerical oracle
was executed here, numerical max error is null, and cumulative G1-G5 remains
unqualified. The proposed RTX 5090 /32 GB VRAM /128 GB RAM configuration and the
smaller-GPU 2048-batch policy remain untested. Existing model/controller gate
requirements and mismatch evidence are unchanged.

The optional browser worker was installed separately in the owned VM. It ran
non-root, with Chromium sandboxing and namespace retirement verified; the
standard BW installer does not yet enable model-driven browser tools.
