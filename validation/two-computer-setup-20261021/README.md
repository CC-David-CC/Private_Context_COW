# Two-computer setup validation

The beginner [desktop-and-laptop guide](../../docs/TWO-COMPUTER-SETUP.md) uses the
new model-preparation command. It reuses the existing downloader and model packer
without running ordinary serving setup or downloading MTP.

Executed results:

- Nine focused preparation tests passed, including real loopback HTTP range
  resume, cached-file rehashing, corruption/symlink rejection, atomic pack
  publication, failed-build cleanup and verified reuse. Pack-publication unit
  fixtures are small filesystem fixtures, not a fake GPU or real-model claim.
- Existing G0 verifier: 51/51 passed; no numerical tolerance or equation changed.
- Fresh server Python environment installed from the updated requirements;
  `pip check` passed.
- Four existing real Q4 shards were fully verified against their pinned hashes.
  The native pack was built in 68.084 seconds including that verification.
- All 10 pre-existing pack files matched the running tested server's pack
  byte-for-byte by SHA-256, including dense.bin and tokenizer/template files.
  The preparation receipt is one additional file. This does not make saved
  checkpoints interchangeable across different execution/model profiles.
- A repeat preparation exited0 in 59.967 seconds. It reverified the model and
  pack; all 11 pack file inode/size/mtime tuples stayed unchanged and the native
  packer was not invoked again.
- The novice tutorial's12 shell blocks and the advanced server page's1 block
  passed shell syntax checks.

The first real attempt exposed missing GGUF Python tooling. The next exposed
its missing PyYAML dependency. The helper now fetches the pinned tooling and the
normal requirements include the required dependencies. Dependency preflight runs
before hashing/downloading the large model. A focused test also caught an extra
nested receipt-named file being excluded from integrity checks; that was fixed
and the unchanged failing assertion now passes.

The complete111GB model download was not repeated. HTTP resume was executed with
small deterministic data through the real downloader; model preparation used
existing, fully verified Q4 files. No inference job or ordinary chat server was
started, stopped or reconfigured. Existing CUDA native code is unchanged.
No new logit-error, TPS, fresh-machine hardware matrix or G1–G5 claim follows.
The release guide points to the separate prior fresh-client/server restart smoke.
