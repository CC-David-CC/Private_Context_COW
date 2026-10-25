# Immutable MiniCPM LoRA evaluation through COW

PCC can bind one LoRA adapter to its dense MiniCPM base using the existing native
per-lease adapter API. This is an opt-in execution path. The original base-only
profiles and live full-weight studies keep their frozen runtimes.

The base GGUF and the adapter are two immutable artifacts. An updated hat creates
a new model contract and execution profile; it never edits an existing context.
Changing the adapter scale also changes both identities. Moving the same bytes
to another file path does not change their semantic identity.

## Owner workflow

1. Build into a new output directory with
   `experimental/branching_context/scripts/build_llama_dense.py --enable-lora`,
   supplying the ordinary `--llama`, `--engine-build` and `--output` arguments.
   This declares the existing native API, not numerical qualification. Leave
   live engine builds and existing proof files untouched.
2. Prepare a profile with `prepare_llama_dense_profile.py`, adding
   `--lora /absolute/adapter.gguf --lora-scale 1`. Use the same explicit numerical
   policy, context and device selection as the intended assessment.
3. Owner drivers that already have a pinned profile can first use
   `tools.pcc_cow_profile.bind_dense_profile()` for the base and placement, then
   `bind_dense_adapter()` with the exact adapter path, SHA256 and scale.
   The optional COW runtime must be importable. Binding does not grant a GPU,
   load a model or certify a build.
4. Qualify attached-adapter execution against the independent fresh native
   oracle with that same base, adapter and scale. Keep declared numerical
   tolerances. Test restoration and rejection of another adapter's snapshots.
5. Run S³'s existing COW evaluator. It records PCC's composite model contract,
   keeps the base file SHA separately, and checks the adapter again at close.

The existing checked native loader attaches the hat before any context input.
No second adapter loader, sampler, cache or training runtime was added. COW's
ordinary contract checks prevent restoring another hat's checkpoints.

## Evidence and limits

CPU integrity tests cover changed versions and scales, an unchanged base profile,
foreign-base contracts, unbound identities, unsupported build declarations,
symlinks, changed artifact bytes and mutation detected when evaluation closes.
S³ tests reject mixed-adapter shards even when their base GGUF hashes match.

Native GPU qualification for this new execution path is pending. Passing these
CPU controls is not evidence of numerical equivalence or learning quality.
The earlier adapter-versus-merged-weight gate remains failed and unchanged; see
[the measured diagnostics](LORA-MERGE-DIAGNOSTICS.md). A directly attached hat and
a merged deployment are distinct measurements. Neither substitutes for the other.

Full-weight and LoRA learning curves must use the same practice sequence and
update counts, report their actual execution identities, and measure learning
and forgetting. Training time is an observed cost, not the comparison's stopping
criterion. No candidate is promoted merely because this adapter loads.
