Start with [Linux server setup](docs/CONTEXT-COW-SERVER.md) and the [BranchWrite client guide](https://github.com/CC-David-CC/BranchWrite/blob/main/docs/LINUX-QUICKSTART.md).

# Context COW

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

This is source, not a turnkey prebuilt server. Operators must provision the
pinned model, build/probe executables and a compatible measured execution
profile. Do not import checkpoints across incompatible profiles. Cross-batch
arrangement equivalence is an accepted prototype limitation; cumulative G1-G5
certification is incomplete. See the preserved mathematical specification for
the broader original contract. An HTTP/client smoke pass is not certification.

