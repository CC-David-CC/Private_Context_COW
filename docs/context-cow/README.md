# Context COW v0.1.2-rc2 — experimental release candidate

Context COW keeps immutable conversation histories while reusing optional model
state in GPU memory, host RAM and on disk. The focus is GPU copy-on-write: share
an unchanged prefix, prepare private tail pages, and continue different branches
without changing their common parent.

This project is based on [Strata](https://github.com/Niko1221/Strata). Compatible
changes are intended to be proposed upstream over time. Publication here does
not imply that these changes have been accepted upstream.

- [Beginner setup and first chat, with screenshots](START_HERE.md)
- [Live web-app validation and literal-tag regression results](WEB_START_VALIDATION.md)
- [Build and run the COW experiment](GETTING_STARTED.md)
- [v0.1.2 math viewers and validation](V0.1.2.md)
- [LaTeX usage](../WEB_MATH.md)
- [Measurements, fingerprints and limitations](RESULTS.md)
- [Runner inventory](RUNNERS.md)
- [Priorities and design](ROADMAP.md)
- [v0.1.1 candidate plan](V0.1.1-PLAN.md)
- [Caller-managed write-integrity experiment](EXTERNAL_WRITE_INTEGRITY.md)
- [Raw reports and export manifest](results/MANIFEST.json)
- [Figure sources and provenance](figures/PROVENANCE.json)

The guarded source checkpoint is `c053ddbb`; the caller-managed probe is
`0bb06645`. The completed candidate comparison is documented in
[EXTERNAL_WRITE_INTEGRITY.md](EXTERNAL_WRITE_INTEGRITY.md). v0.1.0's release tag
remains `context-cow-v0.1.0`; candidate work is on `release/context-cow-v0.1.1`.
The current UI integration is on `release/context-cow-v0.1.2`; native/store
sources are unchanged from v0.1.1-rc1. Context COW package versions are
independent of Strata's version numbers.
