# External checkpoint lifecycle diagnostics

This is an opt-in external observer for the existing P4 lifecycle. It records bounded nested inclusive/exclusive timings on a per-thread basis for native request writes/loads, serialization, `export.state` readback, checkpoint digests, pool admission/validation, prepared checkpoints, payload publication/fsync, and checkpoint validation/load. Nested timings are not additive across parents and children. Instrumentation errors are recorded and do not replace a wrapped function's return value or exception.

The wrapper does not alter production files, native code, checkpoint bytes, integrity checks, or the P4 runner. No lifecycle inference was run under the wrapper, so this work makes no performance or attribution claim about the measured campaign. Instrumentation overhead has not been measured.

## CPU validation

Scratch source: `/home/dflanag3/branching-g3/checkpoint-lifecycle-diagnostics-luna-20261015`.

Pinned lifecycle modules imported from the frozen P5 source: `/home/dflanag3/branching-g3/p5-q4-root24b4-source-20261015`.

Executed command, through the verified `run-command.sh` helper:

```sh
env PYTHONPATH=. CTT_DIAGNOSTIC_SOURCE_ROOT=/home/dflanag3/branching-g3/p5-q4-root24b4-source-20261015 \
  python3 -m unittest \
  experimental.branching_context.tests.implementation.test_checkpoint_lifecycle_diagnostics \
  experimental.branching_context.tests.implementation.test_checkpoint_lifecycle_pinned_binding
```

Receipt: `command.exit=0`, `wrapper.exit=0`, 7 tests passed in 0.046 s, no skips/errors/failures. The CLI refusal messages in stderr are expected assertions covering protected output paths. Tests cover bounded event retention, nested timing accounting, per-thread separation, exception identity and cleanup, named hook attribution, CLI output safety, and binding to the pinned lifecycle modules without constructing a model.

The initial test-first run had one test-fixture failure: its callback used the wrong argument arity, which the observer correctly isolated as an instrumentation error. The fixture was corrected; this did not require a production behavior change. No compiler, GPU, model, or server process was used.

## Activation for a future measurement

Use the exact source tree and P4 arguments selected for that measurement, with a new diagnostics receipt outside the source tree:

```sh
python3 "$SRC/experimental/branching_context/scripts/checkpoint_lifecycle_diagnostics.py" \
  --source-root "$SRC" \
  --diagnostics "$ART/lifecycle-diagnostics.json" \
  --source-store ... --source-result ... --prepare ... --profile ... --fixture ... \
  --output ... --arm ... --arm-order ... --order-seed ... \
  --ram-bytes ... --disk-bytes ... --min-available-bytes ...
```

The trailing P4 arguments are forwarded unchanged. Run only under a separately authorized measurement window. Existing P4 integrity gates remain active. This wrapper is external diagnostic tooling, not a qualified performance result.
