# Checked backend construction

`context_time_travel.backend_registry` is the shared construction entry point for
`BranchingModel.load_backend` and HTTP coordinator workers. It resolves known
backend names before importing native bindings. Unknown and malformed selections
raise `IntegrityError`; they never fall through to another engine.

Built-in constructors remain the existing Qwen35, dense-Llama, Strata and legacy
native-Llama implementations. Original profile objects and constructor arguments
are passed through unchanged. This slice changes no weights, kernels, state codecs,
model/profile hashing, training behavior or checkpoint validation.

`BackendDefinition(name, factory)` and `BackendRegistry(tuple_of_definitions)`
permit an isolated registry to select a new implementation without editing the
scheduler or store. Factories take `(profile, gpu_id)`; callers use
`load_backend(profile, gpu_id=None)`. Duplicate names reject. The built-in registry
has no mutation API. Registration is construction only, not a capability grant:
operation admission and exact qualification remain separate responsibilities.

An absent `backend` preserves the local API's historical `llama.cpp` default.
Explicit invalid values do not mean default. Legacy NativeBackend has no worker
selection API, so explicit `gpu_id` rejects before import. The HTTP path formerly
attempted Strata for this unsupported case; it now refuses. Constructor failures
propagate unchanged and never trigger a silent engine fallback.

## Validation and limits

Run from the repository root:

```sh
PYTHONPATH=.:tools:experimental/branching_context/src python3 -m unittest \
  experimental.branching_context.tests.implementation.test_backend_registry \
  experimental.branching_context.tests.implementation.test_backend_capabilities \
  tools.test_cow_gpu_selection serve.test_cow_capabilities -v
python3 experimental/branching_context/scripts/verify.py --gate G0 --no-write
```

The constructor tests use mocked native constructors, including deliberate errors.
They verify selection, unchanged arguments, identity pass-through, extensibility
and rejection. They do not establish native numerical equivalence, GPU routing on
hardware or Qwen3.6 integration. Existing protocol and training-control fixtures
provide separate bounded regression evidence. No new operation is advertised.

Setup still selects model preparation by architecture. Its model-definition and
operation-contract migration remains subsequent work. This source slice does not
claim that every model-name conditional has already been removed.
