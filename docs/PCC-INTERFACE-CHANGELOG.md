# PCC interface change record

## Policy for the model-platform refactor

BW and SSS may be rewritten later. Keeping their current interfaces working is
not a release requirement. PCC may change HTTP, Python, CLI, capabilities and
native adapter interfaces deliberately. Document each delta so a future client
author does not need to reconstruct it from implementation commits.

Changing an API does not relax immutable history/controller/identity guarantees.
Persistent formats require their own versioned migration and rollback decision.
Detectable incompatible versions and unsupported request shapes must get clear
errors. Some old clients provide no version and cannot be distinguished; record
those limits and the deployment/version selection that prevents silent semantic
reinterpretation.

## Current status, 2026-11-07

**No runtime interface changes shipped by this planning revision.** Existing
interfaces remain as implemented. The registry, operation contract, errors and
state API in the plan are proposed designs, not callable endpoints.

Planned decision areas, finalized in P01:
- One checked backend registration/factory route across CLI and service.
- Constrained independent operation descriptors instead of inferred support.
- Explicit restore versus reconstruction and complete-state compatibility.
- Versioned errors, cancellation and committed-result recovery semantics.
- Native/Python adapter ownership, leases and complete state manifests.

Do not assign real wire names/version numbers solely from these examples.
Inspect existing negotiation and choose the smallest coherent change.

## Required entry for every implemented change

Copy this template, replace every placeholder, and keep the status truthful:

```text
Change ID / title:
Status: proposed | implemented-unreleased | released | reverted
Source commits / contract version:
Affected surfaces: HTTP / Python import / CLI / capability / native ABI / storage
Problem and reason for change:
Old interface and exact semantics:
New interface and exact semantics:
Old request/example -> response/error:
New request/example -> response/error:
Version negotiation, unsupported-version behavior and unversioned detection limits:
Identity implications (model/profile/controller/codec):
Persisted-data migration: none, or exact loader/converter/rollback requirements
Cancellation, retry, ambiguous commit and resource-lifetime changes:
Current BW impact (HTTP plus imports, if any):
Current SSS impact (HTTP plus imports, if any):
Migration instructions for future clients:
Tests and source/evidence references:
Known limitations and untested consumers:
Removal/deprecation decision, if applicable:
```

Examples use synthetic public fixture data. Do not publish authentication tokens,
private prompts, real machine addresses or filesystem access handoffs. Commit the
reproducible interface tests and reviewed contract changes. Raw operational logs
remain outside source Git.

## Acceptance

The controller checks this record before accepting a slice with interface changes.
The PCC-owned conformance client tests actual version/error/continuation behavior.
“No BW changes” is not evidence of compatibility; compatibility stays untested
unless someone later runs the relevant client. A future rewrite is allowed, and
its author should be able to derive required behavior from this record and tests.

## PCC-CONSTRUCT-001: checked shared backend construction

Status: implemented-unreleased on the model-platform feature branch.
Affected surfaces: internal Python construction and invalid HTTP worker profiles.
See [BACKEND-REGISTRY.md](BACKEND-REGISTRY.md) for source entry points and tests.

Before: BranchingModel selected four known backends and rejected unknown names;
the HTTP worker repeated three constructors and otherwise attempted Strata.
After: both use one lazy checked registry. Known supported constructors receive
the same profile object and arguments. Unknown/malformed names reject with
IntegrityError; native constructor errors propagate without engine fallback.
Legacy native-Llama local loading preserves its default. It has no selected-worker
interface, and HTTP worker construction now explicitly refuses that unsupported
combination rather than attempting Strata with the wrong profile.

No successful wire request/response, capability schema, model/profile identity,
checkpoint format, controller boundary or training operation changes. No persisted
migration or rollback conversion is required. No new inference or training
qualification is granted by registration. Auth, publication and cancellation
semantics remain in their existing owners.

Migration: Python callers may use backend_registry.load_backend(profile,gpu_id)
or continue BranchingModel.load_backend(profile). A custom local registry uses
BackendDefinition and an immutable tuple of definitions. Invalid-profile callers
must select a supported backend and retain its native/profile validation; they
cannot rely on the old fallback. BW/SSS consumer compatibility is not tested or
required for this slice. Unversioned external callers still use the existing API;
no semantic reinterpretation of valid requests is introduced here.

Tests: constructor selection/arguments, unknown-before-import, malformed profiles,
legacy worker refusal, duplicate/custom registration, constructor-error identity,
existing protocol/HTTP capability fixtures, G0 and Python training controls.
Mocked constructors do not prove actual GPU behavior or native math. Native
Qwen3.6 state/API and constrained operation admission remain subsequent work.

## PCC-DFLASH2-001: opt-in verified draft windows

Status: implemented-unreleased, experimental. Contract version:
verified-window-v1 / verified-dflash2-consumed-v1.

Affected surfaces: setup CLI, worker capability, native private pipe,
execution receipts and execution-profile identity. Existing HTTP request shapes
are unchanged.

Before: Flash generated tokens with the normal verifier, whose work counter
matched consumed input and output tokens. After: a server may opt in with both
setup options described in [COW-DFLASH2.md](COW-DFLASH2.md). The native worker
checks proposals using the target and commits selected tokens individually.
Its speculative receipt records committed-token boundaries. The separate
accepted and forward counters distinguish accepted progress from verification
work. Other profiles retain their prior accounting rules.

Setup rejects a missing half of the asset pair, dense-model selection and an
image producer. The native worker reports its protocol at startup; a mismatch
with the pinned execution profile fails admission. DFlash2 draft assets are
included in the execution-profile fingerprint, not the base-model weights
contract. A checkpoint remains bound to the execution profile that created it;
there is no automatic migration or relabeling. Resuming a checkpoint under a
different execution profile has not been qualified.

A completed token remains the publication and cancellation boundary. A
speculative suffix is discarded before capture, cancellation, restore or
process exit. This does not add a new durable draft state. The ordinary
request, retry, authorization and branch APIs retain their existing shapes.

BW and SSS imports and workflows were not specifically qualified by this
change. Existing clients need no new request fields; they must not interpret
forward work as accepted output count. New server profiles require both pinned
draft files, sufficient GPU memory and a CUDA build. Omitting both setup options
selects the existing worker.

Focused tests cover receipt validation, native state restoration and greedy
cancellation boundaries. The mode remains experimental pending the broader
lifecycle, client and backend matrix; it is text-only and no HIP/SYCL,
split-device or batched DFlash2 capability is claimed.
