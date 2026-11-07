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
