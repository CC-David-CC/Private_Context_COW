# Operation declarations and admission, version 1

This is an internal pure-Python contract, not a new HTTP endpoint or a change to
existing native profiles. `context_time_travel.operation_contracts` has no runtime
wiring. It loads no model, reads no artifact, reserves no GPU and executes no
qualification. Existing serving and training behavior is unchanged by this slice.

## Independent operations

The seven names are `inference`, `lora_inference`, `lora_training`, `full_training`,
`optimizer_resume`, `checkpoint_restore` and `reconstruction`. Each needs its own
explicit declaration. An implemented inference path supplies no training claim;
a qualified restore supplies no reconstruction claim. A future contract revision
can add operations after defining their semantics and qualification requirements.

Each immutable `OperationDeclaration` specifies:

- An implementation version and independent implementation status: `implemented`,
  `absent` or `unknown`. Unknown means inspection is incomplete, not known absence.
- Qualification status `unqualified` or `qualified`, independently of owner policy
  `policy_enabled`. Disabled qualified code remains disabled. Enabling policy
  does not qualify code.
- Exact model and execution-profile identities as64-character lowercase hex.
  These are existing resolved identities, not paths or display names.
- Exact continuation/optimizer codec identifier, or `None` when no serialized
  codec is applicable. Do not fabricate a codec for plain inference/training.
- Allowed format and device identifiers. These are opaque canonical identifiers
  chosen by the checked profile builder; this module does not infer GPU support.
- Named inclusive `NumericLimit` bounds. Default kind `integer` requires exact
  integer request values (no booleans or rounding). Kind `number` accepts finite
  integer/float quantities for dimensions such as adapter scale. There are no
  implicit unit conversions; the contract author defines units in the name/spec.
- Required prerequisite names and explicit ownership/cancellation semantics.
- Immutable evidence references: location, artifact SHA256, source revision and
  the exact declaration scope digest.

Formats, devices, limits, prerequisites and evidence are tuples of immutable
records. Mutable lists/dicts, duplicate dimensions, NaN/infinity and malformed
identities are rejected as `ContractError`. Declarations for different profiles
may coexist. The tuple `(operation, model_id, profile_id, codec)` is unique; even
two identical declarations with the same identity are refused as ambiguous.

## Evidence and authority

The deterministic `scope_sha256` binds operation, implementation version, exact
identities, formats/devices, bounds, prerequisites and lifecycle semantics.
Qualification claims and policy switches are deliberately excluded. A qualified
operation must be implemented and have explicit evidence references matching this
scope. Changing a bound or lifecycle promise invalidates the old evidence scope.

**A matching hash is not evidence that a test ran.** The trusted owner/controller
must verify receipt provenance, actual executed results, current source relevance
and the entire declared envelope before constructing a qualified declaration.
`EvidenceReference` does not read its location or verify artifact bytes. An
`AdmittedOperation` therefore exposes `evidence_verified=False`; it reports the
owner's declaration, not a fresh verification. Historical receipts never become
new runtime qualification merely because they parse. Client requests must not be
allowed to supply their own trusted declarations or prerequisite attestations.

Example construction, with placeholders describing values the owner must verify:

```python
from dataclasses import replace
from context_time_travel.operation_contracts import (
    OperationDeclaration, EvidenceReference, NumericLimit,
    OperationRequest, Quantity, admit,
)

# model_id/profile_id come from checked existing identity generation.
d = OperationDeclaration(
    operation='inference', implementation_version='example-native-v1',
    implementation_status='implemented', qualification_status='unqualified',
    policy_enabled=True, model_id=model_id, profile_id=profile_id, codec=None,
    formats=('Q4_K',), devices=('CUDA:sm89',),
    limits=(NumericLimit('context_tokens', 1, 2048),),
    prerequisites=('weights_verified',),
    ownership='worker owns native lease; caller owns immutable result',
    cancellation='before dispatch no mutation; during execution at consumed boundary',
)
# Only after the owner verifies the receipt against this entire scope:
e = EvidenceReference(receipt_location, receipt_sha256, source_revision, d.scope_sha256)
d = replace(d, qualification_status='qualified', evidence=(e,))
r = OperationRequest('inference', model_id, profile_id, None,
    'Q4_K', 'CUDA:sm89', (Quantity('context_tokens', 128),), ('weights_verified',))
plan = admit((d,), r)
```

## Admission and errors

`admit(declarations, request, allow_unqualified=False)` resolves one exact
operation/model/profile/codec, then checks implementation, policy, format, device,
every limit, prerequisites and qualification. All checks occur before the caller
allocates native state. Unknown limit dimensions and omitted required dimensions
are errors. No model, precision, device, limit or restore/replay fallback occurs.

Owner-only `allow_unqualified=True` permits an explicit experiment only after all
other constraints pass. The returned plan retains `qualification_status='unqualified'`
and `unqualified_opt_in=True`. It never modifies the declaration or promotes its
support. A disabled policy or absent implementation cannot be bypassed this way.

A plan is not a security credential, resource reservation or live lease. The
runtime must verify current permissions/resources, acquire actual leases and
honor declared cancellation/ownership. Prerequisite names are attestations from
that trusted path; this pure planner cannot verify facts such as a file hash or
currently available memory. Resource estimation/leases and implementation-defined
nonnumeric constraints need their own checked adapters, not arbitrary extra flags.

`AdmissionError` exposes stable `code`, requested `operation`, explanatory `reason`
and `next_action`, plus `to_json()`. The codes in v1 are:

| Code | Meaning |
| --- | --- |
| `operation_unknown` | Name is outside this contract version |
| `operation_undeclared` | Known operation has no supplied declaration |
| `operation_identity_mismatch` | No exact model/profile/codec declaration |
| `implementation_unknown` | Availability has not been established |
| `implementation_absent` | Implementation is explicitly absent |
| `operation_policy_disabled` | Owner policy disables it |
| `operation_format_mismatch` | Format is outside the declared set |
| `operation_device_mismatch` | Device is outside the declared set |
| `operation_limit_unknown` | Request supplied an undeclared dimension |
| `operation_limit_missing` | A required dimension was omitted |
| `operation_limit_type_mismatch` | Integer-only dimension received a fraction/float |
| `operation_limit_exceeded` | Value falls outside inclusive bounds |
| `operation_prerequisite_missing` | Required trusted attestation absent |
| `operation_unqualified` | Qualification absent and no explicit experiment opt-in |

Malformed/ambiguous schemas raise `ContractError` before resolution. This is not
an HTTP error schema; a later service integration must map these errors explicitly,
version its interface and reject untrusted claims. No commit status exists here
because this pure function has no externally committed effects.

## Qualification limits of this slice

The focused tests exercise all seven independent operation names, strict scope
selection, rejection paths, integer/continuous bounds, immutable records and
malformed inputs. Mutation checks must show that bypassing qualification or limits
and making records mutable is detected. These are contract tests only, not evidence
that any model implements or passes the declared operation.

Run from `experimental/branching_context`:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover \
  -s tests/implementation -p test_operation_contracts.py -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify.py --gate G0 --no-write
```

No persisted schema, native ABI, model identity algorithm, HTTP or existing
consumer interface changes in this slice. Future factory/admission wiring needs
its own tests and entry in the PCC interface change record.
