"""Pure, immutable operation declarations and fail-closed envelope admission.

This module performs no model loading, allocation, artifact IO, or evidence
execution. A trusted owner supplies qualification declarations. A valid schema
or scope digest cannot prove that a referenced qualification was actually run.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

OPERATIONS = (
    'inference', 'lora_inference', 'lora_training', 'full_training',
    'optimizer_resume', 'checkpoint_restore', 'reconstruction',
)


class ContractError(ValueError):
    """Malformed or ambiguous declaration/request; no admission performed."""


class AdmissionError(ValueError):
    """Stable refusal category with a human-readable reason and next action."""
    def __init__(self, code, operation, reason, next_action):
        self.code = code
        self.operation = operation
        self.reason = reason
        self.next_action = next_action
        super().__init__(f'{code}: {reason}')

    def to_json(self):
        return dict(code=self.code, operation=self.operation, reason=self.reason,
                    next_action=self.next_action)


def _text(value, field):
    if type(value) is not str or not value.strip():
        raise ContractError(f'{field} must be a nonempty string')


def _hash(value, field, lengths=(64,)):
    if (type(value) is not str or len(value) not in lengths
            or re.fullmatch('[0-9a-f]+', value) is None):
        raise ContractError(f'{field} must be a lowercase hexadecimal identity')


def _number(value, field):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ContractError(f'{field} must be a finite number, not a boolean')


def _tuple(values, field, item_type, *, nonempty=False):
    if type(values) is not tuple or (nonempty and not values):
        raise ContractError(f'{field} must be an immutable tuple')
    if any(type(item) is not item_type for item in values):
        raise ContractError(f'{field} contains an invalid record type')


def _strings(values, field, *, nonempty=False):
    _tuple(values, field, str, nonempty=nonempty)
    for value in values:
        _text(value, field)
    if len(set(values)) != len(values):
        raise ContractError(f'{field} contains duplicate values')


def _identity(model_id, profile_id, codec):
    _hash(model_id, 'model_id')
    _hash(profile_id, 'profile_id')
    if codec is not None:
        _text(codec, 'codec')


@dataclass(frozen=True, slots=True)
class NumericLimit:
    name: str
    minimum: int | float
    maximum: int | float
    kind: str = 'integer'

    def __post_init__(self):
        _text(self.name, 'limit name')
        _number(self.minimum, 'limit minimum')
        _number(self.maximum, 'limit maximum')
        if self.kind not in ('integer', 'number') or type(self.kind) is not str:
            raise ContractError('Unknown limit numeric kind')
        if self.kind == 'integer' and (type(self.minimum) is not int or type(self.maximum) is not int):
            raise ContractError('Integer limit requires integer bounds')
        if self.minimum > self.maximum:
            raise ContractError('Limit minimum exceeds maximum')


@dataclass(frozen=True, slots=True)
class Quantity:
    name: str
    value: int | float

    def __post_init__(self):
        _text(self.name, 'quantity name')
        _number(self.value, 'quantity value')


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    location: str
    sha256: str
    source_revision: str
    scope_sha256: str

    def __post_init__(self):
        _text(self.location, 'evidence location')
        _hash(self.sha256, 'evidence sha256')
        _hash(self.source_revision, 'evidence source revision', (40, 64))
        _hash(self.scope_sha256, 'evidence scope')


@dataclass(frozen=True, slots=True)
class OperationDeclaration:
    operation: str
    implementation_version: str
    implementation_status: str
    qualification_status: str
    policy_enabled: bool
    model_id: str
    profile_id: str
    codec: str | None
    formats: tuple[str, ...]
    devices: tuple[str, ...]
    limits: tuple[NumericLimit, ...]
    prerequisites: tuple[str, ...]
    ownership: str
    cancellation: str
    evidence: tuple[EvidenceReference, ...] = ()

    def __post_init__(self):
        if self.operation not in OPERATIONS or type(self.operation) is not str:
            raise ContractError('Unknown declared operation')
        _text(self.implementation_version, 'implementation_version')
        if type(self.implementation_status) is not str or self.implementation_status not in ('implemented', 'absent', 'unknown'):
            raise ContractError('Unknown implementation status')
        if type(self.qualification_status) is not str or self.qualification_status not in ('unqualified', 'qualified'):
            raise ContractError('Unknown qualification status')
        if type(self.policy_enabled) is not bool:
            raise ContractError('policy_enabled must be boolean')
        _identity(self.model_id, self.profile_id, self.codec)
        _strings(self.formats, 'formats', nonempty=True)
        _strings(self.devices, 'devices', nonempty=True)
        _strings(self.prerequisites, 'prerequisites')
        _tuple(self.limits, 'limits', NumericLimit)
        if len({item.name for item in self.limits}) != len(self.limits):
            raise ContractError('Duplicate limit name')
        _text(self.ownership, 'ownership')
        _text(self.cancellation, 'cancellation')
        _tuple(self.evidence, 'evidence', EvidenceReference)
        if self.qualification_status == 'qualified':
            if self.implementation_status != 'implemented' or not self.evidence:
                raise ContractError('Qualified operation requires implementation and explicit evidence')
        if any(item.scope_sha256 != self.scope_sha256 for item in self.evidence):
            raise ContractError('Evidence scope differs from operation constraints')

    @property
    def identity_key(self):
        return self.operation, self.model_id, self.profile_id, self.codec

    @property
    def scope_sha256(self):
        # Policy and claimed qualification are separate from the verified scope.
        scope = dict(schema=1, operation=self.operation,
            implementation_version=self.implementation_version,
            model_id=self.model_id, profile_id=self.profile_id, codec=self.codec,
            formats=self.formats, devices=self.devices,
            limits=[asdict(item) for item in self.limits],
            prerequisites=self.prerequisites, ownership=self.ownership,
            cancellation=self.cancellation)
        encoded = json.dumps(scope, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class OperationRequest:
    operation: str
    model_id: str
    profile_id: str
    codec: str | None
    format: str
    device: str
    limits: tuple[Quantity, ...]
    prerequisites: tuple[str, ...]

    def __post_init__(self):
        _text(self.operation, 'operation')
        _identity(self.model_id, self.profile_id, self.codec)
        _text(self.format, 'format')
        _text(self.device, 'device')
        _tuple(self.limits, 'limits', Quantity)
        if len({item.name for item in self.limits}) != len(self.limits):
            raise ContractError('Duplicate requested limit')
        _strings(self.prerequisites, 'prerequisites')


@dataclass(frozen=True, slots=True)
class AdmittedOperation:
    declaration: OperationDeclaration
    request: OperationRequest
    unqualified_opt_in: bool

    @property
    def qualification_status(self):
        return self.declaration.qualification_status

    @property
    def scope_sha256(self):
        return self.declaration.scope_sha256

    @property
    def evidence_verified(self):
        # The caller still owns evidence verification and runtime prerequisites.
        return False


def admit(declarations, request, *, allow_unqualified=False):
    """Resolve one exact envelope without IO, allocation or semantic fallback.

    Only an owner-controlled caller may set allow_unqualified or supply claims
    about verified prerequisites. This is planning, not a permission credential
    or a runtime reservation; dispatch must validate still-current resources.
    """
    _tuple(declarations, 'declarations', OperationDeclaration)
    if type(request) is not OperationRequest:
        raise ContractError('request must be an OperationRequest')
    if type(allow_unqualified) is not bool:
        raise ContractError('allow_unqualified must be boolean')
    keys = [item.identity_key for item in declarations]
    if len(set(keys)) != len(keys):
        raise ContractError('Duplicate or ambiguous operation identity')

    def refuse(code, reason, next_action):
        raise AdmissionError(code, request.operation, reason, next_action)

    if request.operation not in OPERATIONS:
        refuse('operation_unknown', 'Requested operation is not defined by this contract',
               'Select a named operation or negotiate a newer contract version')
    options = [item for item in declarations if item.operation == request.operation]
    if not options:
        refuse('operation_undeclared', 'No declaration exists for this operation',
               'Inventory and declare the operation; do not infer support from another operation')
    wanted = request.operation, request.model_id, request.profile_id, request.codec
    matching = [item for item in options if item.identity_key == wanted]
    if not matching:
        refuse('operation_identity_mismatch', 'No declaration matches model, profile and codec',
               'Use a declared exact identity or explicitly qualify a new profile')
    d = matching[0]
    if d.implementation_status == 'unknown':
        refuse('implementation_unknown', 'Implementation availability has not been established',
               'Inspect the implementation and record its availability')
    if d.implementation_status == 'absent':
        refuse('implementation_absent', 'Operation implementation is declared absent',
               'Implement this operation or select an implemented operation explicitly')
    if not d.policy_enabled:
        refuse('operation_policy_disabled', 'Owner policy disables this operation',
               'Request an owner policy decision; qualification alone does not enable it')
    if request.format not in d.formats:
        refuse('operation_format_mismatch', 'Requested format is outside the declaration',
               'Select an explicitly supported format; no implicit conversion is performed')
    if request.device not in d.devices:
        refuse('operation_device_mismatch', 'Requested device is outside the declaration',
               'Select an explicitly supported device/profile')
    provided = {item.name: item.value for item in request.limits}
    expected = {item.name: item for item in d.limits}
    if provided.keys() - expected.keys():
        refuse('operation_limit_unknown', 'Request contains undeclared limit dimensions',
               'Declare and qualify those dimensions before use')
    if expected.keys() - provided.keys():
        refuse('operation_limit_missing', 'Request omits required limit dimensions',
               'Supply explicit values for every declared dimension')
    for name, limit in expected.items():
        if limit.kind == 'integer' and type(provided[name]) is not int:
            refuse('operation_limit_type_mismatch', f'Requested {name} must be an integer',
                   'Supply an exact integer; fractional values are not rounded')
        if not limit.minimum <= provided[name] <= limit.maximum:
            refuse('operation_limit_exceeded', f'Requested {name} is outside its declared bounds',
                   'Choose a value within this profile or qualify a larger envelope')
    if set(d.prerequisites) - set(request.prerequisites):
        refuse('operation_prerequisite_missing', 'Required prerequisite attestations are missing',
               'Verify prerequisites through the owner-controlled admission path')
    if d.qualification_status != 'qualified' and not allow_unqualified:
        refuse('operation_unqualified', 'Operation is implemented but not qualified for this envelope',
               'Run qualification or explicitly opt into an unqualified experiment')
    return AdmittedOperation(d, request, d.qualification_status != 'qualified')
