"""Backend facts exposed only after the adapter validates its native binding.

Descriptors are observations, never authority to import a checkpoint. The normal
model/profile, payload, controller and input checks still apply at admission.
"""
from dataclasses import asdict, dataclass
import re


class CapabilityError(ValueError):
    pass


@dataclass(frozen=True)
class NativeCapabilities:
    backend: str
    family: str
    model_contract_id: str
    execution_profile_id: str
    inputs: tuple[str, ...]
    state_format: str
    resume_contract: str
    max_context: int
    sampling_protocol: str | None
    selected_token_protocol: str | None
    grammar_selection_protocol: str | None
    checkpoint_export: bool = True
    checkpoint_import: bool = True
    native_residency: bool = False
    physical_prefix_sharing: bool = False
    worker_topology: str = 'independent'

    def to_json(self):
        result = asdict(self)
        result['inputs'] = list(self.inputs)
        return result


def _identity(profile):
    for key in ('model_contract_id', 'execution_profile_id'):
        if type(profile.get(key)) is not str or not re.fullmatch('[0-9a-f]{64}', profile[key]):
            raise CapabilityError('Capability requires a checked ' + key)
    execution = profile.get('execution', {})
    capacity = execution.get('capacity')
    if type(capacity) is not int or not 1 <= capacity < 2**31:
        raise CapabilityError('Capability requires a bounded native context')
    return execution


def describe_dense(profile):
    e = _identity(profile)
    mixed = e.get('state_format') == 'qwen35-hybrid-mrope-frontier-v2'
    expected = ('identical-mixed-native-operators-v1' if mixed
                else 'identical-native-batches-v1')
    llama=profile.get('backend')=='llama-dense-llama.cpp'
    formats=(('llama-dense-frontier-v1',) if llama else ('qwen35-hybrid-frontier-v1','qwen35-hybrid-mrope-frontier-v2'))
    if (profile.get('backend') not in ('qwen35-llama.cpp','llama-dense-llama.cpp')
            or e.get('state_format') not in formats
            or e.get('resume_contract') != expected
            or e.get('generated_operator') != 'llama-forward-consumed-v1'
            or e.get('selected_token_protocol') != 'consumed-selected-receipt-v1'
            or e.get('sampling_protocol') != 'full-vocab-min-p-u53-v2'):
        raise CapabilityError('Unknown dense native continuation protocol')
    if mixed and (e.get('vision_protocol') != 'explicit-image-rows-mrope-v3'
                  or e.get('native_binding_protocol') != 'qwen35-streamed-cells-mrope-v2'):
        raise CapabilityError('Unknown dense mixed-input protocol')
    return NativeCapabilities(profile['backend'], 'dense', profile['model_contract_id'],
        profile['execution_profile_id'], ('text', 'image') if mixed else ('text',),
        e['state_format'], expected, e['capacity'], e['sampling_protocol'],
        e['selected_token_protocol'], 'finite-mask-argmax-v1')


def describe_strata(profile, ready):
    e = _identity(profile)
    if (profile.get('backend') != 'strata'
            or e.get('resume_contract') not in ('identical-native-operators-v1', 'identical-mixed-native-operators-v1')
            or e.get('generated_operator') != 'normal-verifier-decode'):
        raise CapabilityError('Unknown Strata continuation protocol')
    for key in ('native_request_protocol', 'selected_token_protocol',
                'grammar_selection_protocol', 'sampling_protocol'):
        if e.get(key) is not None and ready.get(key) != e[key]:
            raise CapabilityError('Native capability differs from pinned ' + key)
    mixed = e['resume_contract'] == 'identical-mixed-native-operators-v1'
    if mixed and e.get('vision_protocol') != 'explicit-image-rows-mrope-v3':
        raise CapabilityError('Unknown Strata mixed-input protocol')
    # The state codec is bound to the complete execution profile. This label
    # deliberately conveys no portability across builds or GPU layouts.
    return NativeCapabilities('strata', 'moe', profile['model_contract_id'],
        profile['execution_profile_id'], ('text', 'image') if mixed else ('text',),
        'execution-profile-bound', e['resume_contract'], e['capacity'],
        e.get('sampling_protocol'), e.get('selected_token_protocol'),
        e.get('grammar_selection_protocol'), native_residency=True)


def feature(supported, enabled=None, *, qualification='startup-checked'):
    if type(supported) is not bool or (enabled is not None and type(enabled) is not bool):
        raise CapabilityError('Feature states must be booleans')
    enabled = supported if enabled is None else enabled
    if enabled and not supported:
        raise CapabilityError('Unsupported feature cannot be enabled')
    return dict(supported=supported, enabled=enabled,
                qualification=qualification if supported else 'unsupported')
