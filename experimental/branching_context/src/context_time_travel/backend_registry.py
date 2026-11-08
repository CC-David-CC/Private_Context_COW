"""Checked backend construction, independent of operation qualification.

Registration selects an implementation. It never grants inference, training,
restore or reconstruction support; native/profile validation remains mandatory.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
import re
from types import MappingProxyType
from typing import Callable

from .store import IntegrityError


@dataclass(frozen=True)
class BackendDefinition:
    name: str
    factory: Callable

    def __post_init__(self):
        if type(self.name) is not str or not re.fullmatch(r'[a-z][a-z0-9_.-]{0,63}', self.name):
            raise ValueError('Backend registration requires a canonical name')
        if not callable(self.factory):
            raise TypeError('Backend registration requires a constructor')


class BackendRegistry:
    def __init__(self, definitions):
        if type(definitions) is not tuple:
            raise TypeError('Backend definitions must be an immutable tuple')
        entries = {}
        for definition in definitions:
            if type(definition) is not BackendDefinition:
                raise TypeError('Expected a BackendDefinition')
            if definition.name in entries:
                raise ValueError('Duplicate backend registration: ' + definition.name)
            entries[definition.name] = definition
        self._entries = MappingProxyType(entries)

    @property
    def names(self):
        return tuple(self._entries)

    def resolve(self, profile):
        if not isinstance(profile, Mapping):
            raise IntegrityError('Backend profile must be a mapping')
        name = profile.get('backend', 'llama.cpp')
        if type(name) is not str or name not in self._entries:
            raise IntegrityError('Unsupported backend profile')
        return self._entries[name]

    def construct(self, profile, gpu_id=None):
        definition = self.resolve(profile)
        return definition.factory(profile, gpu_id)


def _dense_qwen(profile, gpu_id):
    constructor = import_module('.qwen35_backend', __package__).load_qwen35_backend
    return constructor(profile) if gpu_id is None else constructor(profile, gpu_id=gpu_id)


def _dense_llama(profile, gpu_id):
    constructor = import_module('.llama_dense_backend', __package__).LlamaDenseBackend
    return constructor(profile) if gpu_id is None else constructor(profile, gpu_id=gpu_id)


def _strata(profile, gpu_id):
    constructor = import_module('.strata_backend', __package__).StrataBackend
    return constructor(profile) if gpu_id is None else constructor(profile, gpu_id=gpu_id)


def _legacy_llama(profile, gpu_id):
    if gpu_id is not None:
        raise IntegrityError('Legacy native backend does not support worker selection')
    return import_module('.real', __package__).NativeBackend(profile)


DEFAULT_REGISTRY = BackendRegistry((
    BackendDefinition('qwen35-llama.cpp', _dense_qwen),
    BackendDefinition('llama-dense-llama.cpp', _dense_llama),
    BackendDefinition('strata', _strata),
    BackendDefinition('llama.cpp', _legacy_llama),
))


def load_backend(profile, gpu_id=None):
    return DEFAULT_REGISTRY.construct(profile, gpu_id=gpu_id)
