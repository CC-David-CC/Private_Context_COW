"""Bounded immutable grammar groups; no HTTP admission or persistence yet.

A group receives content-derived public state names. Adding another group never
rebuilds an old engine. Mutable parser progress remains in canonical controllers.
Durable definitions and request routing must be implemented before this can serve
restartable client registrations; this class alone makes no such promise.
"""
import json
import threading
from serve.cow_grammar import GrammarRegistry
from context_time_travel.store import InvalidOperation

class GrammarCatalog:
    def __init__(self,tokenizer,eos_tokens,*,max_groups=64,max_source_bytes=64*1024*1024):
        if type(max_groups) is not int or not 1<=max_groups<=256:
            raise ValueError('bounded positive group count required')
        if type(max_source_bytes) is not int or not 1<=max_source_bytes<=256*1024*1024:
            raise ValueError('bounded positive source budget required')
        self._tokenizer=tokenizer
        self._eos=tuple(eos_tokens)
        self._max_groups=max_groups
        self._max_bytes=max_source_bytes
        self._bytes=0
        self._groups={}
        self._by_state={}
        self._lock=threading.RLock()

    def register(self,config):
        # The existing loader performs all source, parser and tokenizer checks.
        # Its engine identity binds definitions, tokenizer, EOS and compiler.
        try:
            detached=json.loads(json.dumps(config,ensure_ascii=False,allow_nan=False))
        except (TypeError,ValueError,UnicodeError,RecursionError) as error:
            raise InvalidOperation('Grammar configuration must be finite JSON') from error
        bare=GrammarRegistry.from_config(detached,self._tokenizer,self._eos)
        group_id=bare.engine_id
        with self._lock:
            old=self._groups.get(group_id)
            if old is not None:return json.loads(old[1])
            if len(self._groups)>=self._max_groups:
                raise InvalidOperation('Grammar group capacity exhausted')
            # Normalize through the existing engine's public immutable spec:
            # serialize caller data only after compilation validated the schema.
            raw=json.dumps(detached,sort_keys=True,ensure_ascii=False).encode('utf-8')
            if self._bytes+len(raw)>self._max_bytes:
                raise InvalidOperation('Grammar source capacity exhausted')
            names={name:'g'+group_id+'_s'+str(i)
                   for i,name in enumerate(sorted(detached['states']))}
            aliased=dict(schema=1,states={names[name]:definition
                for name,definition in detached['states'].items()})
            registry=GrammarRegistry.from_config(aliased,self._tokenizer,self._eos)
            if any(name in self._by_state for name in names.values()):
                raise InvalidOperation('Grammar group name collision')
            receipt=dict(schema=1,group_id=group_id,states=names,
                engine_id=registry.engine_id,spec_sha256=registry.spec_sha256,
                tokenizer_sha256=registry.tokenizer_sha256,
                llguidance_version=registry.llguidance_version)
            encoded=json.dumps(receipt,sort_keys=True)
            # Publish only after every compile/capacity check succeeds.
            self._persist_group(detached,receipt)
            self._groups[group_id]=(registry,encoded)
            for name in names.values():self._by_state[name]=registry
            self._bytes+=len(raw)
            return json.loads(encoded)

    def _persist_group(self,config,receipt):
        """In-memory catalog; durable subclass must publish before admission."""
        return None

    @property
    def states(self):
        with self._lock:return tuple(sorted(self._by_state))

    def resolve(self,state=None,*,controller=None):
        with self._lock:
            if controller is not None:
                if not isinstance(controller,dict):
                    raise InvalidOperation('Saved grammar controller required')
                constraint=controller.get('constraint')
                if not isinstance(constraint,dict):
                    raise InvalidOperation('Saved grammar identity is missing')
                saved=constraint.get('agent_state')
                if state is not None and state!=saved:
                    raise InvalidOperation('Requested grammar differs from saved state')
                state=saved
            registry=self._by_state.get(state) if isinstance(state,str) else None
            if registry is None:
                raise InvalidOperation('Unknown immutable grammar group state')
            if controller is not None and not registry.matches_controller(controller):
                raise InvalidOperation('Saved grammar identity is incompatible')
            return registry
