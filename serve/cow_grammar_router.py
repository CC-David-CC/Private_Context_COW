"""Resolve each admitted request to one retained immutable grammar engine."""
from context_time_travel.store import InvalidOperation
from serve.cow_grammar_catalog import GrammarCatalog

class GrammarRouter:
    def __init__(self,catalog,*,startup=None):
        if not isinstance(catalog,GrammarCatalog):
            raise TypeError('actual immutable grammar catalog required')
        self.catalog=catalog
        self.startup=startup

    @property
    def states(self):
        return tuple(sorted(set(self.catalog.states) |
            set(() if self.startup is None else self.startup.states)))

    def resolve_registry(self,state=None,*,controller=None):
        if controller is not None:
            if not isinstance(controller,dict):
                raise InvalidOperation('Canonical controller required')
            constraint=controller.get('constraint')
            if constraint is None:
                if state is not None:raise InvalidOperation('Plain controller has no saved grammar')
                return None
            if not isinstance(constraint,dict):
                raise InvalidOperation('Invalid saved grammar constraint')
            saved=constraint.get('agent_state')
            if state is not None and state!=saved:
                raise InvalidOperation('Requested grammar differs from saved controller')
            state=saved
        candidates=[]
        if self.startup is not None and state in self.startup.states:
            candidates.append(self.startup)
        try:candidates.append(self.catalog.resolve(state,controller=controller))
        except InvalidOperation:pass
        if controller is not None:
            candidates=[r for r in candidates if r.matches_controller(controller)]
        if not candidates:
            raise InvalidOperation('Saved or selected immutable grammar is unavailable')
        if len({r.engine_id for r in candidates})!=1:
            raise InvalidOperation('Ambiguous immutable grammar state')
        return candidates[0]

    def descriptor(self,state):
        return self.resolve_registry(state).descriptor(state)

    def matches_controller(self,controller):
        try:return self.resolve_registry(controller=controller) is not None
        except (InvalidOperation,TypeError):return False

    def restore(self,controller,history):
        registry=self.resolve_registry(controller=controller)
        if registry is None:raise InvalidOperation('Plain controller has no grammar')
        return registry.restore(controller,history)

    def start(self,state,stream,history,vocab_size,*,sampling=None):
        return self.resolve_registry(state).start(state,stream,history,vocab_size,sampling=sampling)
