"""Opt-in, Store-authoritative boundary for model-free COW serving requests.

This adapter accepts the complete tokenized prompt prepared by the caller. It
does not render text, hold native leases, or run model work on the HTTP thread.
Coordinator request handles are returned unchanged so a transport can stream
provisional events and distinguish them from the durable publication receipt.
"""

from .cow_prefix import exact_append_suffix, exact_input_append_suffix
from context_time_travel.store import InvalidOperation
from context_time_travel.input_spans import InputHistory, history_for_snapshot


class UnsupportedGreedyRequest(ValueError):
    """The request asks for behavior the pinned native greedy operator lacks."""


class CowAdapter:
    def __init__(self, coordinator, grammar_registry=None):
        self.coordinator = coordinator
        self.grammar_registry = grammar_registry

    def head(self, branch):
        """Return the COW Store's canonical branch head."""
        return self.coordinator.store.head(branch)

    def branches(self, root=None):
        """List canonical heads without admitting inference or acquiring a GPU lease."""
        return self.coordinator.store.branches(root=root)

    def context_budget(self, branch):
        """Observe the exact snapshot and profile behind a context budget."""
        coordinator=self.coordinator
        with coordinator._condition:
            if not coordinator._qualified:
                raise InvalidOperation('Coordinator is not ready')
            models=[worker.get('model') for worker in coordinator._workers.values()]
            if not models or any(model is None for model in models):
                raise InvalidOperation('Qualified model profiles are unavailable')
            profiles={(model.profile.get('model_contract_id'),
                       model.profile.get('execution_profile_id'),
                       model.profile.get('execution',{}).get('capacity')) for model in models}
        if len(profiles)!=1:
            raise InvalidOperation('Worker context profiles differ')
        model_id,profile_id,capacity=profiles.pop()
        if type(capacity) is not int or capacity<=0 or not isinstance(model_id,str) or not isinstance(profile_id,str):
            raise InvalidOperation('Pinned context capacity is unavailable')
        head=coordinator.store.head(branch)
        with coordinator.store.read_lease(head.tip):
            snapshot=coordinator.store.get_snapshot(head.tip)
            if snapshot.get('model_contract_id')!=model_id:
                raise InvalidOperation('Snapshot belongs to a different model')
            count=len(history_for_snapshot(coordinator.store,head.tip))
        if count>capacity:
            raise InvalidOperation('Snapshot exceeds pinned context capacity')
        return dict(schema=1,branch=head.name,revision=head.revision,snapshot=head.tip,
                    model_contract_id=model_id,execution_profile_id=profile_id,
                    capacity_tokens=capacity,used_tokens=count,remaining_tokens=capacity-count)

    def namespaces(self):
        return self.coordinator.store.namespaces()

    def create_namespace(self, title, operation_id):
        return self.coordinator.store.create_namespace(title, operation_id)

    def presentation_catalog(self, namespace_id):
        return self.coordinator.store.presentation_catalog(namespace_id)

    def root_presentations(self, namespace_id, root_snapshot):
        """Read only the verified initial-root annotations for this chat/root."""
        return self.coordinator.store.root_presentations(namespace_id, root_snapshot)

    def operation_status(self, operation_id):
        try:
            return self.coordinator.operation_status(operation_id)
        except Exception as error:
            from context_time_travel.store import NotFound
            if not isinstance(error, NotFound):
                raise
            try:
                return self.coordinator.committed_root_status(operation_id)
            except NotFound:
                pass
            try:
                return self.coordinator.store.fork_operation_status(operation_id)
            except NotFound:
                return self.coordinator.store.evaluation_operation_status(operation_id)

    def operation_diagnostic(self, operation_id):
        """Private live-process diagnostics; never dispatch or infer abort."""
        return self.coordinator.operation_diagnostic(operation_id)

    def status(self):
        """Return the coordinator's live metrics snapshot without model work."""
        return self.coordinator.metrics()

    def cache_inventory(self,**cursor):
        return self.coordinator.cache_inventory(**cursor)

    def cache_transfer_enabled(self):
        return getattr(getattr(self.coordinator,'config',None),'cache_transfer',False)

    def export_checkpoint(self,snapshot,model_contract_id,execution_profile_id):
        return self.coordinator.export_checkpoint(snapshot,model_contract_id,execution_profile_id)

    def capabilities(self):
        return self.coordinator.capabilities()

    def pending_request(self, branch):
        return self.coordinator.pending_request(branch)

    def retry_publication(self, operation_id):
        return self.coordinator.retry_publication(operation_id)

    def history(self, branch):
        """Return the canonical typed input view without discarding images."""
        head = self.head(branch)
        return history_for_snapshot(self.coordinator.store,head.tip)

    def source_state(self, branch, revision, operation_id):
        """Read request-bound canonical history/controller under a Store lease."""
        return self.coordinator.request_source_state(branch, revision, operation_id)

    def namespace(self, branch):
        reader = getattr(self.coordinator.store, 'branch_namespace', None)
        return reader(branch) if callable(reader) else None

    def active_presentation(self, branch, snapshot=None):
        return self.coordinator.store.active_presentation(branch, snapshot)

    def published_presentation(self, branch, snapshot=None):
        return self.coordinator.store.published_presentation(branch, snapshot)

    def snapshot_controller(self, snapshot):
        with self.coordinator.store.read_lease(snapshot):
            return self.coordinator.store.get_snapshot(snapshot)['controller']

    def evaluate(self, snapshot_id, operation_id, specification, fixtures):
        """Evaluate one retained candidate and bind its report to an operation ID."""
        from context_time_travel.evaluation import evaluate
        if not isinstance(specification, dict) or 'fixtures' in specification:
            raise ValueError('Evaluation specification must be an object without fixture bytes')
        complete = dict(specification, fixtures=fixtures)
        return self.coordinator.store.evaluate_once(
            operation_id, snapshot_id, specification, fixtures,
            lambda: evaluate(self.coordinator.store, snapshot_id, complete))

    def evaluation(self, evaluation_id):
        return self.coordinator.store.get_evaluation(evaluation_id)

    def select_evaluations(self, evaluation_ids, mandatory):
        import hashlib
        from context_time_travel.evaluation import choose_validated
        from context_time_travel.store import _json_bytes
        record = choose_validated(self.coordinator.store, evaluation_ids, mandatory=mandatory)
        if record is None:
            raise ValueError('No evaluation satisfies the mandatory checks')
        return {'evaluation_id': hashlib.sha256(_json_bytes(record)).hexdigest(),
                'snapshot_id': record['snapshot_id'], 'record': record}

    def adopt_evaluations(self, operation_id, branch, expected_revision,
                          evaluation_ids, mandatory, *,
                          presentation_source_operation=None):
        adopt = getattr(self.coordinator, 'adopt_evaluated_once', None)
        if not callable(adopt):
            raise InvalidOperation('Coordinator does not support serialized evaluation adoption')
        options = {} if presentation_source_operation is None else {
            'presentation_source_operation': presentation_source_operation}
        result = adopt(operation_id, branch, expected_revision, evaluation_ids, mandatory,
                       **options)
        head = result['branch']
        return dict(result, branch={'name': head.name, 'revision': head.revision,
                                    'tip': head.tip, 'root': head.root})

    @staticmethod
    def _validate_settings(settings):
        if settings is None or settings == {}:
            return
        if not isinstance(settings, dict):
            raise UnsupportedGreedyRequest('COW generation settings must be an object')
        fields = {'temperature', 'min_p', 'top_k', 'top_p', 'repetition_penalty',
                  'typical_p', 'dynatemp_range', 'dynatemp_exponent',
                  'xtc_probability', 'xtc_threshold',
                  'mirostat', 'mirostat_tau', 'mirostat_eta',
                  'adaptive_target', 'adaptive_decay',
                  'dry_multiplier', 'dry_base', 'dry_allowed_length', 'dry_penalty_last_n'}
        unsupported = set(settings) - fields
        if unsupported:
            raise UnsupportedGreedyRequest(
                'Unsupported COW generation settings: ' + ', '.join(sorted(map(str, unsupported))))
        if 'temperature' in settings and type(settings['temperature']) not in (int, float):
            raise UnsupportedGreedyRequest('temperature must be a finite number')
        if settings.get('temperature') == 0 and not (set(settings) & (fields - {'temperature'})):
            return None
        from context_time_travel.min_p import validate_policy
        policy = dict(temperature=0.7, min_p=0.0 if settings.get('mirostat') == 2 else 0.05, top_k=0, top_p=1.0,
                      repetition_penalty=1.0)
        policy.update(settings)
        try:
            return validate_policy(policy)
        except (ValueError, TypeError) as exc:
            raise UnsupportedGreedyRequest('Unsupported COW generation policy') from exc

    @staticmethod
    def _validate_bound(max_tokens):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise InvalidOperation('COW max_tokens must be a positive integer')

    def generate(self, branch, rendered_ids, revision, operation_id, max_tokens,
                 settings=None, cancelled=None, *, presentation=None, grammar_state=None,
                 seed=None, text_stops=None):
        """Submit only the new suffix of a complete, fully rendered token prompt.

        The Store head is checked here for a fast stale-request rejection; the
        coordinator repeats the revision check atomically when admitting work.
        Its request handle separates provisional `events`/`snapshot` from the
        durable publication result.
        """
        if text_stops is not None:
            from context_time_travel.text_stops import normalize_stop_sequences
            text_stops=normalize_stop_sequences(text_stops)
            if text_stops and settings is None:
                # Preserve deterministic greedy choice while routing through
                # the consumed-token controller that owns the matcher state.
                settings=dict(temperature=0.0,min_p=0.0,top_k=0,top_p=1.0,
                              repetition_penalty=1.0)
        sampling=self._validate_settings(settings)
        self._validate_bound(max_tokens)
        if cancelled is not None and not callable(cancelled):
            raise InvalidOperation('Cancellation check must be callable')
        head = self.coordinator.request_source_head(branch, revision, operation_id)
        # Keep the original immutable prefix alive while reading it. This also
        # makes durable retries validate against the intent's original source.
        with self.coordinator.store.read_lease(head.tip):
            canonical = history_for_snapshot(self.coordinator.store,head.tip)
            decoder = self.coordinator.store.get_snapshot(head.tip)['controller']
        if grammar_state is not None and self.grammar_registry is None:
            raise UnsupportedGreedyRequest('Grammar state requires server registry configuration')
        constraint = decoder.get('constraint') if isinstance(decoder,dict) else None
        grammar_resume = bool(isinstance(decoder,dict) and decoder.get('status')=='active'
                              and constraint is not None)
        if grammar_resume:
            if self.grammar_registry is None or not self.grammar_registry.matches_controller(decoder):
                raise UnsupportedGreedyRequest('Saved grammar registry identity is unavailable or changed')
            saved_state=constraint.get('agent_state')
            if grammar_state is not None and grammar_state!=saved_state:
                raise UnsupportedGreedyRequest('Cannot change grammar state during an active turn')
            grammar_state=saved_state
        elif constraint is not None and self.grammar_registry is None:
            raise UnsupportedGreedyRequest('Grammar continuation requires the configured registry')
        elif (grammar_state is not None and isinstance(decoder,dict)
                and decoder.get('status')=='active'):
            raise UnsupportedGreedyRequest('Cannot start or switch grammar during an active plain turn')
        if grammar_state is not None:
            try: self.grammar_registry.descriptor(grammar_state)
            except (InvalidOperation,TypeError) as exc:
                raise UnsupportedGreedyRequest('Unknown server-owned grammar state') from exc
        suffix = (exact_input_append_suffix(canonical,rendered_ids) if isinstance(canonical,InputHistory)
                  else exact_append_suffix(canonical,rendered_ids))
        prepared_grammar_answer=(grammar_state is not None and not suffix
            and isinstance(decoder,dict) and decoder.get('status')=='idle'
            and isinstance(presentation,dict) and presentation.get('kind')=='answer')
        if grammar_state is not None and not suffix and not grammar_resume and not prepared_grammar_answer:
            raise UnsupportedGreedyRequest('A stopped grammar turn needs a new user suffix')
        options = {} if presentation is None else {'presentation': presentation}
        if grammar_state is not None:
            options.update(grammar_registry=self.grammar_registry,grammar_state=grammar_state)
        sampled_resume = (isinstance(decoder, dict) and decoder.get('status') == 'active'
                          and ((decoder.get('schema') in (3, 4, 5, 6) and constraint is None)
                               or (decoder.get('schema') == 1 and sampling is not None)))
        if grammar_resume and not suffix:
            options['resume']=True
        if not suffix and sampled_resume:
            saved = decoder.get('policy')
            if not isinstance(saved, dict):
                raise UnsupportedGreedyRequest('Saved sampled controller has no policy')
            if decoder.get('schema') in (3, 4, 5, 6):
                merged = {key: saved[key] for key in
                          ('temperature', 'min_p', 'top_k', 'top_p', 'repetition_penalty',
                           'typical_p', 'dynatemp_range', 'dynatemp_exponent',
                           'xtc_probability', 'xtc_threshold',
                           'mirostat', 'mirostat_tau', 'mirostat_eta',
                           'adaptive_target', 'adaptive_decay',
                           'dry_multiplier', 'dry_base', 'dry_allowed_length',
                           'dry_penalty_last_n')
                          if key in saved}
            else:
                # Explicitly opting an active legacy greedy turn into sampling
                # starts schema 3 from documented defaults while preserving its
                # existing progress in the coordinator's resume path.
                merged = dict(temperature=0.7, min_p=0.05, top_k=0, top_p=1.0,
                              repetition_penalty=1.0)
            if settings is not None:
                merged.update(settings)
            try:
                from context_time_travel.min_p import validate_policy
                sampling = validate_policy(merged)
            except (KeyError, ValueError, TypeError) as exc:
                raise UnsupportedGreedyRequest('Unsupported sampled continuation policy') from exc
            options['sampling'] = sampling
            options['resume'] = True
        if sampling is not None:
            options.setdefault('sampling', sampling)
            if not suffix and isinstance(decoder, dict) and decoder.get('status') == 'stopped':
                raise UnsupportedGreedyRequest('A stopped answer requires new input before generation')
        if text_stops:
            options['text_stops']=tuple(text_stops)
        if seed is not None:
            options['seed']=seed
        return self.coordinator.submit(branch, suffix, max_tokens, operation_id,
                                       expected_revision=revision, cancelled=cancelled, **options)

    def create(self, branch, rendered_ids, operation_id, *, presentation=None, cancelled=None):
        """Create and durably publish the initial canonical prompt snapshot."""
        ids = exact_append_suffix((), rendered_ids)
        options = {} if presentation is None else {'presentation': presentation}
        if cancelled is not None:options['cancelled']=cancelled
        return self.coordinator.create(branch, ids, operation_id, **options)

    def fork(self, name, snapshot, *, source_branch=None, operation_id=None):
        """Fork through the coordinator's durable-checkpoint gate."""
        if source_branch is None and operation_id is None:
            return self.coordinator.fork(name, snapshot)
        return self.coordinator.fork(name, snapshot, source_branch=source_branch,
                                     operation_id=operation_id)

    def rewind(self, branch, revision, snapshot, operation_id):
        """Move the canonical Store reference using the coordinator's CAS API."""
        if self.coordinator.store.branch_namespace(branch) is not None:
            return self.coordinator.rewind_presented(branch, revision, snapshot, operation_id)
        return self.coordinator.rewind(branch, revision, snapshot, operation_id)

    def reseed(self, branch, seed, revision, operation_id, *, presentation=None):
        """Queue a durable controller-only RNG transition under branch CAS."""
        return self.coordinator.reseed(branch,seed,operation_id,expected_revision=revision,
                                       presentation=presentation,
                                       grammar_registry=self.grammar_registry)
