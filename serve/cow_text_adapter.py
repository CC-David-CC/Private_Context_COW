"""Opt-in text boundary over the exact-token COW adapter.

The caller supplies the same full-history renderer and serving tokenizer used
by its frontend. Generated token IDs remain native/COW history; text decoding
is a view over those IDs and is never re-encoded into canonical history.
"""
import codecs
import hashlib
import json
import queue
import re
import copy

from .cow_adapter import CowAdapter
from .cow_prefix import exact_append_suffix
from context_time_travel.store import Conflict, IntegrityError, InvalidOperation, NotFound
from context_time_travel.input_spans import InputHistory, history_for_snapshot, history_segment, generated_text


class UnsupportedTextRequest(ValueError):
    """Text/chat feature outside the initial greedy text-only boundary."""


def _grammar_option(grammar_state):
    """Keep legacy adapter calls free of a keyword that was not requested."""
    return {} if grammar_state is None else {'grammar_state': grammar_state}


def _text_stop_option(text_stops):
    if text_stops is None:
        return {}
    from context_time_travel.text_stops import normalize_stop_sequences
    normalized = normalize_stop_sequences(text_stops)
    return {} if not normalized else {'text_stops': normalized}


def _stops_from_controller(controller):
    descriptor = controller.get('text_stop') if isinstance(controller, dict) else None
    return tuple(descriptor.get('stop_sequences', ())) if isinstance(descriptor, dict) else ()


def _greedy_stop_policy(settings, text_stops):
    if text_stops and settings is None:
        return dict(temperature=0.0, min_p=0.0, top_k=0, top_p=1.0,
                    repetition_penalty=1.0)
    return settings


class _IncrementalUTF8:
    def __init__(self, tokenizer, pending_hex=''):
        self.tokenizer = tokenizer
        token_bytes = getattr(tokenizer, 'token_bytes', None)
        if not callable(token_bytes):
            raise TypeError('COW text decoding requires tokenizer.token_bytes for linear incremental UTF-8')
        self._token_bytes = token_bytes
        self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        try:
            pending = bytes.fromhex(pending_hex)
        except (TypeError, ValueError) as error:
            raise ValueError('Invalid canonical UTF-8 decoder pending bytes') from error
        self._decoder.setstate((pending, 0))

    def push(self, token):
        if type(token) is not int or token < 0:
            raise ValueError('Generated token ID must be a nonnegative integer')
        return self.decode_bytes(self._token_bytes(token))

    def decode_bytes(self, value, *, final=False):
        if type(value) is not bytes:
            raise TypeError('UTF-8 decoder input must be bytes')
        return self._decoder.decode(value, final=final)

    def finish(self):
        return self._decoder.decode(b'', final=True)


class CowTextRequest:
    """Text event view over one coordinator request; native request stays accessible."""
    def __init__(self, request, tokenizer, pending_hex='', controller_reader=None,
                 text_stops=(), stop_state=None,input_preparation=None):
        self.request = request
        self.snapshot = request.snapshot
        self.durable = request.durable
        self._tokenizer = tokenizer
        self._pending_hex = pending_hex
        self._controller_reader = controller_reader
        self.input_preparation=copy.deepcopy(input_preparation)
        self._decoder = _IncrementalUTF8(tokenizer,pending_hex)
        from context_time_travel.text_stops import TextStopMatcher, normalize_stop_sequences
        self._text_stops = normalize_stop_sequences(text_stops)
        if stop_state is not None:
            self._initial_stop_state = copy.deepcopy(stop_state)
        elif self._text_stops:
            self._initial_stop_state = TextStopMatcher(self._text_stops).export_state()
        else:
            self._initial_stop_state = None
        self._stop_matcher = (TextStopMatcher.restore(stop_state) if stop_state is not None
                              else (TextStopMatcher(self._text_stops) if self._text_stops else None))

    def _decode_safe(self, value, *, final=False):
        return self._decoder.decode_bytes(value, final=final)

    def decode_event(self,event):
        if event.get('event') != 'token': return [event]
        token=event['token']
        if event.get('turn_complete'):
            tail=''
            if self._stop_matcher is not None and not self._stop_matcher.finished and not self._stop_matcher.stopped:
                result=self._stop_matcher.feed(b'',final=True)
                tail=self._decode_safe(result.safe_bytes)
            tail+=self._decoder.finish()
            events=[dict(event='text',text=tail)] if tail else []
            return events+[dict(event='turn_complete',token_id=token)]
        if self._stop_matcher is None:
            return [dict(event='text',token_id=token,text=self._decoder.push(token))]
        piece=self._tokenizer.token_bytes(token)
        result=self._stop_matcher.feed(piece)
        text=self._decode_safe(result.safe_bytes)
        if result.stopped:
            text += self._decoder.finish()
            events=[dict(event='text',token_id=token,text=text)] if text else []
            events.append(dict(event='turn_complete',token_id=token,reason='text_stop'))
            return events
        return [dict(event='text',token_id=token,text=text)] if text else []

    def text_result(self):
        """Reconstruct the presentation from committed IDs, including retries."""
        receipt=self.durable.result()
        if not callable(self._controller_reader):
            raise InvalidOperation('Text result requires the canonical controller reader')
        controller=self._controller_reader(receipt['snapshot'])
        eos=controller.get('policy',{}).get('eos_tokens',[])
        decoder=_IncrementalUTF8(self._tokenizer,self._pending_hex);parts=[]
        stops=self._text_stops
        if not stops and isinstance(controller.get('text_stop'),dict):
            stops=tuple(controller['text_stop'].get('stop_sequences',()))
        from context_time_travel.text_stops import TextStopMatcher
        initial_stop_state=self._initial_stop_state
        matcher=(TextStopMatcher.restore(initial_stop_state) if initial_stop_state is not None
                 else (TextStopMatcher(stops) if stops else None))
        terminal=controller.get('status')!='active'
        for token in self.snapshot.result()['tokens']:
            if token in eos:
                if matcher is not None and not matcher.finished and not matcher.stopped:
                    final=matcher.feed(b'',final=True)
                    parts.append(decoder.decode_bytes(final.safe_bytes))
                if terminal: parts.append(decoder.finish())
                break
            if matcher is None:
                parts.append(decoder.push(token));continue
            result=matcher.feed(self._tokenizer.token_bytes(token))
            parts.append(decoder.decode_bytes(result.safe_bytes))
            if result.stopped:
                if terminal: parts.append(decoder.finish())
                break
        else:
            if terminal:
                if matcher is not None and not matcher.finished and not matcher.stopped:
                    final=matcher.feed(b'',final=True)
                    parts.append(decoder.decode_bytes(final.safe_bytes))
                parts.append(decoder.finish())
        return ''.join(parts)

    def iter_events(self, timeout=0.1):
        while not self.request.durable.done() or not self.request.events.empty():
            try:
                event = self.request.events.get(timeout=timeout)
            except queue.Empty:
                continue
            yield from self.decode_event(event)


class CowTextAdapter:
    """Render whole text conversations, then delegate exact-prefix admission."""
    def __init__(self, token_adapter: CowAdapter, tokenizer, render, encode_rendered,
                 *, append_user_suffix=None, append_initial_user_suffix=None,
                 initial_root_mode=None, renderer_id=None, grammar_registry=None,
                 vision_preparer=None,mixed_renderer=None):
        if not callable(render):
            raise TypeError('render must render the complete conversation')
        if not callable(encode_rendered):
            raise TypeError('encode_rendered must preserve frontend literal-token handling')
        if not callable(getattr(tokenizer, 'token_bytes', None)):
            raise TypeError('COW text decoding requires tokenizer.token_bytes for linear incremental UTF-8')
        self.token_adapter = token_adapter
        self.tokenizer = tokenizer
        self.render = render
        self.encode_rendered = encode_rendered
        self.append_user_suffix = append_user_suffix
        self.append_initial_user_suffix = append_initial_user_suffix
        self.initial_root_mode = initial_root_mode
        if renderer_id is not None and (not isinstance(renderer_id,str)
                or re.fullmatch('[0-9a-f]{64}',renderer_id) is None):
            raise UnsupportedTextRequest('Renderer identity must be a SHA-256 digest')
        self.renderer_id = renderer_id
        self.vision_preparer=vision_preparer
        self.mixed_renderer=(mixed_renderer if mixed_renderer is not None else
                             vision_preparer.renderer if vision_preparer is not None else None)
        if self.mixed_renderer is not None and self.mixed_renderer.base_renderer_id!=renderer_id:
            raise UnsupportedTextRequest('Mixed renderer differs from the configured text renderer')
        if vision_preparer is not None and vision_preparer.renderer.renderer_id!=self.mixed_renderer.renderer_id:
            raise UnsupportedTextRequest('Image preparer and mixed renderer identities differ')
        self.grammar_registry = grammar_registry
        if grammar_registry is not None:
            self.token_adapter.grammar_registry = grammar_registry

    def _history_renderer(self,canonical=None):
        if isinstance(canonical,InputHistory) or (canonical is None and self.mixed_renderer is not None):
            if self.mixed_renderer is None:
                raise UnsupportedTextRequest('Mixed history requires its pinned renderer')
            return self.mixed_renderer.renderer_id
        return self.renderer_id

    @staticmethod
    def _history_view(canonical):
        return canonical if isinstance(canonical,InputHistory) else tuple(canonical)

    @staticmethod
    def _text_tail(canonical):
        if not isinstance(canonical,InputHistory):return canonical
        spans=canonical.spans
        return tuple(spans[-1]['tokens']) if spans and spans[-1]['kind']=='text' else ()

    def _mixed_controls(self,canonical,settings=None,grammar_state=None,seed=None,text_stops=None):
        if not isinstance(canonical,InputHistory):return
        greedy=(settings is None or settings=={} or (type(settings) is dict and set(settings)=={'temperature'}
                and type(settings['temperature']) in (int,float) and settings['temperature']==0))
        if text_stops:
            raise UnsupportedTextRequest('Mixed-input text stops are not yet qualified')
        if not greedy or seed is not None:
            from context_time_travel.vision_controllers import supports_basic_sampling
            try:sampling=self.token_adapter._validate_settings(settings)
            except (ValueError,InvalidOperation) as error:
                raise UnsupportedTextRequest('Unsupported mixed-input sampling policy') from error
            models=[worker.get('model') for worker in getattr(getattr(self.token_adapter,'coordinator',None),'_workers',{}).values()]
            if not models or any(model is None or not supports_basic_sampling(
                    model.profile.get('execution',{}),model.backend,sampling) for model in models):
                raise UnsupportedTextRequest('Mixed-input sampler/controller is unavailable; use qualified greedy controls')
        if grammar_state is not None:
            if self.grammar_registry is None:
                raise UnsupportedTextRequest('Image grammar selection requires the configured server registry')
            try:self.grammar_registry.descriptor(grammar_state)
            except (InvalidOperation,TypeError) as error:
                raise UnsupportedTextRequest('Unknown server-owned grammar state') from error

    def _prepare_mixed(self,branch,messages,operation,namespace,add_generation_prompt,cancelled=None):
        if self.vision_preparer is None:
            raise UnsupportedTextRequest('Image preparation is not configured')
        request=dict(schema=1,namespace_id=namespace,messages=messages,add_generation_prompt=add_generation_prompt)
        presentation=self.vision_preparer.presentation(request,operation)
        return self.token_adapter.coordinator.prepare_root(branch,request,operation,presentation=presentation,cancelled=cancelled)

    def _mixed_user_request(self,branch,content,revision,operation,max_tokens,kind,canonical,controller,cancelled,
                            grammar_state=None,settings=None,seed=None):
        """Resolve only the new user input; a committed retry needs no producer."""
        from .cow_vision_preparer import CowVisionPreparer
        coordinator=self.token_adapter.coordinator
        if not isinstance(canonical,InputHistory) or self.mixed_renderer is None:
            raise UnsupportedTextRequest('New image questions require a configured mixed context')
        self.token_adapter._validate_bound(max_tokens)
        if cancelled is not None and not callable(cancelled):
            raise UnsupportedTextRequest('Cancellation check must be callable')
        try:messages=self.mixed_renderer.validate([dict(role='user',content=content)])
        except InvalidOperation as error:raise UnsupportedTextRequest(str(error)) from error
        parts=messages[0]['content']
        if not isinstance(parts,list) or not any(part['type']=='image' for part in parts):
            raise UnsupportedTextRequest('Mixed question parts must include an image')
        namespace=self._namespace(branch)
        display=''.join(part['text'] if part['type']=='text' else '[image sha256:'+part['image_id']+']' for part in parts)
        presentation=self._presentation(namespace,kind,[dict(role='user',content=display)],operation,
            self._message_id(operation,'assistant'),canonical=canonical,seed=seed)
        grammar_options=({} if grammar_state is None else
            dict(grammar_registry=self.grammar_registry,grammar_state=grammar_state))
        grammar_options.update(sampling=self.token_adapter._validate_settings(settings),seed=seed)
        try:committed=coordinator.committed_generation(operation)
        except NotFound:committed=None
        if committed is not None:
            self._validate_retry_compatibility(committed)
            generation=committed['generation']
            if generation is None or generation.get('schema') not in (9,11,13) or generation.get('kind')!=kind:
                raise UnsupportedTextRequest('Operation is bound to a different generation input kind')
            recorded=InputHistory(generation['ids'])
            images=[span for span in recorded.spans if span['kind']=='image']
            delta=CowVisionPreparer.render_suffix(self.mixed_renderer,canonical,controller,messages,images,kind)
            if delta!=recorded:raise InvalidOperation('Operation ID reused for different image inputs')
            preparation=dict(source='committed-replay',consumed_cells=len(delta),forward_tokens=0,encodings=[],
                image_rows=sum(span['rows'] for span in images),images=[dict(original_sha256=span['original_sha256'],
                embedding_sha256=span['embedding_sha256'],rows=span['rows']) for span in images])
            request=coordinator.submit(branch,delta,max_tokens,operation,expected_revision=revision,
                presentation=presentation,cancelled=cancelled,**grammar_options)
        else:
            try:pending=coordinator.operation_status(operation)
            except NotFound:pending=None
            if pending is not None:
                raise Conflict('Operation already admitted; recover its pending request or publication')
            if self.vision_preparer is None:
                raise UnsupportedTextRequest('Image preparation is not configured')
            body=dict(schema=2,namespace_id=namespace,kind=kind,messages=messages,add_generation_prompt=True)
            with coordinator.prepare_suffix(branch,revision,body,operation,max_tokens,
                    presentation=presentation,cancelled=cancelled) as prepared:
                preparation=copy.deepcopy(prepared.receipt)
                request=coordinator.submit(branch,prepared.history,max_tokens,operation,
                    expected_revision=revision,presentation=presentation,cancelled=cancelled,artifacts=prepared.artifacts,
                    **grammar_options)
        return CowTextRequest(request,self.tokenizer,
            controller_reader=getattr(self.token_adapter,'snapshot_controller',None),input_preparation=preparation)

    def reserve_image_upload(self,namespace_id,size):
        """Admit bounded incoming bytes before the transport reads their body."""
        if self.vision_preparer is None:
            raise UnsupportedTextRequest('Image preparation is not configured')
        if type(namespace_id) is not str or re.fullmatch('[0-9a-f]{32}',namespace_id) is None:
            raise UnsupportedTextRequest('Invalid image namespace')
        if type(size) is not int or not 0<size<=self.vision_preparer.normalizer.max_bytes:
            raise UnsupportedTextRequest('Image byte length exceeds the configured bound')
        self.vision_preparer.store.get_namespace(namespace_id)
        return self.token_adapter.coordinator.pool.reserve(size)

    def upload_image(self,namespace_id,original,expected_sha256):
        """Verify client byte identity before granting a normalized upload."""
        if self.vision_preparer is None:
            raise UnsupportedTextRequest('Image preparation is not configured')
        if (type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None
                or type(original) is not bytes or hashlib.sha256(original).hexdigest()!=expected_sha256):
            raise UnsupportedTextRequest('Image content digest does not match the uploaded bytes')
        coordinator=self.token_adapter.coordinator
        return self.vision_preparer.upload(namespace_id,original,pool=coordinator.pool,disk=coordinator.disk)

    def classify_root(self, namespace_id, root_id, canonical, controller, presentations):
        """Derive initial-root capability from verified annotation and token tail."""
        try:renderer_id=self._history_renderer(canonical)
        except UnsupportedTextRequest:return 'unsupported'
        if (renderer_id is None or not callable(self.initial_root_mode)
                or not isinstance(namespace_id, str) or not isinstance(root_id, str)
                or not isinstance(presentations, (list, tuple)) or not presentations):
            return 'unsupported'
        for presentation in presentations:
            if (not isinstance(presentation, dict)
                    or presentation.get('schema') != (3 if isinstance(canonical,InputHistory) else 1)
                    or presentation.get('namespace_id') != namespace_id
                    or presentation.get('kind') != 'root'
                    or presentation.get('renderer_id') != renderer_id):
                return 'unsupported'
        mode = self.initial_root_mode(self._text_tail(canonical), controller)
        return mode if mode in ('context_only', 'prepared_answer') else 'unsupported'

    def root_mode_for_projection(self, namespace_id, root_id, canonical, controller,
                                 presentations):
        """Workspace callback; inputs come from its verified Store snapshot."""
        return self.classify_root(namespace_id, root_id, canonical, controller, presentations)

    def _root_mode_for_source(self, head, canonical, controller):
        if head is None or head.tip != head.root or controller.get('status') != 'idle':
            return 'unsupported'
        namespace = self._namespace(head.name)
        if namespace is None:
            return 'unsupported'
        root_reader = getattr(self.token_adapter, 'root_presentations', None)
        if callable(root_reader):
            presentations = root_reader(namespace, head.root)
        else:
            # Compatibility for external token adapters predating the narrow
            # Store query. A Store-backed CowAdapter always takes the bounded path.
            reader = getattr(self.token_adapter, 'presentation_catalog', None)
            if not callable(reader):
                return 'unsupported'
            catalog = reader(namespace)
            presentations = [event['presentation'] for event in catalog['events']
                if event.get('result', {}).get('tip') == head.root
                and isinstance(event.get('presentation'), dict)
                and event['presentation'].get('kind') == 'root']
        return self.classify_root(namespace, head.root, canonical, controller, presentations)

    def _namespace(self, branch):
        reader = getattr(self.token_adapter, 'namespace', None)
        return reader(branch) if callable(reader) else None

    def retry_info(self, operation_id, settings=None, seed=None, text_stops=None):
        """Resolve a durable answer's branch without trusting a client journal."""
        record=self.token_adapter.coordinator.committed_generation(operation_id)
        result=record['receipt']['result']
        generation=record['generation']
        if seed is not None:
            from context_time_travel.seed_controls import seed_stream
            seed_stream(seed)
        if generation is None:
            replay=settings in (None,{}) and seed is None and text_stops in (None,[])
            return dict(operation_id=operation_id,branch=result['name'],
                snapshot=result['tip'],revision=result['revision'],sampling=None,
                resume=record['presentation'].get('kind')=='resume',
                generation_kind=record['presentation'].get('kind'),legacy=True,replay=replay)
        effective=self._effective_retry_sampling(generation['sampling'],settings)
        controls=generation.get('controls',{})
        old_seed=controls.get('seed') if isinstance(controls,dict) else None
        same_seed=(seed is None or seed==old_seed)
        old_stops=tuple(generation.get('controls',{}).get('text_stops',()))
        if text_stops is None:
            requested_stops=old_stops
        else:
            from context_time_travel.text_stops import normalize_stop_sequences
            requested_stops=normalize_stop_sequences(text_stops)
        return dict(operation_id=operation_id,branch=result['name'],
            snapshot=result['tip'],revision=result['revision'],
            sampling=generation['sampling'],resume=generation['resume'],
            generation_kind=generation['kind'],replay=(effective==generation['sampling']
                and same_seed and requested_stops==old_stops),text_stops=list(old_stops))

    def _effective_retry_sampling(self, old_sampling, settings):
        if settings is None or settings=={}:
            return old_sampling
        if old_sampling is None:
            return self.token_adapter._validate_settings(settings)
        from context_time_travel.min_p import validate_policy
        if not isinstance(settings,dict):
            raise UnsupportedTextRequest('Retry settings must be an object')
        merged=dict(old_sampling);merged.update(settings)
        try: return validate_policy(merged)
        except (ValueError,TypeError,KeyError) as error:
            raise UnsupportedTextRequest('Unsupported retry sampler policy') from error

    def retry(self, operation_id, branch, revision, new_operation_id, *,
              settings=None, cancelled=None, seed=None, text_stops=None):
        """Replay a committed answer or rerun its original inputs with new policy."""
        source=self.token_adapter.coordinator.committed_generation(operation_id)
        self._validate_retry_compatibility(source)
        generation=source['generation']
        if generation is None:
            if settings not in (None,{}) or seed is not None or text_stops not in (None,[]):
                raise UnsupportedTextRequest(
                    'Legacy committed answer can be replayed but has no safe changed-controls retry descriptor')
            return self._replay_committed(source)
        old_sampling=generation['sampling']
        effective=self._effective_retry_sampling(old_sampling,settings)
        controls=generation.get('controls',{})
        old_seed=controls.get('seed') if isinstance(controls,dict) else None
        if seed is not None:
            from context_time_travel.seed_controls import seed_stream
            seed_stream(seed)
        same_seed=(seed is None or seed==old_seed)
        old_stops=tuple(generation.get('controls',{}).get('text_stops',()))
        if text_stops is None:
            effective_stops=old_stops
        else:
            from context_time_travel.text_stops import normalize_stop_sequences
            effective_stops=normalize_stop_sequences(text_stops)
        if effective is None and effective_stops:
            effective=dict(temperature=0.0,min_p=0.0,top_k=0,top_p=1.0,
                           repetition_penalty=1.0)
        if effective==old_sampling and same_seed and effective_stops==old_stops:
            return self._replay_committed(source)
        if generation.get('source_input_kind')=='canonical-mixed-spans-v1':
            canonical=history_for_snapshot(self.token_adapter.coordinator.store,source['source_head'].tip)
            descriptor=generation.get('grammar')
            self._mixed_controls(canonical,effective,descriptor.get('agent_state') if descriptor else None,
                seed if seed is not None else old_seed,effective_stops)
        if new_operation_id==operation_id:
            raise UnsupportedTextRequest('Changed retry requires a new operation ID')
        if effective is None and not effective_stops and not (seed is not None and not same_seed):
            return self._replay_committed(source)
        retry_seed=seed if seed is not None else old_seed
        generation=source['generation'];presentation=source['presentation']
        if presentation.get('kind') in ('answer','initial_turn','user_turn','resume'):
            presentation=dict(presentation)
            if presentation['kind']=='resume':
                presentation['generation_settings']=effective
            if retry_seed is not None:
                presentation['seed']=retry_seed
                presentation['schema']=2
            if presentation['kind']!='resume':
                presentation['output_message_id']=self._message_id(new_operation_id,'assistant')
        else:
            raise UnsupportedTextRequest('This presentation kind cannot be retried')
        request=self.token_adapter.coordinator.submit_retry_generation(
            operation_id,branch,revision,new_operation_id,effective,presentation,
            cancelled=cancelled,grammar_registry=self.grammar_registry,
            seed=retry_seed,text_stops=effective_stops)
        source_controller=self.token_adapter.snapshot_controller(source['source_head'].tip)
        pending=source_controller.get('output_pending_hex','') if generation['resume'] else ''
        return CowTextRequest(request,self.tokenizer,pending,
            controller_reader=getattr(self.token_adapter,'snapshot_controller',None),
            text_stops=effective_stops)

    def _validate_retry_compatibility(self, source):
        """Refuse to decode committed token IDs with a different serving identity."""
        binding=source.get('presentation')
        mixed=(source.get('receipt',{}).get('presentation',{}).get('schema')==3)
        renderer_id=(self._history_renderer(InputHistory([])) if mixed else self.renderer_id)
        if (not isinstance(binding,dict) or not isinstance(self.renderer_id,str)
                or binding.get('renderer_id')!=renderer_id):
            raise UnsupportedTextRequest('Current renderer differs from the committed answer')
        saved=None
        try:
            saved=self.token_adapter.coordinator.store.checkpoint_receipt(
                source['operation_id'])['checkpoint_receipt']
        except NotFound:
            # Older receipts may predate checkpoint metadata. Their token
            # history is still canonical, so also verify every token fits the
            # current native vocabulary before serving a text replay.
            pass
        models=[worker.get('model') for worker in
                self.token_adapter.coordinator._workers.values()]
        models=[model for model in models if model is not None]
        if saved is not None and models:
            current=models[0].profile
            if (saved.get('model_contract_id')!=current.get('model_contract_id')
                    or saved.get('execution_profile_id')!=current.get('execution_profile_id')):
                raise UnsupportedTextRequest('Current model profile differs from the committed answer')
        if models:
            vocab=models[0].backend.vocab_size
            result=source['receipt']['result']
            with self.token_adapter.coordinator.store.read_lease(result['tip']):
                target=history_for_snapshot(self.token_adapter.coordinator.store,result['tip'])
            # Validate text IDs and each image's pad ID once, without treating
            # image rows as generated/repetition text or another history.
            tokens=(target.text_tokens+tuple(span['pad_token_id'] for span in target.spans if span['kind']=='image')
                    if isinstance(target,InputHistory) else target)
            if any(type(token) is not int or not 0<=token<vocab for token in tokens):
                raise UnsupportedTextRequest('Committed token IDs do not fit the current tokenizer vocabulary')

    def thinking_result(self, snapshot):
        """Derived, receipt-bound view; no text-tag splitting or native work.

        Only a fully consumed terminal scope supplies executable final text.
        Replaying the same canonical history also reconstructs this view after
        a lost completion. Nothing here becomes a second stored transcript.
        """
        store = self.token_adapter.coordinator.store
        with store.read_lease(snapshot):
            controller = store.get_snapshot(snapshot)['controller']
            if not isinstance(controller.get('constraint'), dict):
                return None
            if self.grammar_registry is None or not self.grammar_registry.matches_controller(controller):
                raise InvalidOperation('Saved grammar identity is unavailable for the thinking result')
            history = history_for_snapshot(store, snapshot)
        tokens = history.text_tokens if isinstance(history, InputHistory) else history
        session = self.grammar_registry.restore(controller, list(tokens))
        progress = session.thinking_progress()
        if progress is None:
            return None
        scope = tokens[controller['constraint']['scope_start']:]
        close, = self.tokenizer.encode('</think>', parse_special=True)
        closing = scope.index(close) if close in scope else None
        eos = controller['policy']['eos_tokens']
        # A text stop may terminate reasoning or an incomplete final without
        # consuming EOS. It is a valid stopped scope, not an executable answer.
        complete = (controller.get('status') == 'stopped' and closing is not None
                    and bool(scope) and scope[-1] in eos and session.is_complete())
        reasoning = scope if closing is None else scope[:closing]
        final = None
        if complete:
            final = self.tokenizer.decode([token for token in scope[closing + 1:] if token not in eos])
        return dict(schema=1, snapshot=snapshot, progress=progress, complete=complete,
                    reasoning_text=self.tokenizer.decode(list(reasoning)), final_text=final)

    def operation_status(self, operation):
        # Durable publication proves commitment even after checkpoint expiry.
        # Absence is never proof of abort.
        source = self.token_adapter.coordinator.committed_generation(operation)
        receipt = self._committed_durable_receipt(source)
        return dict(operation_id=operation, branch=receipt['branch']['name'],
                    state='durable', scope='canonical-presentation',
                    durable=True, receipt=receipt)

    def _committed_durable_receipt(self, source):
        operation=source['operation_id'];receipt=source['receipt']
        target=receipt['result']['tip']
        result=receipt['result']
        durable=dict(branch={key:result[key] for key in ('name','revision','tip','root')},
                     snapshot=target)
        try:
            checkpoint_record=self.token_adapter.coordinator.store.checkpoint_receipt(operation)
            saved=checkpoint_record['checkpoint_receipt']
        except NotFound:
            saved=None
        if isinstance(saved,dict):
            durable.update(payload_sha256=saved['payload_sha256'],
                model_contract_id=saved['model_contract_id'],
                execution_profile_id=saved['execution_profile_id'])
        return durable

    def _replay_committed(self, source):
        operation=source['operation_id'];receipt=source['receipt'];generation=source['generation']
        target=receipt['result']['tip'];source_head=source['source_head']
        with self.token_adapter.coordinator.store.read_lease(target):
            target_record=self.token_adapter.coordinator.store.get_snapshot(target)
            history=history_for_snapshot(self.token_adapter.coordinator.store,target)
        with self.token_adapter.coordinator.store.read_lease(source_head.tip):
            prior=history_for_snapshot(self.token_adapter.coordinator.store,source_head.tip)
            source_controller=self.token_adapter.coordinator.store.get_snapshot(source_head.tip)['controller']
        binding=receipt['presentation']
        controls=(generation or {}).get('controls',{})
        stops=tuple(controls.get('text_stops',()))
        # committed_generation already validates the descriptor against this
        # canonical span. Its count is model cells, not span dictionary count.
        input_count=binding['input_span']['count']
        output_span=binding.get('output_span')
        if not isinstance(output_span,dict):
            raise InvalidOperation('Committed answer has no canonical output span')
        start=output_span['start']
        tokens=generated_text(history_segment(history,start))
        if (len(tokens)!=output_span['count'] or start!=len(prior)+input_count
                or history_segment(history,0,len(prior))!=prior):
            raise InvalidOperation('Committed answer token span differs from its intent')
        decoder=_IncrementalUTF8(self.tokenizer,
            source_controller.get('output_pending_hex','')
            if binding.get('kind')=='resume' else '')
        eos=target_record['controller'].get('policy',{}).get('eos_tokens',[])
        from context_time_travel.text_stops import TextStopMatcher
        source_stop=source_controller.get('text_stop')
        if (binding.get('kind')=='resume' and stops and isinstance(source_stop,dict)
                and tuple(source_stop.get('stop_sequences',()))==stops):
            matcher=TextStopMatcher.restore(source_stop)
        else:
            matcher=TextStopMatcher(stops) if stops else None
        parts=[]
        for token in tokens:
            if token in eos:
                if matcher is not None:
                    final=matcher.feed(b'',final=True)
                    parts.append(decoder.decode_bytes(final.safe_bytes))
                parts.append(decoder.finish());break
            if matcher is None:
                parts.append(decoder.push(token));continue
            result=matcher.feed(self.tokenizer.token_bytes(token))
            parts.append(decoder.decode_bytes(result.safe_bytes))
            if result.stopped:
                parts.append(decoder.finish());break
        else:
            if target_record['controller'].get('status')!='active' and matcher is not None:
                final=matcher.feed(b'',final=True)
                parts.append(decoder.decode_bytes(final.safe_bytes))
            if target_record['controller'].get('status')!='active':
                parts.append(decoder.finish())
        durable=self._committed_durable_receipt(source)
        answer = dict(snapshot=target,tokens=list(tokens),text=''.join(parts))
        controller=target_record['controller']
        if isinstance(controller.get('constraint'),dict):
            if self.grammar_registry is None or not self.grammar_registry.matches_controller(controller):
                raise InvalidOperation('Saved grammar identity is unavailable for replay completion')
            canonical_tokens=history.text_tokens if isinstance(history,InputHistory) else history
            session=self.grammar_registry.restore(controller,list(canonical_tokens))
            # Acceptance and stopping differ: an accepting prefix can still be
            # active, and a text stop can terminate an incomplete grammar.
            answer.update(grammar_complete=session.is_complete(),stopped=session.is_stopped())
        thinking = self.thinking_result(target)
        if thinking is not None:
            answer['thinking'] = thinking
        return dict(event='complete',replayed=True,answer=answer,receipt=durable)

    @staticmethod
    def _message_id(operation_id, label):
        raw=json.dumps([operation_id,label],ensure_ascii=True,separators=(',',':')).encode('ascii')
        return hashlib.sha256(raw).hexdigest()[:32]

    def _presentation(self, namespace, kind, messages, operation, output_id=None,
                      generation_settings=None, seed=None,*,canonical=None):
        renderer_id=self._history_renderer(canonical)
        if renderer_id is None:
            raise UnsupportedTextRequest('A pinned renderer identity is required for saved chat presentation')
        if not isinstance(namespace,str) or re.fullmatch('[0-9a-f]{32}',namespace) is None:
            raise UnsupportedTextRequest('Invalid presentation namespace')
        clean=self._messages(messages) if messages else []
        result=dict(schema=(2 if seed is not None else 1),namespace_id=namespace,kind=kind,
            messages=[dict(id=self._message_id(operation,f'input-{index}'),**message)
                      for index,message in enumerate(clean)],
            renderer_id=renderer_id,output_message_id=output_id)
        if generation_settings is not None:
            result['generation_settings']=generation_settings
        if seed is not None:
            result['seed']=seed
        return result

    @staticmethod
    def _messages(messages):
        if not isinstance(messages, (list, tuple)) or not messages:
            raise UnsupportedTextRequest('messages must be a nonempty list of text messages')
        clean = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != {'role', 'content'}:
                raise UnsupportedTextRequest(f'message {index} must contain only role and text content')
            if message['role'] not in ('system', 'user', 'assistant'):
                raise UnsupportedTextRequest(f'message {index} has an unsupported role')
            if not isinstance(message['content'], str):
                raise UnsupportedTextRequest(f'message {index} content must be plain text')
            clean.append({'role': message['role'], 'content': message['content']})
        return clean

    def create(self, branch, messages, operation_id, *, namespace_id=None, cancelled=None):
        if isinstance(messages,(list,tuple)) and any(isinstance(message,dict) and isinstance(message.get('content'),list) for message in messages):
            return self._prepare_mixed(branch,messages,operation_id,namespace_id,False,cancelled)
        clean = self._messages(messages)
        options = {} if namespace_id is None else {'presentation':
            self._presentation(namespace_id,'root',clean,operation_id)}
        if cancelled is not None:options['cancelled']=cancelled
        ids = self._render_encode(clean, add_generation_prompt=False)
        return self.token_adapter.create(branch, ids, operation_id, **options)

    def prepare(self, branch, messages, operation_id, *, namespace_id, cancelled=None):
        """Durably prepare a first prompt, including its assistant generation frame."""
        if isinstance(messages,(list,tuple)) and any(isinstance(message,dict) and isinstance(message.get('content'),list) for message in messages):
            return self._prepare_mixed(branch,messages,operation_id,namespace_id,True,cancelled)
        clean=self._messages(messages)
        presentation=self._presentation(namespace_id,'root',clean,operation_id)
        ids=self._render_encode(clean,add_generation_prompt=True)
        return self.token_adapter.create(branch,ids,operation_id,presentation=presentation,
            **({} if cancelled is None else {'cancelled':cancelled}))

    def answer(self, branch, revision, operation_id, max_tokens, *, settings=None, cancelled=None,
               grammar_state=None, seed=None, text_stops=None):
        """Answer the prepared first prompt without rendering its history again."""
        head,canonical,controller=self.token_adapter.source_state(branch,revision,operation_id)
        self._mixed_controls(canonical,settings,grammar_state,seed,text_stops)
        if not isinstance(controller,dict) or controller.get('status')!='idle':
            raise UnsupportedTextRequest('Initial answer requires an idle prepared prompt')
        if self._root_mode_for_source(head, canonical, controller) != 'prepared_answer':
            raise UnsupportedTextRequest('Initial answer requires a renderer-matched prepared root')
        namespace=self._namespace(branch)
        if namespace is None:
            raise UnsupportedTextRequest('Initial chat answer requires a presentation namespace')
        presentation=self._presentation(namespace,'answer',[],operation_id,
                                        self._message_id(operation_id,'assistant'),seed=seed,canonical=canonical)
        stop_option=_text_stop_option(text_stops)
        settings=_greedy_stop_policy(settings,stop_option.get('text_stops'))
        request=self.token_adapter.generate(branch,self._history_view(canonical),revision,operation_id,max_tokens,
            settings=settings,cancelled=cancelled,presentation=presentation,seed=seed,
            **_grammar_option(grammar_state),**stop_option)
        return CowTextRequest(request,self.tokenizer,
            controller_reader=getattr(self.token_adapter,'snapshot_controller',None),
            text_stops=stop_option.get('text_stops',()))

    def generate(self, branch, messages, revision, operation_id, max_tokens, *,
                 settings=None, cancelled=None, grammar_state=None, seed=None,
                 text_stops=None, **unsupported):
        if unsupported:
            raise UnsupportedTextRequest('Unsupported COW text options: ' + ', '.join(sorted(unsupported)))
        if self._namespace(branch) is not None:
            raise UnsupportedTextRequest('Saved chats use answer, append_user or resume, preserving canonical tokens')
        ids = self._render_encode(messages, add_generation_prompt=True)
        _, canonical, controller = self.token_adapter.source_state(branch, revision, operation_id)
        if isinstance(canonical,InputHistory):
            raise UnsupportedTextRequest('Mixed history requires saved-chat continuation endpoints')
        suffix = exact_append_suffix(canonical, ids)
        pending_hex = controller.get('output_pending_hex', '') if not suffix else ''
        stop_option=_text_stop_option(text_stops)
        settings=_greedy_stop_policy(settings,stop_option.get('text_stops'))
        request = self.token_adapter.generate(branch, ids, revision, operation_id, max_tokens,
            settings=settings, cancelled=cancelled, seed=seed,
            **_grammar_option(grammar_state),**stop_option)
        return CowTextRequest(request, self.tokenizer, pending_hex,
            controller_reader=getattr(self.token_adapter,'snapshot_controller',None),
            text_stops=stop_option.get('text_stops',()))

    def resume(self, branch, revision, operation_id, max_tokens, *, settings=None, cancelled=None,
               grammar_state=None, seed=None, text_stops=None):
        """Continue the saved supported policy without rendering or re-encoding text.

        The Store snapshot's token history and controller are authoritative.
        In particular, a pending UTF-8 prefix is carried into the returned text
        view, while generated text is never converted back into token IDs.
        """
        head, canonical, controller = self.token_adapter.source_state(
            branch, revision, operation_id)
        self._mixed_controls(canonical,settings,grammar_state,seed,text_stops)
        if not isinstance(controller, dict) or controller.get('status') != 'active':
            raise UnsupportedTextRequest('COW text resume requires an active controller')
        if seed is not None:
            raise UnsupportedTextRequest('Use the explicit reseed operation to change an active generation stream')
        saved_stops=_stops_from_controller(controller)
        if text_stops is not None:
            from context_time_travel.text_stops import normalize_stop_sequences
            if normalize_stop_sequences(text_stops)!=saved_stops:
                raise UnsupportedTextRequest('Text stops cannot change while resuming an active turn')
        text_stops=saved_stops or None
        constraint=controller.get('constraint')
        grammar_active=isinstance(constraint,dict)
        if grammar_active:
            if self.grammar_registry is None or not self.grammar_registry.matches_controller(controller):
                raise UnsupportedTextRequest('Saved grammar registry identity is unavailable or changed')
            saved_state=constraint.get('agent_state')
            if grammar_state is not None and grammar_state!=saved_state:
                raise UnsupportedTextRequest('Cannot change grammar state during an active turn')
            grammar_state=saved_state
        elif grammar_state is not None:
            raise UnsupportedTextRequest('Grammar state can only resume an active grammar controller')
        policy = controller.get('policy')
        sampled=(controller.get('schema') in (3,4,5,6) and isinstance(policy,dict))
        if sampled and isinstance(policy,dict):
            from context_time_travel.min_p import validate_policy
            try:
                effective={key:policy[key] for key in
                    ('temperature','min_p','top_k','top_p','repetition_penalty',
                     'typical_p','dynatemp_range','dynatemp_exponent',
                     'xtc_probability','xtc_threshold',
                     'mirostat','mirostat_tau','mirostat_eta',
                     'adaptive_target','adaptive_decay',
                     'dry_multiplier','dry_base','dry_allowed_length','dry_penalty_last_n')
                    if key in policy}
                if settings is not None:
                    if not isinstance(settings,dict):
                        raise ValueError('Generation settings must be an object')
                    effective.update(settings)
                settings=validate_policy(effective)
            except (KeyError,ValueError,TypeError) as exc:
                raise UnsupportedTextRequest('Saved Min-P controller has an unsupported policy') from exc
        elif (not isinstance(policy, dict) or type(policy.get('temperature')) not in (int, float)
                or policy['temperature'] != 0):
            raise UnsupportedTextRequest('COW text resume supports an active greedy controller only')
        else:
            explicit_zero = (isinstance(settings,dict) and set(settings)=={'temperature'}
                             and type(settings['temperature']) in (int,float)
                             and settings['temperature']==0)
            if settings is None or settings == {} or explicit_zero:
                # Preserve the legacy greedy route unless a caller explicitly
                # opts into a sampled policy at this generation boundary.
                settings={'temperature':0}
            else:
                from context_time_travel.min_p import validate_policy
                try:
                    effective=dict(temperature=0.7, min_p=0.05, top_k=0, top_p=1.0,
                                   repetition_penalty=1.0)
                    if not isinstance(settings,dict):
                        raise ValueError('Generation settings must be an object')
                    effective.update(settings)
                    settings=validate_policy(effective)
                except (KeyError,ValueError,TypeError) as exc:
                    raise UnsupportedTextRequest('Unsupported sampled continuation policy') from exc
        pending_hex = controller.get('output_pending_hex', '')
        options={}
        namespace=self._namespace(branch)
        if namespace is not None:
            prior=self.token_adapter.active_presentation(branch,head.tip if head is not None else None)
            output_id=prior['presentation']['output_span']['message_id']
            options['presentation']=self._presentation(namespace,'resume',[],operation_id,output_id,
                generation_settings=None if settings=={'temperature':0} else settings,canonical=canonical)
        request = self.token_adapter.generate(
            branch, self._history_view(canonical), revision, operation_id, max_tokens,
            settings=settings, cancelled=cancelled, **_grammar_option(grammar_state),
            **_text_stop_option(text_stops),**options)
        return CowTextRequest(request, self.tokenizer, pending_hex,
            controller_reader=getattr(self.token_adapter, 'snapshot_controller', None),
            text_stops=text_stops or (), stop_state=controller.get('text_stop'))

    def append_user(self, branch, content, revision, operation_id, max_tokens, *,
                    settings=None, cancelled=None, grammar_state=None, seed=None, text_stops=None):
        """Start the next question without rewriting any committed token IDs."""
        if not callable(self.append_user_suffix):
            raise UnsupportedTextRequest('A qualified next-user suffix encoder is required')
        if not isinstance(content, (str,list)):
            raise UnsupportedTextRequest('New user content must be text or ordered image/text parts')
        _, canonical, controller = self.token_adapter.source_state(branch, revision, operation_id)
        self._mixed_controls(canonical,settings,grammar_state,seed,text_stops)
        if not isinstance(controller, dict) or controller.get('status') != 'stopped':
            raise UnsupportedTextRequest('New-user append requires a completed turn; active turns must resume')
        if isinstance(content,list):
            return self._mixed_user_request(branch,content,revision,operation_id,max_tokens,'user_turn',canonical,controller,cancelled,
                grammar_state=grammar_state,settings=settings,seed=seed)
        suffix = tuple(self.append_user_suffix(self._text_tail(canonical), controller, content))
        if not suffix:
            raise UnsupportedTextRequest('New-user suffix must consume a new prompt')
        ids = self._history_view(canonical) + suffix
        options={}
        namespace=self._namespace(branch)
        if namespace is not None:
            options['presentation']=self._presentation(namespace,'user_turn',
                [{'role':'user','content':content}],operation_id,self._message_id(operation_id,'assistant'),seed=seed,canonical=canonical)
        stop_option=_text_stop_option(text_stops)
        settings=_greedy_stop_policy(settings,stop_option.get('text_stops'))
        request = self.token_adapter.generate(branch, ids, revision, operation_id, max_tokens,
            settings=settings, cancelled=cancelled, seed=seed, **_grammar_option(grammar_state),
            **stop_option,**options)
        return CowTextRequest(request, self.tokenizer,
            controller_reader=getattr(self.token_adapter, 'snapshot_controller', None),
            text_stops=stop_option.get('text_stops',()))

    def ask(self, branch, content, revision, operation_id, max_tokens, *, settings=None,
            cancelled=None, grammar_state=None, seed=None, text_stops=None):
        """Append the first question to an idle codebase-only prefix."""
        if not callable(self.append_initial_user_suffix) or not isinstance(content,(str,list)):
            raise UnsupportedTextRequest('A qualified initial-question encoder and text or ordered parts are required')
        head,canonical,controller=self.token_adapter.source_state(branch,revision,operation_id)
        self._mixed_controls(canonical,settings,grammar_state,seed,text_stops)
        if not isinstance(controller,dict) or controller.get('status')!='idle':
            raise UnsupportedTextRequest('First question requires an idle context prefix')
        if self._root_mode_for_source(head, canonical, controller) != 'context_only':
            raise UnsupportedTextRequest('First question requires a renderer-matched closed context root')
        namespace=self._namespace(branch)
        if namespace is None:
            raise UnsupportedTextRequest('First chat question requires a presentation namespace')
        if isinstance(content,list):
            return self._mixed_user_request(branch,content,revision,operation_id,max_tokens,'initial_turn',canonical,controller,cancelled,
                grammar_state=grammar_state,settings=settings,seed=seed)
        presentation=self._presentation(namespace,'initial_turn',[{'role':'user','content':content}],
            operation_id,self._message_id(operation_id,'assistant'),seed=seed,canonical=canonical)
        suffix=tuple(self.append_initial_user_suffix(self._text_tail(canonical),controller,content))
        if not suffix:
            raise UnsupportedTextRequest('First question must consume a prompt suffix')
        stop_option=_text_stop_option(text_stops)
        settings=_greedy_stop_policy(settings,stop_option.get('text_stops'))
        request=self.token_adapter.generate(branch,self._history_view(canonical)+suffix,revision,operation_id,max_tokens,
            settings=settings,cancelled=cancelled,presentation=presentation,seed=seed,
            **_grammar_option(grammar_state),**stop_option)
        return CowTextRequest(request,self.tokenizer,
            controller_reader=getattr(self.token_adapter,'snapshot_controller',None),
            text_stops=stop_option.get('text_stops',()))

    def reseed(self, branch, seed, revision, operation_id):
        """Publish an RNG-only child while retaining the bound assistant identity."""
        head, _canonical, controller = self.token_adapter.source_state(branch,revision,operation_id)
        if isinstance(_canonical,InputHistory):
            self._mixed_controls(_canonical,seed=seed)
        if not isinstance(controller,dict) or controller.get('status') not in ('active','stopped'):
            raise UnsupportedTextRequest('Reseed requires an existing generation scope')
        options={}
        namespace=self._namespace(branch)
        if namespace is not None:
            active=self.token_adapter.published_presentation(branch,head.tip)
            output_id=active['presentation']['output_span']['message_id']
            options['presentation']=self._presentation(namespace,'reseed',[],operation_id,output_id,seed=seed)
        return self.token_adapter.reseed(branch,seed,revision,operation_id,**options)

    def _render_encode(self, messages, *, add_generation_prompt):
        clean = self._messages(messages)
        rendered = self.render(clean, add_generation_prompt=add_generation_prompt)
        if not isinstance(rendered, str):
            raise InvalidOperation('Conversation renderer must return text')
        # Do not default to tokenizer.encode(parse_special=True): that would
        # turn special-looking user text into control tokens. The injected
        # callback must match the serving frontend's literal-data policy.
        ids = self.encode_rendered(clean, rendered, add_generation_prompt)
        return tuple(ids)
