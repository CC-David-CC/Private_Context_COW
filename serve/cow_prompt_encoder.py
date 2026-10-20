"""Opt-in bridge to the exact prompt rendering/encoding rules in serve.server.Service."""
from .server import Service
from .frontend import literal_tags


class CowPromptEncoder:
    """Lightweight Service-method view with no engine instance or server lifecycle.

    `render` and `encode_rendered` match CowTextAdapter's callbacks. Encoding
    delegates to Service.encode_prompt so quoted special-looking message text
    uses the same literal-data protection as ordinary serving.
    """
    def __init__(self, tokenizer, template, template_kwargs=None):
        self.tok = tokenizer
        self.template = template
        specials = getattr(tokenizer, 'special_tokens', None)
        if specials is None:
            specials = (*getattr(tokenizer, 'control_tokens', ()), *getattr(tokenizer, 'ALWAYS', ()))
        self.literals = literal_tags(specials)
        self.effort_end = False
        self.template_kwargs = dict(template_kwargs or {})

    def render(self, messages, *, add_generation_prompt):
        kwargs = {**self.template_kwargs, 'add_generation_prompt': add_generation_prompt}
        return self.render_prompt(messages, None, kwargs)

    def render_prompt(self, messages, tools, kwargs):
        return Service.render_prompt(self, messages, tools, kwargs)

    def encode_rendered(self, messages, rendered, add_generation_prompt):
        expected = self.render(messages, add_generation_prompt=add_generation_prompt)
        if rendered != expected:
            raise ValueError('Rendered prompt changed between render and encode')
        # Service.encode_prompt performs literal_tags_for_data, mark/unmark
        # protection and special-token parsing exactly as the regular route.
        kwargs = {**self.template_kwargs, 'add_generation_prompt': add_generation_prompt}
        return Service.encode_prompt(self, messages, None,
                                    kwargs)

    def append_user_suffix(self, canonical, controller, content):
        """Encode only a new Qwen text turn after a consumed im_end boundary.

        Never render or tokenize the previous assistant output. This deliberately
        has its own token boundary: the separator is encoded separately from the
        new user frame. The native execution profile must retain that arrangement.
        """
        if not isinstance(content, str):
            raise ValueError('New user content must be plain text')
        self._user_suffix_thinking()
        end = tuple(self.tok.encode('<|im_end|>', parse_special=True))
        if len(end) != 1:
            raise ValueError('New-user append requires a single im_end control token')
        if not isinstance(controller, dict) or controller.get('status') != 'stopped':
            raise ValueError('New-user append requires a stopped assistant turn')
        eos_boundary = (bool(canonical) and canonical[-1] == end[0]
                        and end[0] in controller.get('policy', {}).get('eos_tokens', []))
        text_stop = controller.get('text_stop')
        matched_text_stop = False
        if text_stop is not None:
            try:
                from context_time_travel.text_stops import TextStopMatcher
                matcher = TextStopMatcher.restore(text_stop)
                matched_text_stop = matcher.export_state() == text_stop and text_stop['stopped']
            except (TypeError, ValueError, KeyError):
                matched_text_stop = False
        stopped_before_eos = matched_text_stop and (not canonical or canonical[-1] != end[0])
        if not eos_boundary and not stopped_before_eos:
            raise ValueError('New-user append requires a consumed im_end EOS or matched text stop')
        separator = tuple(self.tok.encode('\n', parse_special=False))
        # A text stop leaves the previous canonical answer exactly where the
        # stop-bearing token ended. Consume im_end as part of this new prompt
        # suffix; never rewrite the already published assistant history.
        return ((end if stopped_before_eos else ()) + separator + self._user_frame(content))

    def append_initial_user_suffix(self, canonical, controller, content):
        """Append a first question after an idle context's closed message frame."""
        end=tuple(self.tok.encode('<|im_end|>',parse_special=True))
        newline=tuple(self.tok.encode('\n',parse_special=False))
        boundary=end+newline
        if (len(end)!=1 or not newline or not isinstance(controller,dict)
                or controller.get('status')!='idle' or len(canonical)<len(boundary)
                or tuple(canonical[-len(boundary):])!=boundary):
            raise ValueError('First question requires an idle context ending in im_end and newline')
        return self._user_frame(content)

    def initial_root_mode(self, canonical, controller):
        """Classify an idle initial root from its exact pinned token suffix.

        This is a read-only capability check, not a role or annotation guess.
        The prepared suffix is measured by differentially rendering the same
        reference user message with and without the assistant generation frame.
        """
        # Root and new-user suffixes are independently checked against the
        # exact pinned template; neither may rewrite canonical prefix tokens.
        if (set(self.template_kwargs) != {'enable_thinking'}
                or type(self.template_kwargs['enable_thinking']) is not bool
                or not isinstance(controller, dict)
                or controller.get('status') != 'idle'
                or not isinstance(canonical, (tuple, list))):
            return 'unsupported'
        end = tuple(self.tok.encode('<|im_end|>', parse_special=True))
        newline = tuple(self.tok.encode('\n', parse_special=False))
        if len(end) != 1 or not newline:
            return 'unsupported'
        closed_suffix = end + newline
        tokens = tuple(canonical)
        if len(tokens) >= len(closed_suffix) and tokens[-len(closed_suffix):] == closed_suffix:
            return 'context_only'

        reference = [{'role': 'user', 'content': 'COW root capability reference'}]
        without_generation = tuple(self.encode_rendered(
            reference, self.render(reference, add_generation_prompt=False), False))
        with_generation = tuple(self.encode_rendered(
            reference, self.render(reference, add_generation_prompt=True), True))
        if (len(with_generation) <= len(without_generation)
                or with_generation[:len(without_generation)] != without_generation):
            return 'unsupported'
        generation_suffix = with_generation[len(without_generation):]
        if len(tokens) >= len(generation_suffix) and tokens[-len(generation_suffix):] == generation_suffix:
            return 'prepared_answer'
        return 'unsupported'

    def _user_suffix_thinking(self):
        if (set(self.template_kwargs) != {'enable_thinking'}
                or type(self.template_kwargs['enable_thinking']) is not bool):
            raise ValueError('User suffix requires an explicit qualified thinking profile')
        return self.template_kwargs['enable_thinking']

    def _user_frame(self, content):
        if not isinstance(content, str):
            raise ValueError('User frame requires plain text')
        thinking = self._user_suffix_thinking()
        messages = [{'role': 'user', 'content': content}]
        if thinking:
            # Service injects a reasoning-system message when thinking is on.
            # Render only a fixed synthetic anchor, never the retained history.
            # Removing its exact text AND token prefix gives the new user frame
            # without repeating that system message or retokenizing old output.
            anchor = [{'role': 'system', 'content': 'COW suffix boundary reference.'}]
            prefix = self.render(anchor, add_generation_prompt=False)
            complete = self.render(anchor + messages, add_generation_prompt=True)
            prefix_ids = tuple(self.encode_rendered(anchor, prefix, False))
            complete_ids = tuple(self.encode_rendered(anchor + messages, complete, True))
            if (not prefix or not prefix_ids or not complete.startswith(prefix)
                    or complete_ids[:len(prefix_ids)] != prefix_ids):
                raise ValueError('Thinking suffix anchor text/token prefix is not stable')
            rendered = complete[len(prefix):]
            frame = complete_ids[len(prefix_ids):]
        else:
            rendered = self.render(messages, add_generation_prompt=True)
            # Keep the qualified non-thinking encoding byte-identical.
            frame = tuple(self.encode_rendered(messages, rendered, True))
        tail = '<think>\n' if thinking else '<think>\n\n</think>\n\n'
        expected = ('<|im_start|>user\n' + content.strip() + '<|im_end|>\n'
                    '<|im_start|>assistant\n' + tail)
        if rendered != expected:
            raise ValueError('Template is incompatible with the qualified new-user suffix layout')
        # Service protects literal special-looking user text before tokenization.
        start = tuple(self.tok.encode('<|im_start|>', parse_special=True))
        end = tuple(self.tok.encode('<|im_end|>', parse_special=True))
        if len(start) != 1 or frame.count(start[0]) != 2 or frame.count(end[0]) != 1:
            raise ValueError('New-user suffix contains unexpected control-token boundaries')
        return frame
