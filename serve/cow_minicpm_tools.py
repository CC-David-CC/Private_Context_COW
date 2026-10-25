"""Owner-bound MiniCPM5 XML tools. No model work or general XML evaluation.

The pinned llama.cpp MiniCPM5 grammar uses bare function/param tags, CDATA for
strings, and JSON for other parameter values. Qwen's codec is separate.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import re

from jinja2 import TemplateError, Undefined


DIALECT = 'minicpm5-xml-v1'
SERIALIZER = 'minicpm5-json-values-v1'
MAX_BUFFER = 262144
MAX_CALLS = 32
MAX_CALL_CHARS = 1048576
_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_.:-]{0,127}\Z')
_TYPES = {'object', 'array', 'string', 'number', 'integer', 'boolean', 'null'}
_KEYWORDS = {'type', 'properties', 'required', 'additionalProperties', 'items',
             'enum', 'const', 'anyOf', 'oneOf', 'allOf', 'minimum', 'maximum',
             'exclusiveMinimum', 'exclusiveMaximum', 'minLength', 'maxLength',
             'minItems', 'maxItems', 'uniqueItems', 'description', 'title', 'default'}


def strict_json(value):
    """No Python repr, NaN/Infinity, non-string object keys, or coercion."""
    def check(v, depth=0):
        if depth > 32:
            raise ValueError('MiniCPM JSON nesting limit exceeded')
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float and math.isfinite(v):
            return
        if type(v) is list:
            for item in v: check(item, depth + 1)
            return
        if type(v) is dict and all(type(k) is str for k in v):
            for item in v.values(): check(item, depth + 1)
            return
        raise ValueError('MiniCPM values must be finite JSON values with string keys')
    check(value)
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def json_value_finalizer(value):
    # Undefined must remain Jinja Undefined, and strings must not gain JSON quotes
    # or lose their template-controlled CDATA wrapping.
    if isinstance(value, Undefined) or type(value) is str:
        return value
    if value is None or type(value) in (dict, list, bool, int, float):
        return strict_json(value)
    raise ValueError('Unsupported MiniCPM template value')


@dataclass(frozen=True)
class ToolBinding:
    template_sha256: str
    dialect: str = DIALECT
    serializer: str = SERIALIZER

    def receipt(self):
        return asdict(self)


def compatible_template(source):
    return (type(source) is str and all(s in source for s in (
        'Tool usage guidelines:', '<function name="', '<param name="',
        '</function>', '</param>', '<tool_response>'))
        and '<function=' not in source and '<parameter=' not in source)


def bind_template(source, *, expected_template_sha256):
    """Called by the owner loading verified model metadata, never request data."""
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if expected_template_sha256 != digest or not compatible_template(source):
        raise ValueError('MiniCPM tool binding does not match the exact template')
    return ToolBinding(digest)


def validate_binding(binding, source):
    if type(binding) is not ToolBinding or binding != bind_template(
            source, expected_template_sha256=binding.template_sha256):
        raise ValueError('Explicit owner MiniCPM tool binding required')
    return binding


def _schema_types(schema):
    if schema is True: return set(_TYPES)
    if schema is False: return set()
    kind = schema.get('type')
    kinds = set(kind if type(kind) is list else [kind]) if kind else set(_TYPES)
    if 'anyOf' in schema or 'oneOf' in schema:
        variants = schema.get('anyOf', schema.get('oneOf'))
        kinds &= set().union(*(_schema_types(s) for s in variants))
    for item in schema.get('allOf', []): kinds &= _schema_types(item)
    if 'const' in schema or 'enum' in schema:
        values = [schema['const']] if 'const' in schema else schema['enum']
        inferred = set()
        for value in values:
            inferred.add('null' if value is None else
                         {str: 'string', bool: 'boolean', int: 'integer', float: 'number',
                          list: 'array', dict: 'object'}[type(value)])
        # integer values also satisfy number schemas.
        if 'integer' in inferred: inferred.add('number')
        kinds &= inferred
    return kinds


def _check_schema(schema, depth=0, budget=None):
    budget = [0] if budget is None else budget
    budget[0] += 1
    if depth > 24 or budget[0] > 4096:
        raise ValueError('MiniCPM schema complexity limit exceeded')
    if type(schema) is bool: return
    if type(schema) is not dict or set(schema) - _KEYWORDS:
        raise ValueError('Unsupported MiniCPM tool schema keyword or shape')
    kind = schema.get('type')
    if kind is not None:
        kinds = kind if type(kind) is list else [kind]
        if not kinds or any(type(k) is not str or k not in _TYPES for k in kinds) or len(set(kinds)) != len(kinds):
            raise ValueError('Invalid MiniCPM schema type')
    props = schema.get('properties', {})
    if type(props) is not dict or any(type(k) is not str for k in props):
        raise ValueError('Invalid MiniCPM object properties')
    for value in props.values(): _check_schema(value, depth + 1, budget)
    required = schema.get('required', [])
    if (type(required) is not list or any(type(k) is not str for k in required)
            or len(set(required)) != len(required)):
        raise ValueError('Invalid MiniCPM required properties')
    for key in ('items', 'additionalProperties'):
        if key in schema: _check_schema(schema[key], depth + 1, budget)
    for key in ('anyOf', 'oneOf', 'allOf'):
        if key in schema:
            if type(schema[key]) is not list or not schema[key]:
                raise ValueError('Invalid MiniCPM schema alternatives')
            for value in schema[key]: _check_schema(value, depth + 1, budget)
    if 'enum' in schema and (type(schema['enum']) is not list or not schema['enum']):
        raise ValueError('Invalid MiniCPM enum')
    for key in ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum'):
        if key in schema and (type(schema[key]) not in (int, float)
                              or type(schema[key]) is float and not math.isfinite(schema[key])):
            raise ValueError('Invalid MiniCPM numeric bound')
    for key in ('minLength', 'maxLength', 'minItems', 'maxItems'):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError('Invalid MiniCPM length bound')
    if 'uniqueItems' in schema and type(schema['uniqueItems']) is not bool:
        raise ValueError('Invalid MiniCPM uniqueItems')
    strict_json(schema)


def _equal(a, b):
    if type(a) is bool or type(b) is bool: return type(a) is type(b) and a == b
    if type(a) is dict and type(b) is dict:
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if type(a) is list and type(b) is list:
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _value_key(value):
    if type(value) is dict:
        return ('object', tuple(sorted((k, _value_key(v)) for k, v in value.items())))
    if type(value) is list:
        return ('array', tuple(_value_key(v) for v in value))
    return ('number' if type(value) in (int, float) else type(value).__name__, value)


def _matches(schema, value):
    if type(schema) is bool: return schema
    kind = schema.get('type')
    if kind is not None:
        kinds = kind if type(kind) is list else [kind]
        number = type(value) is int or type(value) is float and math.isfinite(value)
        tests = dict(object=type(value) is dict, array=type(value) is list,
                     string=type(value) is str, boolean=type(value) is bool,
                     null=value is None, number=number,
                     integer=number and (type(value) is int or value.is_integer()))
        if not any(tests[k] for k in kinds): return False
    if 'enum' in schema and not any(_equal(value, v) for v in schema['enum']): return False
    if 'const' in schema and not _equal(value, schema['const']): return False
    for key in ('anyOf', 'oneOf', 'allOf'):
        if key in schema:
            matches = sum(_matches(s, value) for s in schema[key])
            if (key == 'anyOf' and not matches or key == 'oneOf' and matches != 1
                    or key == 'allOf' and matches != len(schema[key])): return False
    if type(value) is dict:
        if not all(k in value for k in schema.get('required', [])): return False
        props = schema.get('properties', {})
        if not all(_matches(props.get(k, schema.get('additionalProperties', True)), v) for k, v in value.items()): return False
    if type(value) is list:
        if not all(_matches(schema.get('items', True), x) for x in value): return False
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', len(value)): return False
        if schema.get('uniqueItems') and len({_value_key(x) for x in value}) != len(value): return False
    if type(value) is str:
        if len(value) < schema.get('minLength', 0) or len(value) > schema.get('maxLength', len(value)): return False
    if type(value) in (int, float):
        for key, invalid in [('minimum', lambda x: value < x), ('maximum', lambda x: value > x),
                             ('exclusiveMinimum', lambda x: value <= x), ('exclusiveMaximum', lambda x: value >= x)]:
            if key in schema and invalid(schema[key]): return False
    return True


def admit_tools(tools):
    """Return private immutable-by-copy schemas; refuse unsupported declarations."""
    if type(tools) is not list or not 1 <= len(tools) <= 128:
        raise ValueError('MiniCPM tools require one to 128 declarations')
    if len(strict_json(tools).encode('utf-8')) > MAX_BUFFER:
        raise ValueError('MiniCPM tool declarations exceed the admission limit')
    result = {}
    for tool in tools:
        if (type(tool) is not dict or type(tool.get('name')) is not str
                or not _NAME.fullmatch(tool['name'])):
            raise ValueError('Invalid MiniCPM declared tool name')
        name = tool['name']
        if name in result: raise ValueError('Duplicate MiniCPM tool declaration')
        schema = tool.get('parameters', {'type': 'object', 'properties': {}})
        _check_schema(schema)
        if type(schema) is not dict or schema.get('type') != 'object':
            raise ValueError('MiniCPM tool parameters require an explicit object schema')
        if any(key in schema for key in ('allOf', 'anyOf', 'oneOf')):
            raise ValueError('MiniCPM root schema combinators are not supported on the named-parameter wire')
        props = schema.get('properties', {})
        if any(not _NAME.fullmatch(k) for k in props) or not set(schema.get('required', [])).issubset(props):
            raise ValueError('MiniCPM wire parameters must be declared safe names')
        for prop in props.values():
            kinds = _schema_types(prop)
            if not kinds or 'string' in kinds and kinds != {'string'}:
                raise ValueError('MiniCPM parameter has ambiguous string/JSON wire types')
        result[name] = json.loads(strict_json(schema))
    return result


def validate_history(messages, schemas):
    for message in messages:
        for call in message.get('tool_calls') or []:
            fn = call.get('function', call)
            name, args = fn.get('name'), fn.get('arguments')
            if name not in schemas or type(args) is not dict:
                raise ValueError('MiniCPM history needs a declared tool and JSON object arguments')
            strict_json(args)
            if set(args) - set(schemas[name].get('properties', {})) or not _matches(schemas[name], args):
                raise ValueError('MiniCPM history arguments do not match the declared schema')
            for key, value in args.items():
                if type(value) is str and ']]>' in value and any(c in value for c in '<&\n'):
                    raise ValueError('MiniCPM template cannot represent this CDATA terminator')


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError('Duplicate JSON object key')
        out[key] = value
    return out


def _constant(_): raise ValueError('Nonfinite JSON value')
_DECODER = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)


def load_json(text):
    """Strict MiniCPM transport/history JSON; Qwen's normalizer is unchanged."""
    if type(text) is bytes: text = text.decode('utf-8')
    try: value = _DECODER.decode(text)
    except RecursionError:
        raise ValueError('MiniCPM JSON nesting limit exceeded') from None
    strict_json(value)
    return value


def validate_wire_history(request, dialect):
    """Refuse malformed API argument bytes before legacy normalization can repair them."""
    def objects(value, field):
        if value is None: return []
        if type(value) is not list or any(type(item) is not dict for item in value):
            raise ValueError('MiniCPM ' + field + ' requires an array of objects, not encoded wrappers')
        return value

    def declarations(value, field='tools'):
        for tool in objects(value, field):
            if 'function' in tool and type(tool['function']) is not dict:
                raise ValueError('MiniCPM tool function must be an object')
            if tool.get('type') == 'namespace': declarations(tool.get('tools'), field + '.tools')

    declarations(request.get('tools'))
    values = []
    if dialect == 'responses':
        items = request.get('input')
        # Standard Responses input:string is a USER MESSAGE, even if it looks
        # like JSON. It must never be decoded as a control/history container.
        if type(items) is not str:
            for item in objects(items, 'input'):
                if item.get('type') == 'function_call': values.append(item.get('arguments'))
                if item.get('type') == 'additional_tools': declarations(item.get('tools'), 'additional_tools.tools')
    else:
        for message in objects(request.get('messages'), 'messages'):
            for call in objects(message.get('tool_calls'), 'tool_calls'):
                if 'function' in call and type(call['function']) is not dict:
                    raise ValueError('MiniCPM historical function must be an object')
                fn = call.get('function', call)
                values.append(fn.get('arguments'))
    for value in values:
        if type(value) is str:
            if len(value) > MAX_BUFFER: raise ValueError('MiniCPM history arguments exceed the buffer bound')
            value = load_json(value)
        if type(value) is not dict:
            raise ValueError('MiniCPM history requires explicit JSON object arguments')
        strict_json(value)


class MiniCPMOutputParser:
    """Bounded incremental codec; tool events commit only on a clean stop.

    Unquoted function tags are reserved executable syntax even after prose,
    outside backtick/tilde fenced code and inline backticks. CDATA is always
    data. A malformed candidate refuses the
    turn: no earlier pending call becomes actionable.
    """
    def __init__(self, *, thinking=False, tools=None, parallel=True, max_buffer=MAX_BUFFER):
        if type(thinking) is not bool or type(parallel) is not bool:
            raise ValueError('Explicit MiniCPM parser modes required')
        if type(max_buffer) is not int or not 256 <= max_buffer <= MAX_BUFFER:
            raise ValueError('Invalid MiniCPM parser buffer bound')
        self.schemas = admit_tools(tools) if tools else {}
        self.parallel, self.limit = parallel, max_buffer
        self.buf, self.fence = '', None
        self.inline, self.ticks = 0, 0
        self._reset_line()
        self.reasoning, self.reasoning_start = thinking, thinking
        self.pending, self.call_chars, self.done = [], 0, False

    def _reset_line(self, *, had_content=False):
        self.line_started = had_content
        self.line_inline = bool(self.inline)
        self.line_marker, self.line_run = None, 0
        self.line_run_ended, self.line_trailing = False, False

    def _settle_ticks(self):
        if self.ticks and not self.fence:
            if not self.inline: self.inline = self.ticks
            elif self.inline == self.ticks: self.inline = 0
        self.ticks = 0

    def _quoted(self):
        # A following non-backtick marker proves the preceding delimiter run
        # complete. Match its exact width, including multi-line inline spans.
        self._settle_ticks()
        opening_fence = (not self.line_inline and self.line_marker in ('`', '~')
                         and self.line_run >= 3)
        return bool(self.fence or self.inline or opening_fence)

    def _track(self, text):
        for char in text:
            if char == '\n':
                self._settle_ticks()
                if (not self.fence and not self.line_inline and self.line_marker in ('`', '~')
                        and self.line_run >= 3):
                    self.fence = (self.line_marker, self.line_run)
                    self.inline = 0
                elif (self.fence and self.line_marker == self.fence[0]
                        and self.line_run >= self.fence[1] and not self.line_trailing):
                    self.fence, self.inline = None, 0
                self._reset_line()
                continue
            if char == '`': self.ticks += 1
            else: self._settle_ticks()
            if not self.line_started:
                if char.isspace(): continue
                self.line_started = True
                if char in ('`', '~'): self.line_marker, self.line_run = char, 1
            elif self.line_marker and not self.line_run_ended:
                if char == self.line_marker: self.line_run += 1
                else:
                    self.line_run_ended = True
                    self.line_trailing = not char.isspace()
            elif self.line_marker and not char.isspace(): self.line_trailing = True

    def _allowed(self):
        return not self._quoted()

    @staticmethod
    def _hold(text, marker):
        return next((n for n in range(min(len(text), len(marker) - 1), 0, -1) if marker.startswith(text[-n:])), 0)

    def _call(self):
        gt = self.buf.find('>')
        if gt < 0:
            if len(self.buf) > 256: raise ValueError('MiniCPM function header exceeds bound')
            return None
        header = re.fullmatch(r'<function name="([A-Za-z_][A-Za-z0-9_.:-]{0,127})">', self.buf[:gt + 1])
        if header is None: raise ValueError('Malformed MiniCPM function header')
        name = header[1]
        if name not in self.schemas: raise ValueError('Undeclared MiniCPM tool')
        props = self.schemas[name].get('properties', {})
        pos, args = gt + 1, {}
        while True:
            pos += len(self.buf[pos:]) - len(self.buf[pos:].lstrip())
            rest = self.buf[pos:]
            if rest.startswith('</function>'):
                if not _matches(self.schemas[name], args): raise ValueError('MiniCPM call violates its declared schema')
                from .frontend import ToolCall
                return pos + len('</function>'), ToolCall(name, args)
            if not rest or any(s.startswith(rest) for s in ('<param name="', '</function>')): return None
            gt = self.buf.find('>', pos)
            if gt < 0: return None
            param = re.fullmatch(r'<param name="([A-Za-z_][A-Za-z0-9_.:-]{0,127})">', self.buf[pos:gt + 1])
            if param is None: raise ValueError('Malformed or nested MiniCPM parameter')
            key = param[1]
            if key not in props or key in args: raise ValueError('Undeclared or duplicate MiniCPM parameter')
            pos = gt + 1
            if _schema_types(props[key]) == {'string'}:
                rest = self.buf[pos:]
                if '<![CDATA['.startswith(rest) and len(rest) < 9: return None
                if rest.startswith('<![CDATA['):
                    end = self.buf.find(']]>', pos + 9)
                    if end < 0: return None
                    value, pos = self.buf[pos + 9:end], end + 3
                    tail = self.buf[pos:]
                    if '</param>'.startswith(tail) and len(tail) < 8: return None
                    if not tail.startswith('</param>'): raise ValueError('Malformed MiniCPM CDATA close')
                    pos += 8
                else:
                    end = self.buf.find('</param>', pos)
                    if end < 0: return None
                    value = self.buf[pos:end]
                    if any(c in value for c in '<&\n'): raise ValueError('MiniCPM string markup requires CDATA')
                    pos = end + 8
            else:
                start = pos + len(self.buf[pos:]) - len(self.buf[pos:].lstrip())
                try: value, end = _DECODER.raw_decode(self.buf, start)
                except json.JSONDecodeError: return None
                except RecursionError:
                    raise ValueError('MiniCPM JSON nesting limit exceeded') from None
                strict_json(value)
                pos = end + len(self.buf[end:]) - len(self.buf[end:].lstrip())
                tail = self.buf[pos:]
                if '</param>'.startswith(tail) and len(tail) < 8: return None
                if not tail.startswith('</param>'):
                    # raw_decode may have consumed only a number's settled
                    # prefix: 1 from 1e, or -2 from -2.5e+. Do not declare the
                    # remaining suffix invalid until the actual delimiter has
                    # arrived. The incremental buffer remains bounded.
                    if '</param>' not in tail: return None
                    raise ValueError('Invalid MiniCPM JSON parameter close')
                pos += 8
            if not _matches(props[key], value): raise ValueError('MiniCPM parameter violates its schema')
            args[key] = value

    def feed(self, text):
        from .frontend import Event
        if self.done: raise ValueError('MiniCPM parser already settled')
        if type(text) is not str or len(self.buf) + len(text) > self.limit:
            raise ValueError('MiniCPM incremental buffer limit exceeded')
        self.buf += text
        out = []
        while self.buf:
            if self.reasoning:
                if self.reasoning_start:
                    if '<think>'.startswith(self.buf) and len(self.buf) < 7: return out
                    if self.buf.startswith('<think>'): self.buf = self.buf[7:]
                    self.reasoning_start = False
                at = self.buf.find('</think>')
                if at < 0:
                    safe = len(self.buf) - self._hold(self.buf, '</think>')
                    if safe: out.append(Event('reasoning', self.buf[:safe])); self.buf = self.buf[safe:]
                    return out
                if at: out.append(Event('reasoning', self.buf[:at]))
                self.buf, self.reasoning = self.buf[at + 8:], False
                self.inline, self.ticks = 0, 0
                self._reset_line()
                continue
            markers = ('<function', '<think>', '</think>')
            candidates = [(self.buf.find(marker), marker) for marker in markers if marker in self.buf]
            if not candidates:
                safe = len(self.buf) - max(self._hold(self.buf, marker) for marker in markers)
                if safe:
                    value, self.buf = self.buf[:safe], self.buf[safe:]
                    self._track(value); out.append(Event('content', value))
                return out
            at, marker = min(candidates)
            if at:
                value, self.buf = self.buf[:at], self.buf[at:]
                self._track(value); out.append(Event('content', value))
            if marker != '<function':
                if not self._quoted():
                    raise ValueError('Unexpected MiniCPM thinking block in content; thinking is owner configured')
                value, self.buf = self.buf[:len(marker)], self.buf[len(marker):]
                self._track(value); out.append(Event('content', value)); continue
            if not self._allowed() or not self.schemas:
                value, self.buf = self.buf[:9], self.buf[9:]
                self._track(value); out.append(Event('content', value)); continue
            parsed = self._call()
            if parsed is None: return out
            end, call = parsed
            self.call_chars += end
            if self.call_chars > MAX_CALL_CHARS or len(self.pending) >= (MAX_CALLS if self.parallel else 1):
                raise ValueError('MiniCPM call count or cumulative size limit exceeded')
            self.pending.append(call)
            self.buf, self.inline, self.ticks = self.buf[end:], 0, 0
            self._reset_line(had_content=True)
        return out

    def finish(self, reason='stop'):
        from .frontend import Event
        if self.done: raise ValueError('MiniCPM parser already settled')
        self.done = True
        if self.pending and reason != 'stop': raise ValueError('MiniCPM calls require a naturally completed turn')
        if (self.buf and not self.reasoning and not self._quoted()
                and len(self.buf) >= 3 and any(marker.startswith(self.buf) for marker in ('<think>', '</think>'))):
            raise ValueError('Incomplete unexpected MiniCPM thinking marker')
        if self.buf and not self.reasoning and self._allowed() and self.schemas and (
                self.buf.startswith('<function') or '<function'.startswith(self.buf)):
            raise ValueError('Incomplete or malformed MiniCPM call')
        if self.reasoning and self.pending: raise ValueError('Incomplete MiniCPM reasoning')
        out = [Event('reasoning' if self.reasoning else 'content', self.buf)] if self.buf else []
        self.buf = ''
        return out + [Event('tool_call', call=call) for call in self.pending]


def template_capabilities(template):
    """Actual codec/template history roundtrip; no claim about model quality."""
    tools = [dict(name='minicpm_caps', description='codec probe', parameters=dict(
        type='object', properties=dict(text=dict(type='string'), payload=dict(type='object')),
        required=['text', 'payload']))]
    user = dict(role='user', content='MiniCPM capability probe')
    def probe(count):
        expected = [dict(text='probe' + str(i), payload=dict(ok=True)) for i in range(count)]
        try:
            rendered = template.render([user, dict(role='assistant', content='', tool_calls=[dict(
                function=dict(name='minicpm_caps', arguments=args)) for args in expected]),
                *[dict(role='tool', content='minicpm_result_' + str(i)) for i in range(count)]],
                tools=tools, bos_token='<s>', add_generation_prompt=False, enable_thinking=False)
            start, end = rendered.find('<function name="minicpm_caps">'), rendered.rfind('</function>')
            if start < 0 or end < 0: return False
            parser = MiniCPMOutputParser(tools=tools)
            events = parser.feed(rendered[start:end + 11]) + parser.finish()
        except (ValueError, TypeError, TemplateError):
            return False
        return ([e.call.arguments for e in events if e.kind == 'tool_call'] == expected
                and all('minicpm_result_' + str(i) in rendered for i in range(count)))
    ok = probe(1)
    return dict(supports_tools=ok, supports_tool_calls=ok, supports_parallel_tool_calls=ok and probe(2))
