"""CPU HTTP/admission gates. A scripted fake COW request is not model inference."""
import copy
from concurrent.futures import Future
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from serve.cow_compat import Frontend, make_server
from serve.cow_compat_cache import PrefixIndex
from serve.cow_renderer import renderer_identity, renderer_identity_receipt
from serve.cow_minicpm_tools import bind_template
from serve.test_minicpm_tools import MINIMAL_TEMPLATE, SCHEMA, wire


def future(value):
    f = Future(); f.set_result(value); return f


def encoder(bound=True):
    binding = bind_template(MINIMAL_TEMPLATE, expected_template_sha256=hashlib.sha256(MINIMAL_TEMPLATE.encode()).hexdigest()) if bound else None
    tokenizer = SimpleNamespace(tokens=['a', 'b', 'ab'], pre='minicpm5', token_types=[1, 1, 1],
        special_ids={'tokenizer.ggml.bos_token_id': 0}, ranks={('a', 'b'): 0})
    return SimpleNamespace(tool_binding=binding, tok=tokenizer, template_kwargs=dict(enable_thinking=False), template=SimpleNamespace(
        source=MINIMAL_TEMPLATE, tool_binding=binding,
        caps=dict(supports_tool_calls=bound, supports_parallel_tool_calls=bound)))


def frontend(bound=True, thinking=False):
    adapter = SimpleNamespace(snapshot_controller=lambda snapshot: dict(status='stopped'))
    result = Frontend(adapter, encoder(bound), dict(model_contract_id='model',
        execution_profile_id='execution', execution=dict(capacity=100000)), 'owned', 256, thinking=thinking)
    result.cache = Mock(identity=result.cache.identity)
    request = SimpleNamespace(durable=future(dict(snapshot='owned-snapshot')), snapshot=future(dict(tokens=[1, 2, 3])))
    result.cache.start.return_value = ('owned-branch', request, 0)
    return result


def chat_request(**extra):
    return dict(model='owned', messages=[dict(role='user', content='CPU synthetic request')],
        tools=[dict(type='function', function=SCHEMA)], **extra)


@contextmanager
def endpoint(raw, *, bound=True, length=False, cancelled=False):
    f = frontend(bound)
    if length: f.adapter.snapshot_controller = lambda snapshot: dict(status='active')
    if cancelled:
        from concurrent.futures import CancelledError
        durable = Future(); durable.set_exception(CancelledError('owned request cancelled'))
        f.cache.start.return_value[1].durable = durable
    class View:
        def __init__(self, *args, **kwargs): pass
        def iter_events(self):
            for part in [raw[:5], raw[5:17], raw[17:]]:
                yield dict(event='text', text=part)
    stop = threading.Event()
    with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]), patch('serve.cow_compat.CowTextRequest', View):
        server = make_server(f, '127.0.0.1', 0, 'owned-key', stop)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try: yield f, 'http://127.0.0.1:' + str(server.server_port)
        finally: stop.set(); server.shutdown(); server.server_close(); thread.join(5)


def post(url, req):
    request = Request(url, data=json.dumps(req).encode(), headers=dict(
        Authorization='Bearer owned-key', **{'Content-Type': 'application/json'}))
    with urlopen(request, timeout=5) as response: return response.read().decode()


class MiniCPMCompatibilityTests(unittest.TestCase):
    def test_noncanonical_chat_control_wrappers_refuse_before_legacy_repair(self):
        tool = dict(name='raw', parameters=dict(type='object', properties=dict(arguments=dict(type='string'))))
        declarations = [dict(type='function', function=tool)]
        cases = []
        for raw in ['not JSON', '{"arguments":"x","arguments":"y"}', '{"arguments":NaN}']:
            fn = dict(name='raw', arguments=raw)
            call = dict(function=fn)
            message = dict(role='assistant', content='', tool_calls=[call])
            messages = [dict(role='user', content='synthetic'), message]
            cases += [dict(messages=json.dumps(messages), tools=declarations),
                      dict(messages=[json.dumps(m) for m in messages], tools=declarations),
                      dict(messages=[messages[0], dict(message, tool_calls=json.dumps([call]))], tools=declarations),
                      dict(messages=[messages[0], dict(message, tool_calls=[json.dumps(call)])], tools=declarations),
                      dict(messages=[messages[0], dict(message, tool_calls=[dict(function=json.dumps(fn))])], tools=declarations)]
        user = [dict(role='user', content='synthetic')]
        cases += [dict(messages=user, tools=json.dumps(declarations)),
                  dict(messages=user, tools=[json.dumps(declarations[0])]),
                  dict(messages=user, tools=[dict(type='function', function=json.dumps(tool))]),
                  dict(messages=user, tools='[{"type":"function","function":{"name":"raw","name":"other","parameters":{"type":"object"}}}]'),
                  dict(messages=user, tools='[{"type":"function","function":{"name":"raw","parameters":{"type":"object","properties":{"arguments":{"type":"string","type":"number"}}}}}]')]
        f = frontend()
        with patch('serve.cow_compat.openai_to_messages', side_effect=AssertionError('must not normalize')) as normalizer, \
             patch('serve.cow_compat.Service.encode_prompt', side_effect=AssertionError('must not encode')) as encode:
            for req in cases:
                with self.subTest(req=req), self.assertRaises(ValueError): f.prepare(dict(req, model='owned'), 'chat')
            normalizer.assert_not_called(); encode.assert_not_called()
        f.cache.start.assert_not_called()

    def test_noncanonical_responses_containers_refuse_but_string_input_is_user_data(self):
        tool = dict(type='function', name='raw', parameters=dict(type='object', properties=dict(arguments=dict(type='string'))))
        cases = [dict(input=[json.dumps(dict(type='function_call', name='raw', arguments='not JSON'))], tools=[tool]),
                 dict(input='synthetic', tools=json.dumps([tool])),
                 dict(input='synthetic', tools=[json.dumps(tool)]),
                 dict(input='synthetic', tools=[dict(type='function', name='raw', function=json.dumps(tool))]),
                 dict(input='synthetic', tools=[dict(type='namespace', name='n', tools=json.dumps([tool]))]),
                 dict(input=[dict(type='additional_tools', tools=json.dumps([tool]))], tools=None),
                 dict(input=[dict(type='additional_tools', tools=[json.dumps(tool)])], tools=None)]
        f = frontend()
        with patch('serve.cow_compat.responses.input_messages', side_effect=AssertionError('must not normalize')) as normalizer, \
             patch('serve.cow_compat.Service.encode_prompt', side_effect=AssertionError('must not encode')) as encode:
            for req in cases:
                with self.subTest(req=req), self.assertRaises(ValueError): f.prepare(dict(req, model='owned', store=False), 'responses')
            normalizer.assert_not_called(); encode.assert_not_called()
        f.cache.start.assert_not_called()
        raw = '[{"type":"function_call","name":"raw","arguments":"not JSON"}]'
        with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
            for extra in [{}, dict(tools=None), dict(tools=[])]:
                plan = f.prepare(dict(model='owned', input=raw, store=False, **extra), 'responses')
                self.assertEqual(plan['messages'], [dict(role='user', content=raw)])
                self.assertIsNone(plan['tools'])

    def test_none_absent_empty_control_arrays_keep_text_only_semantics(self):
        f = frontend()
        with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
            for tools in [None, []]:
                for supplied in [{}, dict(tool_calls=None), dict(tool_calls=[])]:
                    messages = [dict(role='user', content='synthetic'), dict(role='assistant', content='plain', **supplied)]
                    self.assertEqual(f.prepare(dict(model='owned', messages=messages, tools=tools), 'chat')['messages'][-1]['content'], 'plain')
            for messages in [None, []]:
                with self.assertRaises(ValueError): f.prepare(dict(model='owned', messages=messages), 'chat')

    def test_qwen_legacy_array_wrapper_and_history_repair_behavior_is_unchanged(self):
        from serve.frontend import ChatTemplate
        template = ChatTemplate(Path(__file__).with_name('chat_template.jinja'))
        e = SimpleNamespace(template=template)
        f = Frontend(object(), e, dict(model_contract_id='m', execution_profile_id='e', execution=dict(capacity=10000)), 'owned', 16)
        tool = dict(type='function', function=dict(name='raw', parameters=dict(type='object', properties=dict(arguments=dict(type='string')))))
        for wrapped_calls in [False, True]:
            calls_ = [dict(function=dict(name='raw', arguments='not JSON'))]
            messages = [dict(role='user', content='synthetic'), dict(role='assistant', content='', tool_calls=json.dumps(calls_) if wrapped_calls else calls_)]
            with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
                plan = f.prepare(dict(model='owned', messages=json.dumps(messages), tools=json.dumps([tool])), 'chat')
            self.assertEqual(plan['messages'][1]['tool_calls'][0]['function']['arguments'], dict(arguments='not JSON'))

    def test_text_only_minicpm_family_factory_starts_and_refuses_tools_before_encoding(self):
        from serve.cow_http import model_text_components
        source = '''{{- bos_token }}{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{%- endif %}'''
        metadata = {'general.architecture': 'llama', 'tokenizer.ggml.pre': 'minicpm5', 'tokenizer.chat_template': source}
        tok = SimpleNamespace(special_tokens=['<s>'], encode=lambda text, **kwargs: [1])
        with patch('tools.gguf_reader.GGUFFile', return_value=SimpleNamespace(metadata=metadata)):
            components = model_text_components('/owned/text-only.gguf', tokenizer=tok, template_kwargs=dict(enable_thinking=False))
        owner = components['render'].__self__
        self.assertIsNone(owner.tool_binding); self.assertEqual(owner.tool_family, 'minicpm5')
        self.assertEqual(owner.render([dict(role='user', content='hello')], add_generation_prompt=True),
            '<s><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        f = Frontend(object(), owner, dict(model_contract_id='m', execution_profile_id='e', execution=dict(capacity=10000)), 'owned', 16)
        with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
            self.assertEqual(f.prepare(dict(model='owned', messages=[dict(role='user', content='hello')]), 'chat')['full'], (1, 2))
        # Even a misleading generic capability probe cannot select the Qwen
        # tool parser for an explicitly MiniCPM-family owner without a codec.
        owner.template.caps['supports_tool_calls'] = True
        with patch('serve.cow_compat.Service.encode_prompt', side_effect=AssertionError('must not encode')) as encode:
            with self.assertRaises(ValueError): f.prepare(chat_request(), 'chat')
            encode.assert_not_called()

    def test_health_labels_codec_support_without_native_or_quality_claim(self):
        with endpoint('synthetic text') as (f, url):
            req = Request(url + '/health', headers=dict(Authorization='Bearer owned-key'))
            with urlopen(req, timeout=5) as response: health = json.load(response)
            self.assertEqual(health['tool_codec']['qualification'], 'cpu-codec-tests-only')
            self.assertFalse(health['tool_codec']['native_tool_loop_qualified'])
            self.assertFalse(health['tool_codec']['model_quality_qualified'])
        with endpoint('synthetic text', bound=False) as (f, url):
            req = Request(url + '/health', headers=dict(Authorization='Bearer owned-key'))
            with urlopen(req, timeout=5) as response: self.assertNotIn('tool_codec', json.load(response))

    def test_admission_rejects_unsupported_before_encoding_or_native_work(self):
        f = frontend()
        bad = copy.deepcopy(SCHEMA); bad['parameters']['properties']['text']['pattern'] = '.*'
        root_combo = dict(name='root_structure', parameters=dict(type='object', allOf=[dict(properties=dict(x=dict(type='string')), required=['x'])]))
        with patch('serve.cow_compat.Service.encode_prompt', side_effect=AssertionError('must not encode')) as encode:
            for req in [chat_request(tools_override=True), dict(chat_request(), tools=[dict(type='function', function=bad)]),
                        dict(chat_request(), tools=[dict(type='function', function=root_combo)]),
                        dict(chat_request(), parallel_tool_calls='false'), dict(chat_request(), tool_choice='required')]:
                if req.pop('tools_override', False): req['tools'] = [dict(type='function', function=dict(name='bad"name', parameters=dict(type='object')))]
                with self.assertRaises(ValueError): f.prepare(req, 'chat')
            unknown = frontend(bound=False)
            with self.assertRaises(ValueError): unknown.prepare(chat_request(), 'chat')
            encode.assert_not_called()
        f.cache.start.assert_not_called()

    def test_unsupported_bound_roundtrip_keeps_text_only_available(self):
        f = frontend(); f.encoder.template.caps['supports_tool_calls'] = False
        with patch('serve.cow_compat.Service.encode_prompt', side_effect=AssertionError('must not encode')):
            with self.assertRaises(ValueError): f.prepare(chat_request(), 'chat')
        with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
            self.assertEqual(f.prepare(dict(model='owned', messages=[dict(role='user', content='synthetic')]), 'chat')['full'], (1, 2))

    def test_actual_hermes_dsh_tools_admit_in_both_api_dialects(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/minicpm_external_tool_schemas.json').read_text())
        with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
            for tools in fixture['declarations'].values():
                f = frontend()
                chat = dict(chat_request(), tools=[dict(type='function', function=t) for t in tools])
                original = copy.deepcopy(chat)
                self.assertEqual(len(f.prepare(chat, 'chat')['tools']), len(tools))
                responses = dict(model='owned', input='CPU synthetic request', store=False,
                    tools=[dict(type='function', **t) for t in tools])
                self.assertEqual(len(f.prepare(responses, 'responses')['tools']), len(tools))
                self.assertEqual(chat, original)

    def test_real_qwen_template_admission_and_legacy_cache_identity(self):
        from serve.frontend import ChatTemplate
        template = ChatTemplate(Path(__file__).with_name('chat_template.jinja'))
        self.assertTrue(template.caps['supports_tool_calls'])
        e = SimpleNamespace(template=template)
        profile = dict(model_contract_id='m', execution_profile_id='e', execution=dict(capacity=10000))
        f = Frontend(object(), e, profile, 'owned', 16)
        self.assertEqual(f.cache.identity, ('m', 'e', hashlib.sha256(template.source.encode()).hexdigest()))
        with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
            self.assertEqual(f.prepare(chat_request(), 'chat')['tools'], [SCHEMA])

    def test_raw_history_json_is_never_repaired_by_shared_normalizers(self):
        # A real tool may have a parameter named "arguments". That must not make
        # the shared legacy malformed-JSON wrapper a valid historical tool call.
        tool = dict(name='raw', parameters=dict(type='object', properties=dict(arguments=dict(type='string'))))
        f = frontend()
        for raw in ['not JSON', '[1]', 'null', '1', '{"arguments":"x","arguments":"y"}',
                    '{"arguments":NaN}', None, dict(arguments=float('inf'))]:
            messages = [dict(role='user', content='synthetic'), dict(role='assistant', content='',
                tool_calls=[dict(function=dict(name='raw', arguments=raw))])]
            req = dict(chat_request(), messages=messages, tools=[dict(type='function', function=tool)])
            response_req = dict(model='owned', input=[dict(type='function_call', name='raw', arguments=raw)],
                tools=[dict(type='function', **tool)], store=False)
            with patch('serve.cow_compat.Service.encode_prompt', side_effect=AssertionError('must not encode')) as encode:
                for value, dialect in [(req, 'chat'), (response_req, 'responses')]:
                    with self.subTest(raw=raw, dialect=dialect), self.assertRaises(ValueError): f.prepare(value, dialect)
                encode.assert_not_called()

    def test_http_duplicate_nonfinite_and_deep_request_json_refuse_before_work(self):
        for payload in ['{"model":"owned","model":"other"}', '{"seed":NaN}',
                        '{"extra":' + '[' * 1200 + '0' + ']' * 1200 + '}']:
            with self.subTest(payload=payload[:50]), endpoint('synthetic') as (f, url):
                request = Request(url + '/v1/chat/completions', data=payload.encode(), headers=dict(
                    Authorization='Bearer owned-key', **{'Content-Type': 'application/json'}))
                with self.assertRaises(HTTPError) as error: urlopen(request, timeout=5)
                self.assertEqual(error.exception.code, 400); error.exception.close()
                f.cache.start.assert_not_called()

    def test_history_strict_values_and_choice_none_keep_declarations(self):
        f = frontend()
        history = [dict(role='user', content='CPU synthetic request'), dict(role='assistant', content='',
            tool_calls=[dict(type='function', function=dict(name='write_file', arguments='{"text":"ok"}'))]),
            dict(role='tool', content='synthetic result')]
        req = dict(chat_request(), messages=history, tool_choice='none')
        with patch('serve.cow_compat.Service.encode_prompt', return_value=[1, 2]):
            plan = f.prepare(req, 'chat'); self.assertIsNone(plan['tools'])
        invalid = copy.deepcopy(req); invalid['messages'][1]['tool_calls'][0]['function']['arguments'] = 'not JSON'
        with patch('serve.cow_compat.Service.encode_prompt', side_effect=AssertionError('must not encode')):
            with self.assertRaises(ValueError): f.prepare(invalid, 'chat')

    def test_http_chat_streaming_and_nonstreaming_valid_json_arguments(self):
        args = dict(text='synthetic', payload=dict(flag=True, values=[None, False]))
        for stream in (False, True):
            with self.subTest(stream=stream), endpoint(wire('write_file', args)) as (f, url):
                body = post(url + '/v1/chat/completions', chat_request(stream=stream))
                if stream:
                    rows = [json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
                    tool = next(row['choices'][0]['delta']['tool_calls'][0] for row in rows if 'tool_calls' in row['choices'][0]['delta'])
                    self.assertIn('data: [DONE]', body)
                    self.assertEqual(rows[-1]['choices'][0]['finish_reason'], 'tool_calls')
                else:
                    response = json.loads(body); tool = response['choices'][0]['message']['tool_calls'][0]
                    self.assertEqual(response['choices'][0]['finish_reason'], 'tool_calls')
                self.assertEqual(tool['function']['name'], 'write_file')
                self.assertEqual(json.loads(tool['function']['arguments']), args)
                f.cache.start.assert_called_once(); f.cache.committed.assert_called_once()

    def test_http_responses_streaming_and_nonstreaming(self):
        args = dict(text='synthetic')
        for stream in (False, True):
            with self.subTest(stream=stream), endpoint(wire('write_file', args)) as (f, url):
                req = dict(model='owned', input='CPU synthetic request', tools=[dict(type='function', **SCHEMA)], store=False, stream=stream)
                body = post(url + '/v1/responses', req)
                if stream:
                    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: ')]
                    response = next(e['response'] for e in events if e['type'] == 'response.completed')
                else: response = json.loads(body)
                call = next(x for x in response['output'] if x['type'] == 'function_call')
                self.assertEqual(call['name'], 'write_file'); self.assertEqual(json.loads(call['arguments']), args)

    def test_bad_or_truncated_turn_never_exposes_pending_tool_call(self):
        good = wire('write_file', dict(text='synthetic'))
        for bad in [good + '<function name="unknown"></function>', good + '<function name="write_file">',
                    '<think>\n' + good + '\n</think>', 'Reasoning: <think>\n' + good + '\n</think>',
                    good + '\n<think>\n' + good + '\n</think>']:
            with self.subTest(bad=bad), endpoint(bad) as (f, url):
                body = post(url + '/v1/chat/completions', chat_request(stream=True))
                self.assertIn('event: error', body)
                self.assertNotIn('"tool_calls"', body)
                self.assertNotIn('data: [DONE]', body)
                f.cache.committed.assert_not_called()

    def test_length_and_cancelled_requests_do_not_commit_or_expose_tools(self):
        call = wire('write_file', dict(text='synthetic'))
        for mode in [dict(length=True), dict(cancelled=True)]:
            with self.subTest(mode=mode), endpoint(call, **mode) as (f, url):
                body = post(url + '/v1/chat/completions', chat_request(stream=True))
                self.assertIn('event: error', body)
                self.assertNotIn('"tool_calls"', body)
                f.cache.committed.assert_not_called()

    def test_parallel_false_refuses_second_action_and_none_exposes_no_action(self):
        call = wire('write_file', dict(text='synthetic'))
        with endpoint(call + call) as (f, url):
            with self.assertRaises(HTTPError) as error:
                post(url + '/v1/chat/completions', chat_request(parallel_tool_calls=False))
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        with endpoint(call) as (f, url):
            body = json.loads(post(url + '/v1/chat/completions', chat_request(tool_choice='none')))
            self.assertNotIn('tool_calls', body['choices'][0]['message'])

    def test_owner_binding_mismatch_and_new_cache_identity(self):
        good, old = frontend(), frontend(bound=False)
        self.assertNotEqual(good.cache.identity, old.cache.identity)
        index = PrefixIndex(); index.remember(old.cache.identity, [1, 2], 'old-snapshot')
        self.assertIsNone(index.find(good.cache.identity, [1, 2, 3]))
        index.remember(good.cache.identity, [1, 2], 'new-snapshot')
        self.assertEqual(index.find(good.cache.identity, [1, 2, 3]).snapshot, 'new-snapshot')
        self.assertIsNone(index.find(good.cache.identity, [1, 9, 3]))
        wrong = encoder(); wrong.template.tool_binding = None
        with self.assertRaises(ValueError): Frontend(object(), wrong, dict(model_contract_id='m', execution_profile_id='e'), 'owned')
        missing = encoder(); del missing.tool_binding
        with self.assertRaises(ValueError): Frontend(object(), missing, dict(model_contract_id='m', execution_profile_id='e'), 'owned')

    def test_shared_renderer_implementation_hash_is_the_cache_authority(self):
        first = frontend(); original_read = Path.read_bytes
        binding_receipt = encoder().tool_binding.receipt()
        def changed_codec_source(path):
            data = original_read(path)
            return data + b'\n# simulated changed codec implementation\n' if path.name == 'cow_minicpm_tools.py' else data
        with patch.object(Path, 'read_bytes', changed_codec_source):
            second = frontend()
            self.assertEqual(encoder().tool_binding.receipt(), binding_receipt)
            self.assertNotEqual(first.cache.identity, second.cache.identity)
            index = PrefixIndex(); index.remember(first.cache.identity, [1, 2], 'old-implementation')
            self.assertIsNone(index.find(second.cache.identity, [1, 2, 3]))
            stale = encoder(); stale.renderer_id = first.cache.identity[2]
            profile = dict(model_contract_id='model', execution_profile_id='execution', execution=dict(capacity=10000))
            with self.assertRaises(ValueError): Frontend(object(), stale, profile, 'owned')

    def test_renderer_receipt_binds_serializer_without_changing_default_shape(self):
        tok = SimpleNamespace(tokens=['a', 'b', 'ab'], pre='minicpm5', token_types=[1, 1, 1],
            special_ids={'tokenizer.ggml.bos_token_id': 0}, ranks={('a', 'b'): 0})
        binding = encoder().tool_binding
        args = (tok, MINIMAL_TEMPLATE, dict(enable_thinking=False))
        old = renderer_identity_receipt(*args, model_contract_id='m')
        new = renderer_identity_receipt(*args, model_contract_id='m', tool_binding=binding)
        self.assertNotIn('tool_codec', old); self.assertEqual(new['tool_codec'], binding.receipt())
        self.assertIn('serve/cow_minicpm_tools.py', new['tool_implementation_sha256'])
        self.assertNotEqual(renderer_identity(*args, model_contract_id='m'), renderer_identity(*args, model_contract_id='m', tool_binding=binding))


if __name__ == '__main__': unittest.main()
