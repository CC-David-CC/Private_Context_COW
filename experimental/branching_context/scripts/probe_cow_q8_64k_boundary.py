#!/usr/bin/env python3
"""Q8 tokenizer/template CPU probe with source-code messages split at a real line boundary."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from probe_cow_prompt_parity import SYSTEM, file_sha, sha_bytes, source_body


TARGET_SOURCE_TOKENS = 65536
MIN_REMAINDER_TOKENS = 7000


def source_token_count(tokenizer, text):
    return len(tokenizer.encode(text, parse_special=False, plain=[(0, len(text))]))


def choose_line_boundary(tokenizer, body, source_ids):
    pivot = len(tokenizer.decode(source_ids[:TARGET_SOURCE_TOKENS], errors='strict'))
    candidates = {body.rfind('\n', 0, pivot + 1) + 1, body.find('\n', pivot) + 1}
    scored = []
    for offset in candidates:
        if offset <= 0 or offset >= len(body):
            continue
        prefix, remainder = body[:offset], body[offset:]
        prefix_count = source_token_count(tokenizer, prefix)
        remainder_count = source_token_count(tokenizer, remainder)
        if remainder_count >= MIN_REMAINDER_TOKENS:
            scored.append((abs(prefix_count - TARGET_SOURCE_TOKENS), offset,
                           prefix, remainder, prefix_count, remainder_count))
    if not scored:
        raise RuntimeError('No nearby UTF-8 source line boundary retains an ~8K token remainder')
    _, offset, prefix, remainder, prefix_count, remainder_count = min(scored)
    line_index = body[:offset].count('\n')
    # Locate the same offset in the receipt-verified concatenation, including
    # the deterministic SOURCE FILE framing used by source_body().
    return prefix, remainder, offset, line_index, prefix_count, remainder_count


def source_segments(context):
    segments = []
    cursor = 0
    for receipt in context['source_receipts']:
        path = Path(receipt['path'])
        raw = path.read_bytes()
        if len(raw) != receipt['bytes'] or sha_bytes(raw) != receipt['sha256']:
            raise RuntimeError('Source receipt mismatch: ' + str(path))
        relative = path.relative_to(Path(receipt['source_root'])).as_posix()
        content = raw.decode('utf-8', errors='strict')
        heading = f'\nSOURCE FILE: {relative}\n'
        framing = f'\nEND SOURCE FILE\n'
        start = cursor + len(heading)
        segments.append({'path': relative, 'text': content, 'start': start,
                         'end': start + len(content)})
        cursor = start + len(content) + len(framing)
    return segments


def boundary_provenance(body, offset, segments):
    for segment in segments:
        if segment['start'] <= offset <= segment['end']:
            in_file = offset - segment['start']
            content = segment['text']
            line_number = content[:in_file].count('\n') + 1
            line_start = content.rfind('\n', 0, in_file) + 1
            line_end = content.find('\n', in_file)
            if line_end < 0:
                line_end = len(content)
            return {'source_file': segment['path'], 'source_line_1based': line_number,
                    'line_text': content[line_start:line_end],
                    'source_file_split_body_sha256': sha_bytes(content.encode()),
                    'source_file_split_body_utf8_bytes': len(content.encode()),
                    'offset_within_source_file_characters': in_file}
    raise RuntimeError('Chosen split offset does not fall inside a source file body')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture_raw = args.fixture.read_bytes()
    fixture = json.loads(fixture_raw)
    profile_path = Path(fixture['profile_path'])
    profile_raw = profile_path.read_bytes()
    profile = json.loads(profile_raw)
    model_path = Path(profile['model_path'])
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / 'tools'))
    sys.path.insert(0, str(repo_root))
    from gguf_reader import GGUFFile
    from strata_tokenizer import Tokenizer
    from serve.cow_prompt_encoder import CowPromptEncoder
    from serve.frontend import ChatTemplate
    from serve.server import MockEngine, Service
    from serve.cow_prefix import exact_append_suffix
    import regex
    import platform

    metadata = GGUFFile(model_path).metadata
    template_source = metadata.get('tokenizer.chat_template')
    if not isinstance(template_source, str) or not template_source:
        raise RuntimeError('Q8 GGUF does not contain scalar tokenizer.chat_template')
    tokenizer = Tokenizer.from_gguf(model_path)
    with tempfile.TemporaryDirectory(prefix='cow-q8-boundary-template-') as temp:
        template_path = Path(temp) / 'q8-chat-template.jinja'
        template_path.write_text(template_source, encoding='utf-8', newline='\n')
        template = ChatTemplate(template_path)
        template_kwargs = {'enable_thinking': False}
        prompt_encoder = CowPromptEncoder(tokenizer, template, template_kwargs)
        ordinary = Service(MockEngine(tokenizer, 'cpu-parity-only'), tokenizer, template)
        contexts = []
        for context in fixture['contexts']:
            body, full_count, all_source_ids, receipts = source_body(context, 73728, tokenizer)
            base_source, remaining_source, split_offset, line_index, base_source_count, remaining_count = \
                choose_line_boundary(tokenizer, body, tokenizer.encode(body, parse_special=False,
                    plain=[(0, len(body))]))
            boundary = boundary_provenance(body, split_offset, source_segments(context))
            root_messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': base_source}]
            base_text = prompt_encoder.render(root_messages, add_generation_prompt=False)
            base_ids = prompt_encoder.encode_rendered(root_messages, base_text, False)
            ordinary_base = ordinary.encode_prompt(root_messages, None,
                {**template_kwargs, 'add_generation_prompt': False})
            if base_ids != ordinary_base:
                raise AssertionError('Base IDs differ from ordinary Service')

            branches = {}
            full_messages = {}
            for label in ('A', 'B'):
                appended_user = remaining_source + '\nQUESTION:\n' + context['question_' + label]
                messages = root_messages + [{'role': 'user', 'content': appended_user}]
                rendered = prompt_encoder.render(messages, add_generation_prompt=True)
                ids = prompt_encoder.encode_rendered(messages, rendered, True)
                ordinary_ids = ordinary.encode_prompt(messages, None,
                    {**template_kwargs, 'add_generation_prompt': True})
                if ids != ordinary_ids:
                    raise AssertionError(f'{label} IDs differ from ordinary Service')
                suffix = exact_append_suffix(base_ids, ids)
                branches[label] = {
                    'question': context['question_' + label],
                    'appended_user_message': appended_user,
                    'rendered_sha256': sha_bytes(rendered.encode()),
                    'rendered_utf8_bytes': len(rendered.encode()),
                    'full_token_count': len(ids),
                    'full_token_ids': ids,
                    'full_token_ids_sha256': sha_bytes(json.dumps(ids, separators=(',', ':')).encode()),
                    'suffix_count': len(suffix),
                    'suffix_token_ids': suffix,
                    'ordinary_service_ids_match': True,
                    'exact_base_prefix': True,
                }
                full_messages[label] = messages

            contexts.append({
                'name': context['name'],
                'source_receipts': receipts,
                'source_token_count': full_count,
                'source_corpus_token_count': len(all_source_ids),
                'split': {
                    'method': 'nearest preceding/following newline around source token 65536; nearest with >=7000 source tokens remaining',
                    'body_character_offset': split_offset,
                    'line_index_1based_in_concatenated_source': line_index,
                    'source_file_line_boundary': boundary,
                    'base_source_token_count': base_source_count,
                    'remaining_source_token_count': remaining_count,
                    'base_source_sha256': sha_bytes(base_source.encode()),
                    'remaining_source_sha256': sha_bytes(remaining_source.encode()),
                    'source_body_sha256': sha_bytes(body.encode()),
                    'source_body_utf8_bytes': len(body.encode()),
                },
                'root_messages': root_messages,
                'A_messages': full_messages['A'],
                'B_messages': full_messages['B'],
                'base_rendered_sha256': sha_bytes(base_text.encode()),
                'base_rendered_utf8_bytes': len(base_text.encode()),
                'base_token_count': len(base_ids),
                'base_token_ids': base_ids,
                'base_token_ids_sha256': sha_bytes(json.dumps(base_ids, separators=(',', ':')).encode()),
                'branches': branches,
            })

    value = {
        'schema': 'cow-q8-64k-boundary-prompt-v1',
        'status': 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY',
        'scope': 'Actual Q8 GGUF tokenizer/template and source-receipt code text; no model inference or GPU execution.',
        'fixture_path': str(args.fixture),
        'fixture_sha256': sha_bytes(fixture_raw),
        'fixture_variant': fixture.get('variant'),
        'profile_path': str(profile_path),
        'profile_sha256': sha_bytes(profile_raw),
        'model_metadata_path': str(model_path),
        'model_metadata_file_bytes': model_path.stat().st_size,
        'model_metadata_file_sha256': file_sha(model_path),
        'model_weights_sha256_profile_identity': profile.get('model_weights_sha256'),
        'tokenizer_vocab_count': len(tokenizer.tokens),
        'tokenizer_merge_count': len(tokenizer.ranks),
        'tokenizer_pre': tokenizer.pre,
        'tokenizer_source_sha256': file_sha(repo_root / 'tools' / 'strata_tokenizer.py'),
        'gguf_reader_source_sha256': file_sha(repo_root / 'tools' / 'gguf_reader.py'),
        'gguf_template_sha256': sha_bytes(template_source.encode()),
        'frontend_source_sha256': file_sha(repo_root / 'serve' / 'frontend.py'),
        'service_source_sha256': file_sha(repo_root / 'serve' / 'server.py'),
        'cow_prompt_encoder_source_sha256': file_sha(repo_root / 'serve' / 'cow_prompt_encoder.py'),
        'probe_source_path': str(Path(__file__).resolve()),
        'probe_source_sha256': file_sha(Path(__file__).resolve()),
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'regex_version': regex.__version__,
        'regex_module_path': str(Path(regex.__file__).resolve()),
        'template_kwargs': template_kwargs,
        'contexts': contexts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': value['status'], 'output': str(args.output),
                      'sha256': file_sha(args.output),
                      'contexts': [{'name': item['name'], 'base_tokens': item['base_token_count'],
                                    'source_base_tokens': item['split']['base_source_token_count'],
                                    'source_added_tokens': item['split']['remaining_source_token_count'],
                                    'A_suffix': item['branches']['A']['suffix_count'],
                                    'B_suffix': item['branches']['B']['suffix_count']}
                                   for item in contexts]}, indent=2))


if __name__ == '__main__':
    main()
