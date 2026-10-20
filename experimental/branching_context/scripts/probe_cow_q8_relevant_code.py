#!/usr/bin/env python3
"""CPU-only Q8 prompt parity over explicitly prioritized implementation files."""
import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

from probe_cow_prompt_parity import SYSTEM, file_sha, sha_bytes

SOURCE_TOKENS = 73728
BASE_TARGET = 65536
MIN_SUFFIX_SOURCE_TOKENS = 7000


def token_ids(tokenizer, text):
    return tokenizer.encode(text, parse_special=False, plain=[(0, len(text))])


def section_bundle(context):
    sections, receipts = [], []
    cursor = 0
    root = Path(context['source_root'])
    for relative in context['source_files']:
        path = root / relative
        raw = path.read_bytes()
        text = raw.decode('utf-8', errors='strict')
        heading, tail = f'\nSOURCE FILE: {relative}\n', '\nEND SOURCE FILE\n'
        start = cursor + len(heading)
        sections.append({'path': relative, 'text': text, 'start': start,
                         'end': start + len(text), 'sha256': sha_bytes(raw),
                         'bytes': len(raw)})
        receipts.append({'path': str(path), 'relative_path': relative,
                         'source_root': str(root), 'bytes': len(raw),
                         'sha256': sha_bytes(raw)})
        cursor = start + len(text) + len(tail)
    body = ''.join(f"\nSOURCE FILE: {s['path']}\n{s['text']}\nEND SOURCE FILE\n" for s in sections)
    return body, sections, receipts


def source_boundary(body, offset, sections):
    for section in sections:
        if section['start'] <= offset <= section['end']:
            local = offset - section['start']
            content = section['text']
            line = content[:local].count('\n') + 1
            line_start = content.rfind('\n', 0, local) + 1
            line_end = content.find('\n', local)
            if line_end < 0:
                line_end = len(content)
            return {'file': section['path'], 'line_1based': line,
                    'line_text': content[line_start:line_end],
                    'file_sha256': section['sha256'],
                    'file_utf8_bytes': section['bytes'],
                    'character_offset_in_file': local}
    raise RuntimeError('Split point is outside an included source file')


def consumed_file_coverage(consumed, sections):
    result = []
    limit = len(consumed)
    for section in sections:
        chars = max(0, min(section['end'], limit) - section['start'])
        text = section['text']
        partial_line = text[:chars].count('\n') + (1 if chars else 0)
        result.append({'file': section['path'], 'file_sha256': section['sha256'],
                       'file_utf8_bytes': section['bytes'],
                       'consumed_characters': chars,
                       'consumed_lines_1based_through': partial_line,
                       'fully_consumed': chars == len(text),
                       'truncated_inside_file': 0 < chars < len(text),
                       'not_consumed': chars == 0})
    return result


def split_near(tokenizer, body):
    ids = token_ids(tokenizer, body)
    if len(ids) < SOURCE_TOKENS:
        raise RuntimeError(f'Only {len(ids)} prioritized source tokens available; need {SOURCE_TOKENS}')
    end = SOURCE_TOKENS
    while end:
        try:
            consumed = tokenizer.decode(ids[:end], errors='strict')
            break
        except UnicodeDecodeError:
            end -= 1
    if end < SOURCE_TOKENS:
        raise RuntimeError('Unable to decode source token prefix at a UTF-8 boundary')
    pivot = len(tokenizer.decode(ids[:BASE_TARGET], errors='strict'))
    candidates = {consumed.rfind('\n', 0, pivot + 1) + 1, consumed.find('\n', pivot) + 1}
    scored = []
    for offset in candidates:
        if not 0 < offset < len(consumed):
            continue
        base, suffix = consumed[:offset], consumed[offset:]
        base_count, suffix_count = len(token_ids(tokenizer, base)), len(token_ids(tokenizer, suffix))
        if suffix_count >= MIN_SUFFIX_SOURCE_TOKENS:
            scored.append((abs(base_count - BASE_TARGET), offset, base, suffix, base_count, suffix_count))
    if not scored:
        raise RuntimeError('No source line boundary near 64K with at least 7K actual code remaining')
    _, offset, base, suffix, base_count, suffix_count = min(scored)
    return consumed, end, ids, base, suffix, offset, base_count, suffix_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True, help='Q8 fixture, used for pinned GGUF profile')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
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

    metadata = GGUFFile(model_path).metadata
    template_source = metadata.get('tokenizer.chat_template')
    if not isinstance(template_source, str) or not template_source:
        raise RuntimeError('Pinned Q8 GGUF has no scalar chat template')
    tokenizer = Tokenizer.from_gguf(model_path)
    with tempfile.TemporaryDirectory(prefix='cow-q8-relevant-template-') as temp:
        template_path = Path(temp) / 'template.jinja'
        template_path.write_text(template_source, encoding='utf-8', newline='\n')
        template = ChatTemplate(template_path)
        kwargs = {'enable_thinking': False}
        encoder = CowPromptEncoder(tokenizer, template, template_kwargs=kwargs)
        service = Service(MockEngine(tokenizer, 'cpu-parity-only'), tokenizer, template)
        outputs = []
        for context in manifest['contexts']:
            body, sections, receipts = section_bundle(context)
            consumed, consumed_token_count, corpus_ids, base_source, added_source, offset, base_source_count, added_count = split_near(tokenizer, body)
            markers = context['required_source_markers']
            missing = [marker for marker in markers if marker.casefold() not in consumed.casefold()]
            if missing:
                raise RuntimeError(f"{context['name']}: required implementation markers not consumed: {missing}")
            boundary = source_boundary(consumed, offset, sections)
            root_messages = [{'role': 'system', 'content': SYSTEM},
                             {'role': 'user', 'content': base_source}]
            rendered_base = encoder.render(root_messages, add_generation_prompt=False)
            base_ids = encoder.encode_rendered(root_messages, rendered_base, False)
            if base_ids != service.encode_prompt(root_messages, None,
                    {**kwargs, 'add_generation_prompt': False}):
                raise AssertionError(f"{context['name']}: base prompt differs from Service")
            branches = {}
            full_messages = {}
            for label in ('A', 'B'):
                question = context['question_' + label]
                appended = added_source + '\nQUESTION:\n' + question
                messages = root_messages + [{'role': 'user', 'content': appended}]
                rendered = encoder.render(messages, add_generation_prompt=True)
                ids = encoder.encode_rendered(messages, rendered, True)
                if ids != service.encode_prompt(messages, None,
                        {**kwargs, 'add_generation_prompt': True}):
                    raise AssertionError(f"{context['name']} {label}: request IDs differ from Service")
                suffix = exact_append_suffix(base_ids, ids)
                branches[label] = {
                    'question': question,
                    'appended_user_message': appended,
                    'rendered_sha256': sha_bytes(rendered.encode()),
                    'full_token_count': len(ids),
                    'full_token_ids': ids,
                    'full_token_ids_sha256': sha_bytes(json.dumps(ids, separators=(',', ':')).encode()),
                    'suffix_count': len(suffix), 'suffix_token_ids': suffix,
                    'ordinary_service_ids_match': True, 'exact_base_prefix': True,
                    'required_markers_present_in_consumed_code': True,
                }
                full_messages[label] = messages
            outputs.append({
                'name': context['name'], 'source_root': context['source_root'],
                'included_files_in_order': receipts,
                'consumed_file_coverage': consumed_file_coverage(consumed, sections),
                'required_source_markers': markers,
                'source_bundle_sha256': sha_bytes(body.encode()),
                'source_bundle_token_count': len(corpus_ids),
                'consumed_source_token_count': consumed_token_count,
                'consumed_source_sha256': sha_bytes(consumed.encode()),
                'split': {'method': 'nearest real newline around 65,536 Q8 source tokens; no padding or repeated source',
                          'source_file_line_boundary': boundary,
                          'base_source_tokens': base_source_count,
                          'added_source_tokens': added_count,
                          'base_source_sha256': sha_bytes(base_source.encode()),
                          'added_source_sha256': sha_bytes(added_source.encode())},
                'root_messages': root_messages,
                'A_messages': full_messages['A'], 'B_messages': full_messages['B'],
                'base_token_count': len(base_ids),
                'base_token_ids': base_ids,
                'base_token_ids_sha256': sha_bytes(json.dumps(base_ids, separators=(',', ':')).encode()),
                'branches': branches,
            })
    result = {
        'schema': 'cow-q8-relevant-code-prompt-v1',
        'status': 'PASS_CPU_TOKENIZER_TEMPLATE_SOURCE_RELEVANCE',
        'scope': 'Actual Q8 GGUF tokenizer/template over prioritized implementation source receipts; no inference or GPU execution.',
        'fixture_path': str(args.fixture), 'fixture_sha256': sha_bytes(fixture_raw),
        'manifest_path': str(args.manifest), 'manifest_sha256': sha_bytes(manifest_raw),
        'profile_path': str(profile_path), 'profile_sha256': sha_bytes(profile_raw),
        'model_metadata_path': str(model_path), 'model_metadata_sha256': file_sha(model_path),
        'model_weights_sha256_profile_identity': profile.get('model_weights_sha256'),
        'tokenizer_vocab_count': len(tokenizer.tokens), 'tokenizer_merge_count': len(tokenizer.ranks),
        'tokenizer_pre': tokenizer.pre, 'tokenizer_sha256': file_sha(repo_root / 'tools' / 'strata_tokenizer.py'),
        'template_sha256': sha_bytes(template_source.encode()),
        'frontend_sha256': file_sha(repo_root / 'serve' / 'frontend.py'),
        'service_sha256': file_sha(repo_root / 'serve' / 'server.py'),
        'cow_prompt_encoder_sha256': file_sha(repo_root / 'serve' / 'cow_prompt_encoder.py'),
        'probe_sha256': file_sha(Path(__file__).resolve()), 'python': platform.python_version(),
        'python_executable': sys.executable, 'regex_version': regex.__version__,
        'template_kwargs': kwargs, 'contexts': outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'sha256': file_sha(args.output),
                      'contexts': [{'name': c['name'], 'base': c['base_token_count'],
                                    'source_base': c['split']['base_source_tokens'],
                                    'source_added': c['split']['added_source_tokens'],
                                    'A_tokens': c['branches']['A']['full_token_count'],
                                    'B_tokens': c['branches']['B']['full_token_count']}
                                   for c in outputs]}, indent=2))


if __name__ == '__main__':
    main()
