#!/usr/bin/env python3
"""Build CPU-only Q8 token receipts for the real browser file-context framing.

This helper reads tokenizer/template metadata and the existing 8K source fixture;
it never reads or hashes model tensor bytes and performs no model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def file_block(name: str, text: str) -> str:
    """Match `app.js:fileBlock` byte-for-byte for a text file attachment."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("attachment name must be a plain filename")
    if not isinstance(text, str):
        raise ValueError("attachment body must be text")
    longest = max([2, *(len(run) for run in re.findall(r"`+", text))])
    fence = "`" * (longest + 1)
    return f"File: {name}\n{fence}\n{text}\n{fence}"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_reusable_context_set(rows: list[dict]) -> dict:
    """Fail closed on a three-source browser run with mismatched prompt pins."""
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError('Reusable-context set requires exactly three source contexts')
    names, bodies, attachments = set(), set(), set()
    profiles, renderers = set(), set()
    counts, suffix_counts = [], []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('fixture'), dict):
            raise ValueError('Reusable-context set row lacks its fixture receipt')
        fixture = row['fixture']
        name = fixture.get('name')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Reusable-context set requires unique source names')
        names.add(name)
        if fixture.get('schema') != 'cow-browser-context-fixture-v1' or fixture.get('status') != 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY':
            raise ValueError(f'{name}: source fixture lacks CPU tokenizer/template parity')
        body = fixture.get('source_body_sha256')
        if not isinstance(body, str) or len(body) != 64 or body in bodies:
            raise ValueError('Three contexts must contain distinct source bodies')
        bodies.add(body)
        receipts = fixture.get('source_receipts')
        if not isinstance(receipts, list) or not receipts:
            raise ValueError(f'{name}: source provenance is required')
        profiles.add(fixture.get('execution_profile_id'))
        renderers.add(fixture.get('renderer_id'))
        root_tokens = fixture.get('root_token_count')
        if type(root_tokens) is not int or not 6000 <= root_tokens <= 10000:
            raise ValueError(f'{name}: root token count is outside the bounded ~8K range')
        separator_tokens = fixture.get('append_user_separator_token_count')
        if type(separator_tokens) is not int or separator_tokens <= 0:
            raise ValueError(f'{name}: append-user separator token receipt is missing')
        counts.append(root_tokens)
        branches = fixture.get('branches')
        if not isinstance(branches, dict) or set(branches) != {'A', 'B'}:
            raise ValueError(f'{name}: A and B question receipts are required')
        current_suffixes = []
        questions = set()
        for label in ('A', 'B'):
            branch = branches[label]
            question, suffix = branch.get('question'), branch.get('suffix_token_count')
            if not isinstance(question, str) or not question.strip() or question in questions:
                raise ValueError(f'{name}: A and B require different nonempty questions')
            questions.add(question)
            if type(suffix) is not int or not 400 <= suffix <= 700:
                raise ValueError(f'{name}: {label} question token count is outside the ~512 range')
            stopped_suffix = branch.get('append_user_suffix_token_count')
            if type(stopped_suffix) is not int or stopped_suffix != suffix + separator_tokens:
                raise ValueError(f'{name}: {label} stopped-turn suffix is not tokenizer-bound')
            current_suffixes.append(suffix)
        suffix_counts.append(current_suffixes)
        attachment = row.get('attachment_name')
        file_name = row.get('context_file')
        if (not isinstance(attachment, str) or not attachment or Path(attachment).name != attachment
                or attachment in attachments):
            raise ValueError('Reusable-context set requires unique plain attachment names')
        attachments.add(attachment)
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f'{name}: context file path is required')
    if len(profiles) != 1 or None in profiles:
        raise ValueError('Reusable-context set execution profile IDs differ')
    if len(renderers) != 1 or None in renderers:
        raise ValueError('Reusable-context set renderer identities differ')
    return {'context_names': [row['fixture']['name'] for row in rows],
            'root_token_counts': counts, 'question_suffix_counts': suffix_counts,
            'execution_profile_id': next(iter(profiles)), 'renderer_id': next(iter(renderers)),
            'distinct_source_bodies': len(bodies)}


def load_reusable_context_set(manifest_path: Path, expected_profile_id: str | None = None) -> list[dict]:
    """Load and verify all fixture/file receipts before any browser request."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'cow-browser-context-set-v1'
            or manifest.get('status') != 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY'):
        raise ValueError('Reusable-context set lacks its tokenizer/template fixture status')
    rows = []
    for entry in manifest.get('contexts', []):
        base=manifest_path.parent.resolve()
        fixture_path = (base / entry['fixture_file']).resolve()
        context_path = (base / entry['context_file']).resolve()
        if not fixture_path.is_relative_to(base) or not context_path.is_relative_to(base):
            raise ValueError('Context-set manifest paths must remain inside its fixture directory')
        fixture_raw = fixture_path.read_bytes()
        if sha(fixture_raw) != entry.get('fixture_sha256'):
            raise ValueError(f"{entry.get('name')}: fixture file digest mismatch")
        fixture = json.loads(fixture_raw)
        context_bytes = context_path.read_bytes()
        if sha(context_bytes) != fixture.get('source_body_sha256'):
            raise ValueError(f"{entry.get('name')}: source context file digest mismatch")
        if (entry.get('name') != fixture.get('name')
                or entry.get('source_body_sha256') != fixture.get('source_body_sha256')
                or entry.get('root_token_count') != fixture.get('root_token_count')):
            raise ValueError(f"{entry.get('name')}: manifest and fixture identity differ")
        rows.append({'fixture': fixture, 'fixture_file': str(fixture_path),
                     'fixture_sha256': entry.get('fixture_sha256'),
                     'context_file': str(context_path), 'attachment_name': entry.get('attachment_name'),
                     'context_bytes': context_bytes})
    summary = validate_reusable_context_set(rows)
    if manifest.get('execution_profile_id') != summary['execution_profile_id']:
        raise ValueError('Context-set manifest execution profile differs from its fixtures')
    if manifest.get('renderer_id') != summary['renderer_id']:
        raise ValueError('Context-set manifest renderer differs from its fixtures')
    if expected_profile_id is not None and expected_profile_id != summary['execution_profile_id']:
        raise ValueError('Server execution profile ID differs from the context-set fixtures')
    return rows


def build(repo_root: Path, profile_path: Path, source_fixture: Path,
          attachment_name: str, output_dir: Path, *, context_index: int = 0) -> dict:
    repo_root = repo_root.resolve()
    profile_path = profile_path.resolve()
    source_fixture = source_fixture.resolve()
    profile_raw = profile_path.read_bytes()
    profile = json.loads(profile_raw)
    fixture_raw = source_fixture.read_bytes()
    source = json.loads(fixture_raw)
    contexts = source.get('contexts')
    if not isinstance(contexts, list) or not 0 <= context_index < len(contexts):
        raise ValueError('Source fixture context index is unavailable')
    context = contexts[context_index]
    source_body = context['root_messages'][-1]['content']
    if not isinstance(source_body, str) or not source_body:
        raise ValueError('source fixture does not contain a nonempty source body')
    messages_a = context['A_messages']
    messages_b = context['B_messages']
    question_a, question_b = messages_a[-1]['content'], messages_b[-1]['content']
    if messages_a[:-1] != context['root_messages'] or messages_b[:-1] != context['root_messages']:
        raise ValueError('source fixture A/B questions do not share the source root')
    if question_a == question_b:
        raise ValueError('A/B questions must be different')

    sys.path[:0] = [str(repo_root / 'tools'), str(repo_root)]
    from gguf_reader import GGUFFile
    from strata_tokenizer import Tokenizer
    from serve.cow_prompt_encoder import CowPromptEncoder
    from serve.cow_renderer import renderer_identity
    from serve.frontend import ChatTemplate
    from serve.server import MockEngine, Service

    model_path = Path(profile['model_path'])
    metadata = GGUFFile(model_path).metadata
    template_source = metadata.get('tokenizer.chat_template')
    if not isinstance(template_source, str) or not template_source:
        raise ValueError('pinned GGUF has no scalar chat template')
    tokenizer = Tokenizer.from_gguf(model_path)
    wrapped_source = file_block(attachment_name, source_body)
    root_messages = [{'role': 'user', 'content': wrapped_source}]
    template_kwargs = profile['execution'].get('text_template_kwargs')
    if template_kwargs != {'enable_thinking': False}:
        raise ValueError('fixture requires the pinned non-thinking Q8 text template')
    output_dir = output_dir.resolve()
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    template_path = output_dir / 'chat_template.jinja'
    template_path.write_text(template_source, encoding='utf-8', newline='\n')
    encoder = CowPromptEncoder(tokenizer, ChatTemplate(template_path), template_kwargs)
    service = Service(MockEngine(tokenizer, 'cpu-browser-tokenizer-fixture'), tokenizer,
                      ChatTemplate(template_path))
    renderer_id = renderer_identity(tokenizer, template_source, template_kwargs,
                                    model_contract_id=profile['model_contract_id'])

    def encode(messages, add_generation_prompt):
        rendered = encoder.render(messages, add_generation_prompt=add_generation_prompt)
        ids = tuple(encoder.encode_rendered(messages, rendered, add_generation_prompt))
        ordinary = tuple(service.encode_prompt(messages, None,
            {**template_kwargs, 'add_generation_prompt': add_generation_prompt}))
        if ids != ordinary:
            raise AssertionError('COW text rendering differs from ordinary Service prompt encoding')
        return rendered, ids

    root_rendered, root_ids = encode(root_messages, False)
    end = tuple(tokenizer.encode('<|im_end|>', parse_special=True))
    newline = tuple(tokenizer.encode('\n', parse_special=False))
    if len(end) != 1 or not newline or root_ids[-len(end + newline):] != end + newline:
        raise AssertionError('context-only root did not preserve the exact closed-message boundary')
    branches = {}
    for label, question in (('A', question_a), ('B', question_b)):
        full_messages = root_messages + [{'role': 'user', 'content': question}]
        rendered, full_ids = encode(full_messages, True)
        suffix = tuple(encoder.append_initial_user_suffix(root_ids,
            {'status': 'idle', 'policy': {'eos_tokens': profile['execution']['eos_tokens']}}, question))
        if tuple(full_ids[:len(root_ids)]) != root_ids or full_ids[len(root_ids):] != suffix:
            raise AssertionError(f'{label}: exact UI root prefix or appended suffix differs')
        stopped_suffix = tuple(encoder.append_user_suffix((end[0],),
            {'status':'stopped','policy':{'eos_tokens':profile['execution']['eos_tokens']}}, question))
        if stopped_suffix != newline + suffix:
            raise AssertionError(f'{label}: stopped-turn append-user encoder differs from fixture question frame')
        if not 400 <= len(suffix) <= 700:
            raise ValueError(f'{label}: expected roughly 512 question tokens, got {len(suffix)}')
        branches[label] = {
            'question': question,
            'suffix_token_count': len(suffix),
            'suffix_token_ids': list(suffix),
            'suffix_token_ids_sha256': sha(json.dumps(list(suffix), separators=(',', ':')).encode()),
            'append_user_suffix_token_count': len(stopped_suffix),
            'append_user_suffix_token_ids_sha256': sha(json.dumps(
                list(stopped_suffix), separators=(',', ':')).encode()),
            'full_token_count': len(full_ids),
            'full_token_ids_sha256': sha(json.dumps(list(full_ids), separators=(',', ':')).encode()),
            'rendered_sha256': sha(rendered.encode('utf-8')),
            'ordinary_service_ids_match': True,
            'exact_root_prefix': True,
        }

    context_bytes = source_body.encode('utf-8')
    attachment_bytes = wrapped_source.encode('utf-8')
    (output_dir / attachment_name).write_bytes(context_bytes)
    record = {
        'schema': 'cow-browser-context-fixture-v1',
        'status': 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY',
        'scope': 'real UI file framing and pinned tokenizer/template parity only; no model-weight hashing or inference',
        'profile_sha256': sha(profile_raw),
        'execution_profile_id': profile['execution_profile_id'],
        'model_contract_id': profile['model_contract_id'],
        'model_weights_sha256_profile_identity': profile['model_weights_sha256'],
        'model_path': str(model_path),
        'renderer_id': renderer_id,
        'template_sha256': sha(template_source.encode('utf-8')),
        'template_kwargs': template_kwargs,
        'attachment_name': attachment_name,
        'source_fixture_sha256': sha(fixture_raw),
        'source_body_sha256': sha(context_bytes),
        'source_body_bytes': len(context_bytes),
        'ui_attachment_sha256': sha(attachment_bytes),
        'ui_attachment_bytes': len(attachment_bytes),
        'root_messages': root_messages,
        'root_token_count': len(root_ids),
        'append_user_separator_token_count': len(newline),
        'root_token_ids': list(root_ids),
        'root_token_ids_sha256': sha(json.dumps(list(root_ids), separators=(',', ':')).encode()),
        'root_rendered_sha256': sha(root_rendered.encode('utf-8')),
        'branches': branches,
        'name': context.get('name', f'context-{context_index}'),
        'source_receipts': context.get('source_receipts', source.get('source_receipts',
            context.get('included_files_in_order', context.get('source_excerpts', [])))),
        'ui_source_hashes': {relative: sha((repo_root / relative).read_bytes()) for relative in (
            'serve/web/app.js', 'serve/web/workspace.js', 'serve/web/workspace-graph.js')},
        'artifacts': {'context_file': attachment_name, 'template_file': template_path.name},
    }
    raw = (json.dumps(record, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    (output_dir / 'fixture.json').write_bytes(raw)
    return record


def build_all(repo_root: Path, profile_path: Path, source_fixture: Path,
              output_dir: Path) -> dict:
    """Build tokenizer/template receipts for every distinct source context."""
    source = json.loads(source_fixture.read_text(encoding='utf-8'))
    contexts = source.get('contexts')
    if not isinstance(contexts, list) or len(contexts) != 3:
        raise ValueError('All-context build requires exactly three source contexts')
    output_dir = output_dir.resolve()
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows, records = [], []
    model_contract_ids = set()
    for index, context in enumerate(contexts):
        name = context.get('name') if isinstance(context, dict) else None
        if not isinstance(name, str) or not name.strip() or Path(name).name != name:
            raise ValueError(f'Context {index} requires a plain unique name')
        slug = f'{index + 1:02d}-{name}'
        attachment_name = slug + '.txt'
        case_dir = output_dir / slug
        record = build(repo_root, profile_path, source_fixture, attachment_name,
                       case_dir, context_index=index)
        model_contract_ids.add(record['model_contract_id'])
        rows.append({'fixture': record, 'context_file': str(case_dir / attachment_name),
                     'attachment_name': attachment_name, 'context_bytes': (case_dir / attachment_name).read_bytes()})
        records.append({'name': record['name'], 'fixture_file': str(Path(slug) / 'fixture.json'),
                        'context_file': str(Path(slug) / attachment_name),
                        'attachment_name': attachment_name,
                        'fixture_sha256': sha((case_dir / 'fixture.json').read_bytes()),
                        'source_body_sha256': record['source_body_sha256'],
                        'root_token_count': record['root_token_count'],
                        'question_suffix_counts': {key: branch['suffix_token_count']
                                                   for key, branch in record['branches'].items()}})
    summary = validate_reusable_context_set(rows)
    if len(model_contract_ids) != 1:
        raise ValueError('All source contexts must use the same model contract')
    manifest = {'schema': 'cow-browser-context-set-v1',
                'status': 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY',
                'scope': 'three distinct real-source UI attachment/tokenizer fixtures; no inference',
                'source_fixture_sha256': sha(source_fixture.read_bytes()),
                'profile_sha256': sha(profile_path.read_bytes()),
                'execution_profile_id': summary['execution_profile_id'],
                'model_contract_id': next(iter(model_contract_ids)),
                'renderer_id': summary['renderer_id'], 'contexts': records,
                'validation': summary}
    path = output_dir / 'fixture-set.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8', newline='\n')
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--source-fixture', type=Path, required=True)
    parser.add_argument('--attachment-name')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--all-contexts', action='store_true',
                        help='build all three distinct contexts into a fixture-set directory')
    args = parser.parse_args()
    if args.all_contexts:
        manifest = build_all(args.repo_root, args.profile, args.source_fixture, args.output_dir)
        print(json.dumps({'status': manifest['status'], 'fixture_set': str(args.output_dir / 'fixture-set.json'),
            'contexts': manifest['validation']['context_names'],
            'root_tokens': manifest['validation']['root_token_counts'],
            'question_suffix_tokens': manifest['validation']['question_suffix_counts'],
            'model_tensor_bytes_read_or_hashed': False}))
        return 0
    if not args.attachment_name:
        parser.error('--attachment-name is required unless --all-contexts is used')
    record = build(args.repo_root, args.profile, args.source_fixture,
                   args.attachment_name, args.output_dir)
    print(json.dumps({'status': record['status'], 'fixture': str(args.output_dir / 'fixture.json'),
        'root_tokens': record['root_token_count'],
        'question_suffix_tokens': {key: row['suffix_token_count'] for key, row in record['branches'].items()},
        'model_tensor_bytes_read_or_hashed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
