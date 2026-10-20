#!/usr/bin/env python3
"""Read-only CPU probe of a Q8 bounded reply followed by a rendered next turn."""
import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def first_diff(left, right):
    upto = min(len(left), len(right))
    for i in range(upto):
        if left[i] != right[i]:
            return i
    return upto if len(left) != len(right) else None


def summarize_comparison(expected, rendered, decode, label):
    mismatch = first_diff(expected, rendered)
    return {
        'label': label,
        'expected_tokens': len(expected),
        'rendered_tokens': len(rendered),
        'expected_is_exact_rendered_prefix': len(expected) <= len(rendered) and rendered[:len(expected)] == expected,
        'first_mismatch_index_zero_based': mismatch,
        'expected_tail_before_mismatch': decode(expected[max(0, (mismatch or 0)-8):(mismatch or 0)+8]),
        'rendered_tail_before_mismatch': decode(rendered[max(0, (mismatch or 0)-8):(mismatch or 0)+8]),
        'expected_ids_near_mismatch': expected[max(0, (mismatch or 0)-8):(mismatch or 0)+8],
        'rendered_ids_near_mismatch': rendered[max(0, (mismatch or 0)-8):(mismatch or 0)+8],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-evidence', type=Path, required=True)
    parser.add_argument('--resume-raw', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture_raw = args.fixture_evidence.read_bytes()
    fixture = json.loads(fixture_raw)
    resume_raw = args.resume_raw.read_bytes()
    resume = json.loads(resume_raw)
    profile_raw = args.profile.read_bytes()
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
    import regex

    tokenizer = Tokenizer.from_gguf(model_path)
    template_source = GGUFFile(model_path).metadata['tokenizer.chat_template']
    with tempfile.TemporaryDirectory(prefix='cow-next-turn-q8-') as temp:
        template_path = Path(temp) / 'q8-template.jinja'
        template_path.write_text(template_source, encoding='utf-8', newline='\n')
        template = ChatTemplate(template_path)
        template_kwargs = {'enable_thinking': False}
        encoder = CowPromptEncoder(tokenizer, template, template_kwargs)
        ordinary = Service(MockEngine(tokenizer, 'cpu-next-turn-parity-only'), tokenizer, template)
        context = fixture['contexts'][0]
        branch = context['branches']['A']
        prompt_ids = branch['full_token_ids']
        request = next(item for item in resume['requests'] if item['name'] == 'A')
        generated_ids = request['all_tokens']
        generated_text = request['all_text']
        canonical_ids = prompt_ids + generated_ids
        recorded_history = request['state']['history']
        if recorded_history != canonical_ids:
            raise AssertionError('Recorded Q8 canonical history is not full prompt IDs followed by generated IDs')

        decoded_generated = tokenizer.decode(generated_ids, errors='replace')
        generated_text_ids = tokenizer.encode(generated_text, parse_special=False,
                                               plain=[(0, len(generated_text))])
        messages = context['A_messages'] + [
            {'role': 'assistant', 'content': generated_text},
            {'role': 'user', 'content': context['branches']['B']['question']},
        ]
        rendered = encoder.render(messages, add_generation_prompt=True)
        rendered_ids = encoder.encode_rendered(messages, rendered, True)
        ordinary_ids = ordinary.encode_prompt(messages, None,
            {**template_kwargs, 'add_generation_prompt': True})
        if rendered_ids != ordinary_ids:
            raise AssertionError('CowPromptEncoder next-turn IDs differ from ordinary Service.encode_prompt')

        eot_text = '<|im_end|>'
        if eot_text not in tokenizer.special_tokens:
            raise RuntimeError('Q8 tokenizer does not expose <|im_end|> special token')
        forced_eot_id = tokenizer.special_tokens[eot_text]
        active = summarize_comparison(canonical_ids, rendered_ids, tokenizer.decode,
                                      'active_bounded_reply_without_eot')
        completed_expected = canonical_ids + [forced_eot_id]
        completed = summarize_comparison(completed_expected, rendered_ids, tokenizer.decode,
                                         'synthetic_forced_eot_cpu_case_not_native_eof_test')
        assistant_only = encoder.render(context['A_messages'] + [
            {'role': 'assistant', 'content': generated_text}], add_generation_prompt=False)
        assistant_only_ids = encoder.encode_rendered(context['A_messages'] + [
            {'role': 'assistant', 'content': generated_text}], assistant_only, False)
        last_user_start = rendered.rfind('<|im_start|>user\n')
        assistant_start = rendered.rfind('<|im_start|>assistant\n', 0, last_user_start)
        assistant_after_start = rendered[assistant_start + len('<|im_start|>assistant\n'):] if assistant_start >= 0 else ''
        assistant_body = assistant_after_start.split('<|im_end|>', 1)[0]
        template_thinking_prefix = '<think>\n\n</think>\n\n'
        relevant_template_lines = [
            {'line_1based': i + 1, 'text': line}
            for i, line in enumerate(template_source.splitlines())
            if '|trim' in line or '<|im_end|>' in line
        ]

        output = {
            'schema': 'cow-q8-next-turn-boundary-probe-v1',
            'status': 'PASS_CPU_REPRODUCED_WITH_PREFIX_MISMATCH_DETAILS',
            'scope': 'Recorded Q8 prompt/output IDs and actual Q8 tokenizer/template; CPU only, no model inference, no GPU/native EOF test.',
            'fixture_evidence_path': str(args.fixture_evidence),
            'fixture_evidence_sha256': sha(fixture_raw),
            'resume_raw_path': str(args.resume_raw),
            'resume_raw_sha256': sha(resume_raw),
            'resume_request_name': request['name'],
            'recorded_output_tokens': len(generated_ids),
            'recorded_output_text_utf8_bytes': len(generated_text.encode()),
            'recorded_output_token_ids_sha256': sha(json.dumps(generated_ids, separators=(',', ':')).encode()),
            'recorded_output_text_sha256': sha(generated_text.encode()),
            'recorded_output_text_tail': generated_text[-96:],
            'recorded_output_text_ends_with_newline': generated_text.endswith('\n'),
            'recorded_output_text': generated_text,
            'decoded_recorded_ids_equal_saved_text': decoded_generated == generated_text,
            'reencoded_saved_text_token_count': len(generated_text_ids),
            'reencoded_saved_text_token_ids_equal_recorded_ids': generated_text_ids == generated_ids,
            'canonical_token_count': len(canonical_ids),
            'recorded_history_matches_prompt_plus_output': True,
            'eot_token_id': forced_eot_id,
            'new_user_question': context['branches']['B']['question'],
            'rendered_next_turn_sha256': sha(rendered.encode()),
            'rendered_next_turn_token_count': len(rendered_ids),
            'rendered_assistant_content_tail_before_eot': assistant_body[-96:],
            'rendered_assistant_content': assistant_body,
            'rendered_assistant_content_ends_with_newline': assistant_body.endswith('\n'),
            'rendered_assistant_content_chars': len(assistant_body),
            'rendered_assistant_content_matches_template_prefix_plus_trimmed_text':
                assistant_body == template_thinking_prefix + generated_text.rstrip(),
            'template_thinking_prefix': template_thinking_prefix,
            'template_preserves_generated_text_after_thinking_prefix': assistant_body.startswith(template_thinking_prefix),
            'template_strips_generated_trailing_newline': generated_text.endswith('\n') and
                assistant_body == template_thinking_prefix + generated_text.rstrip(),
            'frontend_jinja_trim_blocks': True,
            'frontend_jinja_lstrip_blocks': True,
            'actual_template_relevant_lines': relevant_template_lines,
            'ordinary_service_ids_match': True,
            'active_reply_comparison': active,
            'synthetic_forced_eot_comparison': completed,
            'assistant_only_template_tail': tokenizer.decode(assistant_only_ids[-16:], errors='replace'),
            'assistant_only_template_tail_ids': assistant_only_ids[-16:],
            'rendered_assistant_text_contains_think_open': '<think>' in generated_text,
            'rendered_assistant_text_contains_think_close': '</think>' in generated_text,
            'template_kwargs': template_kwargs,
            'template_source_sha256': sha(template_source.encode()),
            'tokenizer_source_sha256': file_sha(repo_root / 'tools' / 'strata_tokenizer.py'),
            'gguf_reader_source_sha256': file_sha(repo_root / 'tools' / 'gguf_reader.py'),
            'frontend_source_sha256': file_sha(repo_root / 'serve' / 'frontend.py'),
            'service_source_sha256': file_sha(repo_root / 'serve' / 'server.py'),
            'cow_prompt_encoder_source_sha256': file_sha(repo_root / 'serve' / 'cow_prompt_encoder.py'),
            'tokenizer_vocab_count': len(tokenizer.tokens),
            'tokenizer_merge_count': len(tokenizer.ranks),
            'tokenizer_pre': tokenizer.pre,
            'gguf_model_shard_sha256': file_sha(model_path),
            'model_weights_sha256_profile_identity': profile.get('model_weights_sha256'),
            'regex_version': regex.__version__,
            'python_version': platform.python_version(),
            'python_executable': sys.executable,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': output['status'], 'output': str(args.output),
                      'active_prefix': output['active_reply_comparison']['expected_is_exact_rendered_prefix'],
                      'active_mismatch': output['active_reply_comparison']['first_mismatch_index_zero_based'],
                      'forced_eot_prefix': output['synthetic_forced_eot_comparison']['expected_is_exact_rendered_prefix'],
                      'forced_eot_mismatch': output['synthetic_forced_eot_comparison']['first_mismatch_index_zero_based'],
                      'raw_text_reencodes_to_ids': output['reencoded_saved_text_token_ids_equal_recorded_ids']}, indent=2))


if __name__ == '__main__':
    main()
