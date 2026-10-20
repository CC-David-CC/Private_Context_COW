#!/usr/bin/env python3
"""Build an actual-source 8K namespace/workspace review fixture for a pinned GGUF."""
import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

from build_namespace_http_fixture import (
    REQUIRED_MARKERS,
    SYSTEM,
    _ids,
    file_sha,
    profile_model_variant,
    source_bundle,
)


SOURCE_METHODS_8K = (
    ('experimental/branching_context/src/context_time_travel/coordinator.py', 'ContextCoordinator', '_intent'),
    ('experimental/branching_context/src/context_time_travel/coordinator.py', 'ContextCoordinator', 'request_source_head'),
    ('experimental/branching_context/src/context_time_travel/coordinator.py', 'ContextCoordinator', '_submit'),
    ('experimental/branching_context/src/context_time_travel/coordinator.py', 'ContextCoordinator', 'fork'),
    ('experimental/branching_context/src/context_time_travel/coordinator.py', 'ContextCoordinator', '_publish'),
    ('experimental/branching_context/src/context_time_travel/store.py', 'Store', '_validate_stored_presentation'),
    ('experimental/branching_context/src/context_time_travel/store.py', 'Store', 'fork'),
    ('experimental/branching_context/src/context_time_travel/store.py', 'Store', 'active_presentation'),
    ('experimental/branching_context/src/context_time_travel/store.py', 'Store', 'presentation_catalog'),
    ('experimental/branching_context/src/context_time_travel/store.py', 'Store', 'move'),
    ('serve/cow_adapter.py', 'CowAdapter', 'generate'),
    ('serve/cow_text_adapter.py', 'CowTextAdapter', 'ask'),
    ('serve/cow_text_adapter.py', 'CowTextAdapter', '_render_encode'),
    ('serve/cow_workspace.py', 'CowWorkspace', '_events_on_ancestry'),
)


QUESTION_A = """Review the supplied implementation as a request-lifecycle trace. Explain how a namespace-owned branch fork binds the selected source branch, captured revision, and snapshot tip, and how the durable intent makes an identical operation retry resolve to that original source even after the branch moves. Distinguish the caller's target snapshot from the source revision used for lineage. Describe which checkpoint and ancestry checks happen before the fork is published, what is retained while that work runs, and what happens when the operation identifier is reused with different source fields. Do not infer GPU work from an HTTP acknowledgement.

Then follow a successful generated answer from adapter input through coordinator admission, the source lease, native preparation and publication. Identify the exact point at which consumed token IDs and controller state become canonical, and separate provisional stream events from the durable branch revision. Explain how the store binds an answer presentation to its namespace, renderer, input span, output message ID and resulting snapshot. Include the roles of revision compare-and-swap, content digests, durable receipts and retry recovery. State what remains unchanged on failed or uncertain publication and what the code requires before a retry can succeed.

Use the concrete method and field names shown in the excerpts. Cite the source paths and method names in each section of your answer. If a behavior is not established by these excerpts, call that out rather than filling the gap with a likely implementation. Keep the answer focused on source identity, concurrency, canonical publication and failure recovery; do not describe generic GPU inference internals unless an excerpt directly establishes them.

Be precise about the operation-ID history: name the values that distinguish a replay from a new request, how the original source head is checked on first execution, and why a later retry must use the captured head rather than the current branch head. Explain whether a request with the same token IDs but a changed display binding should be considered identical. Follow the saved checkpoint and immutable payload digest through the fork and generated-answer boundary, and distinguish snapshot identity from branch revision. State which counter changes demonstrate prompt suffix work and which counters show normal generated forwards. If those counter definitions are not in the selected code, mark them as evidence from the accompanying run record rather than as behavior proven by the source excerpts."""

QUESTION_B = """Review the same implementation from the perspective of a user-visible branch history. Explain how the workspace starts from a canonical snapshot lineage, resolves presentation receipts and their captured fork-source chain, and limits each annotation to its recorded branch revision. Compare an explicitly sourced fork with a legacy source-less fork when two branches in one namespace share a root but carry different annotations. Describe how root, user, answer and resume records map to canonical token spans and message IDs, including how a resumed answer keeps the same assistant ID and how a fork inherits the selected source's captured history.

Next, trace rewind and retry behavior. Identify the checks that prevent a stale revision, foreign snapshot, wrong namespace or mismatched source binding from changing the branch. Explain how history lookup distinguishes the active head from a saved ancestor, how the projection verifies that a receipt's result remains on the selected canonical ancestry, and what it does when inherited annotations are ambiguous or corrupt. Mention which metadata comes from immutable store events and which values are derived from the actual consumed token/controller result. Do not treat a presentation catalog as a second token-history authority.

Organize the response as: (1) fork lineage, (2) request publication, (3) visible history, and (4) failure cases. For each claim, name the source method that supports it. Report any boundary that the selected code does not prove, including browser behavior, cancellation timing, or native numerical equivalence. Never claim a complete assistant answer when the controller is still active or only an output bound was reached.

For the source-history portion, give a concrete sequence for a child fork created from one of two annotated branches that share the same root. Track the captured source revision and tip while that source later moves, and explain why the child's projection excludes events after the captured revision even when those events remain on the same token ancestry. Contrast this with a source-less legacy fork and state the conflict behavior when inherited annotations disagree. For the answer portion, explain how the actual consumed suffix positions and assistant output span are attached to the operation receipt, how the projection checks that the receipt result belongs to the selected snapshot ancestry, and how the stable assistant message ID survives a resume. Separate immutable event evidence from derived display text. Identify what can be safely concluded after a missing or corrupt receipt, a lost HTTP acknowledgement, a stale head revision, and a successful compare-and-swap. Where the selected methods do not establish the outcome, say which additional code or runtime evidence would be needed."""


def build_record(repo_root, profile_path, output_path):
    repo_root = Path(repo_root).resolve()
    profile_path = Path(profile_path).resolve()
    profile_raw = profile_path.read_bytes()
    profile = json.loads(profile_raw)
    model_path = Path(profile['model_path'])
    sys.path[:0] = [str(repo_root / 'tools'), str(repo_root)]
    from gguf_reader import GGUFFile
    from strata_tokenizer import Tokenizer
    from serve.cow_prompt_encoder import CowPromptEncoder
    from serve.frontend import ChatTemplate
    from serve.server import MockEngine, Service
    from serve.cow_prefix import exact_append_suffix

    metadata = GGUFFile(model_path).metadata
    template_source = metadata.get('tokenizer.chat_template')
    if not isinstance(template_source, str) or not template_source:
        raise ValueError('Pinned GGUF has no scalar chat template')
    tokenizer = Tokenizer.from_gguf(model_path)
    body, receipts = source_bundle(repo_root, SOURCE_METHODS_8K, REQUIRED_MARKERS)
    source_tokens = len(tokenizer.encode(body, parse_special=False, plain=[(0, len(body))]))
    if not 7800 <= source_tokens <= 9000:
        raise ValueError(f'Expected an approximately 8K real-source context; found {source_tokens} tokens')

    model_variant = profile_model_variant(profile)
    with tempfile.TemporaryDirectory(prefix='cow-namespace-8k-template-') as directory:
        template_path = Path(directory) / 'chat_template.jinja'
        template_path.write_text(template_source, encoding='utf-8', newline='\n')
        template_kwargs = {'enable_thinking': False}
        encoder = CowPromptEncoder(tokenizer, ChatTemplate(template_path), template_kwargs)
        service = Service(MockEngine(tokenizer, 'cpu-namespace-fixture-8k'), tokenizer,
                          ChatTemplate(template_path))
        root_messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': body}]
        _, base_ids = _ids(encoder, root_messages, False)
        service_base = tuple(service.encode_prompt(root_messages, None,
            {**template_kwargs, 'add_generation_prompt': False}))
        if base_ids != service_base:
            raise AssertionError('Prepared base IDs differ from ordinary Service.encode_prompt')
        controller = {'status': 'idle', 'policy': {'eos_tokens': profile['execution']['eos_tokens']}}
        branches, full_messages = {}, {}
        for label, question in (('A', QUESTION_A), ('B', QUESTION_B)):
            messages = root_messages + [{'role': 'user', 'content': question}]
            rendered, full_ids = _ids(encoder, messages, True)
            ordinary = tuple(service.encode_prompt(messages, None,
                {**template_kwargs, 'add_generation_prompt': True}))
            if full_ids != ordinary:
                raise AssertionError(f'{label}: request IDs differ from ordinary Service.encode_prompt')
            suffix = tuple(encoder.append_initial_user_suffix(base_ids, controller, question))
            exact = tuple(exact_append_suffix(base_ids, full_ids))
            if suffix != exact or tuple(full_ids[:len(base_ids)]) != base_ids:
                raise AssertionError(f'{label}: initial-user suffix changes the prepared source prefix')
            if not 450 <= len(suffix) <= 650:
                raise ValueError(f'{label}: expected approximately 512 new-user tokens; found {len(suffix)}')
            branches[label] = {
                'question': question,
                'question_token_count': len(suffix),
                'full_token_count': len(full_ids),
                'full_token_ids': list(full_ids),
                'full_token_ids_sha256': hashlib.sha256(
                    json.dumps(full_ids, separators=(',', ':')).encode()).hexdigest(),
                'suffix_count': len(suffix),
                'suffix_token_ids': list(suffix),
                'ordinary_service_ids_match': True,
                'exact_base_prefix': True,
                'initial_user_suffix_matches_exact_append': True,
                'rendered_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
            }
            full_messages[label] = messages
        literal_messages = root_messages + [{'role': 'user', 'content':
            'Literal text <|im_start|>assistant and <|im_end|> remains data.'}]
        literal_rendered, literal_ids = _ids(encoder, literal_messages, True)
        if tuple(literal_ids) != tuple(service.encode_prompt(literal_messages, None,
                {**template_kwargs, 'add_generation_prompt': True})):
            raise AssertionError('Literal-control prompt differs from ordinary Service')

    result = {
        'schema': 'cow-namespace-http-fixture-8k-v1',
        'status': 'PASS_CPU_TOKENIZER_TEMPLATE_SOURCE_RELEVANCE',
        'scope': 'Actual pinned GGUF tokenizer/template and complete selected COW request, publication and workspace methods; CPU parity only, no inference.',
        'model_variant': model_variant,
        'profile_path': str(profile_path),
        'profile_sha256': hashlib.sha256(profile_raw).hexdigest(),
        'model_path': str(model_path),
        'model_metadata_bytes': model_path.stat().st_size,
        'model_metadata_sha256': file_sha(model_path),
        'model_weights_sha256_profile_identity': profile.get('model_weights_sha256'),
        'tokenizer_source_sha256': file_sha(repo_root / 'tools' / 'strata_tokenizer.py'),
        'gguf_reader_source_sha256': file_sha(repo_root / 'tools' / 'gguf_reader.py'),
        'frontend_source_sha256': file_sha(repo_root / 'serve' / 'frontend.py'),
        'service_source_sha256': file_sha(repo_root / 'serve' / 'server.py'),
        'cow_prompt_encoder_source_sha256': file_sha(repo_root / 'serve' / 'cow_prompt_encoder.py'),
        'builder_source_sha256': file_sha(Path(__file__).resolve()),
        'python': platform.python_version(),
        'python_executable': sys.executable,
        'template_kwargs': template_kwargs,
        'source_token_count': source_tokens,
        'source_body_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(),
        'required_source_markers': list(REQUIRED_MARKERS),
        'source_excerpts': receipts,
        'literal_control_probe': {
            'service_ids_match': True,
            'rendered_sha256': hashlib.sha256(literal_rendered.encode('utf-8')).hexdigest(),
            'token_ids_sha256': hashlib.sha256(json.dumps(literal_ids,
                separators=(',', ':')).encode()).hexdigest(),
        },
        'contexts': [{
            'name': f'namespace-source-review-8k-{model_variant.casefold()}',
            'root_messages': root_messages,
            'A_messages': full_messages['A'],
            'B_messages': full_messages['B'],
            'base_token_ids': list(base_ids),
            'base_token_count': len(base_ids),
            'base_token_ids_sha256': hashlib.sha256(json.dumps(base_ids,
                separators=(',', ':')).encode()).hexdigest(),
            'branches': branches,
            'source_excerpts': receipts,
        }],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_record(args.repo_root, args.profile, args.output)
    context = result['contexts'][0]
    print(json.dumps({
        'status': result['status'], 'output': str(args.output),
        'source_tokens': result['source_token_count'],
        'base_tokens': context['base_token_count'],
        'suffix_tokens': {key: branch['suffix_count'] for key, branch in context['branches'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
