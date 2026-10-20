"""Bounded Q4 thinking through the actual BranchWrite CLI/HTTP/GPU path.

This first probe selects host-registered budgets via existing grammar-state
controls. It does not claim a general thinking_length option or certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_probe_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'branchwrite', 'profile', 'dependency-root', 'vision-config', 'image', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-image-sha256', required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[0, 128],
                        help='Strictly increasing diagnostic thinking budgets, 0..32768')
    parser.add_argument('--questions', choices=('expanded', 'short'), nargs='+', default=['expanded', 'short'])
    parser.add_argument('--greedy-only', action='store_true', help='Do not repeat already passing sampled arms')
    parser.add_argument('--until-natural-end', action='store_true',
                        help='Stop at natural thought closure, regardless of final quality; greedy only')
    args = parser.parse_args(argv)
    if (any(not 0 <= budget <= 32768 for budget in args.budgets)
            or args.budgets != sorted(set(args.budgets))):
        parser.error('budgets must be strictly increasing integers in 0..32768')
    if len(args.questions) != len(set(args.questions)):
        parser.error('questions must be unique')
    if args.until_natural_end and not args.greedy_only:
        parser.error('natural-end escalation requires --greedy-only')
    return args


def natural_end_outcome(arm):
    """A forced correct answer does not stop escalation; a natural wrong one does."""
    final = arm.get('greedy', arm['first'])
    if final['controller']['status'] == 'stopped' and final['progress']['close_reason'] == 'natural':
        return dict(budget=arm['budget'], quality=final['image_quality'], final=final['final'],
                    reasoning_tokens=final['progress']['reasoning_tokens'], close_reason='natural')
    return None


def main():
    args = parse_probe_args()
    runtime, bw = args.runtime.resolve(strict=True), args.branchwrite.resolve(strict=True)
    sys.path[:0] = [str(p) for p in (bw, runtime, runtime / 'tools',
        runtime / 'experimental/branching_context/src', runtime / 'experimental/branching_context/scripts',
        args.dependency_root)]
    from run_cli_grammar_probe import GrammarProbe, _atomic_json
    from branchwrite.cow_client import CowClient, OperationJournal
    from context_time_travel.input_spans import history_for_snapshot, InputHistory
    from serve.cow_grammar import GrammarRegistry
    from tools.strata_tokenizer import Tokenizer

    image = args.image.read_bytes()
    if sha(image) != args.expected_image_sha256:
        raise ValueError('Image differs from the retained H02 fixture')
    output = args.output.absolute()
    output.mkdir(mode=0o700, exist_ok=False)

    class ImageOwner(GrammarProbe):
        def _bootstrap(self, registry_path):
            path = super()._bootstrap(registry_path)
            body = path.read_text()
            marker = "sys.argv=['cow-http',*sys.argv[5:]]\n"
            if body.count(marker) != 1:
                raise ValueError('Bootstrap contract changed')
            path.write_text(body.replace(marker, "sys.argv=['cow-http','--vision-config'," +
                repr(str(args.vision_config.resolve(strict=True))) + ",*sys.argv[5:]]\n"), newline='\n')
            return path

    probe = ImageOwner(SimpleNamespace(runtime_root=runtime, profile=args.profile,
        store=output / 'store', run_dir=output, dependency_root=args.dependency_root, ram_gib=96, disk_gib=128))
    probe.record.update(status='RUNNING', numerical_max_error=None, numerical_oracle='NOT_RUN',
        scope='Initial thinking controls, fixed registry budgets, greedy plus Min-P; not VC/G qualification',
        cases=[], chat_server='NOT_STARTED', q8='DEFERRED_BY_OWNER',
        requested_budgets=args.budgets, requested_questions=args.questions,
        greedy_only=args.greedy_only, until_natural_end=args.until_natural_end)
    error = None
    try:
        probe._imports()
        profile = json.loads(args.profile.read_bytes())
        if profile['execution']['text_template_kwargs'] != {'enable_thinking': True}:
            raise ValueError('Probe requires its new thinking rendering profile')
        probe.runtime_closure(profile, runtime)
        probe.record.update(execution_profile_id=profile['execution_profile_id'],
            model_contract_id=profile['model_contract_id'], profile_sha256=sha(args.profile.read_bytes()),
            image_sha256=sha(image), native_sha256=profile['native_build']['sha256'])
        command = profile['execution']['sut_command']
        tok = Tokenizer.from_gguf(command[command.index('--native') + 1])
        close, = tok.encode('</think>', parse_special=True)
        probe.record['thinking_token'] = dict(id=close, gguf_type=tok.token_types[close], bytes='</think>')
        registry_path = output / 'grammar-registry.json'
        _atomic_json(registry_path, dict(schema=1, states={
            'THINK' + str(budget): dict(format='thinking', source=dict(schema=1, max_tokens=budget,
                final=dict(format='gbnf', source='root ::= "LEFT" | "RIGHT" | "NOT_VISIBLE"')))
            for budget in args.budgets}))
        registry = GrammarRegistry.load(registry_path, tok, profile['execution']['eos_tokens'])
        probe.record['grammar'] = dict(engine_id=registry.engine_id, spec_sha256=registry.spec_sha256)
        probe.tracker = probe.ProgressTracker(output, 'thinking-cli').start()
        probe.start_service(registry_path)
        pin = sha((str(output) + probe.record['profile_sha256']).encode())
        client = CowClient.cli(probe.ready_url(probe.service.ready), runtime / 'tools/cow_cli.py',
            state_dir=output / 'cli-state', journal=OperationJournal(output / 'journal'),
            api_key=probe.secret, endpoint_pin=pin)
        namespace = client.create_namespace('Thinking image test', 'thinking-namespace')['namespace_id']
        upload = client.upload_image(namespace, image)

        def view(result):
            controller = probe.store.get_snapshot(result.receipt.snapshot)['controller']
            history = history_for_snapshot(probe.store, result.receipt.snapshot)
            tokens = history.text_tokens if isinstance(history, InputHistory) else history
            session = registry.restore(controller, list(tokens))
            progress = session.thinking_progress()
            scope = tokens[controller['constraint']['scope_start']:]
            thought = final = None
            if close in scope:
                i = scope.index(close)
                thought = tok.decode(list(scope[:i]))
                final = tok.decode([t for t in scope[i+1:] if t not in profile['execution']['eos_tokens']])
            if progress['reasoning_tokens'] > progress['max_tokens']:
                raise ValueError('Thinking exceeded the immutable budget')
            if controller['status'] == 'stopped' and final not in ('LEFT', 'RIGHT', 'NOT_VISIBLE'):
                raise ValueError('Stopped answer violates the final grammar')
            presented = result.answer.get('thinking')
            reasoning = tok.decode(list(scope[:scope.index(close)] if close in scope else scope))
            expected = dict(schema=1, snapshot=result.receipt.snapshot, progress=progress,
                complete=controller['status'] == 'stopped', reasoning_text=reasoning,
                final_text=final if controller['status'] == 'stopped' else None)
            if presented != expected:
                raise ValueError('CLI/HTTP thinking view differs from canonical consumed state')
            return dict(receipt=dict(result.receipt.raw), answer=dict(result.answer), progress=progress,
                controller=controller, thought=thought, final=final,
                image_quality=('UNFINISHED' if final is None else 'PASS' if final == 'LEFT' else 'FAIL'))

        for question in args.questions:
            text = (bw / 'fixtures/image_questions' / (question + '.md')).read_text().split('\n', 1)[1].strip()
            probe.phase(question + '-prepare')
            root = client.prepare(question, namespace, text, question + '-prepare', images=(upload,))
            parent = probe.store.get_snapshot(root.receipt.snapshot)
            payload = root.receipt.raw['payload_sha256']
            payload_path = output / 'store/checkpoints' / (payload + '.state')
            with payload_path.open('rb') as stream:
                parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            row = dict(question=question, question_sha256=sha(text.encode()), cases=[])
            probe.record['cases'].append(row)
            for budget in args.budgets:
                state = 'THINK' + str(budget)
                branch = question + '-' + state
                head = client.fork(question, root.receipt.snapshot, branch, branch + '-fork')
                probe.phase(branch + '-first-four')
                started = time.monotonic()
                first = client.answer(branch, head.revision, branch + '-first', 4,
                    settings=dict(temperature=0.), grammar_state=state)
                partial = view(first)
                arm = dict(budget=budget, first=partial)
                row['cases'].append(arm)
                probe.publish()
                if partial['controller']['status'] == 'active':
                    if not args.greedy_only:
                        sampled_branch = branch + '-minp'
                        fork = client.fork(branch, first.receipt.snapshot, sampled_branch, sampled_branch + '-fork')
                    probe.phase(branch + '-greedy-resume')
                    done = client.continue_(branch, first.receipt.branch.revision, branch + '-resume', budget + 32)
                    arm['greedy'] = view(done)
                    if not args.greedy_only:
                        probe.phase(branch + '-minp-resume')
                        policy = dict(temperature=.7, min_p=.05, top_k=60, top_p=.95)
                        sampled = client.continue_(sampled_branch, fork.revision, sampled_branch + '-resume',
                            budget + 32, settings=policy)
                        arm['min_p'] = view(sampled)
                    if any(arm[name]['controller']['status'] != 'stopped' for name in ('greedy', 'min_p') if name in arm):
                        raise ValueError('Thinking/final did not finish within its bounded request')
                    if probe.store.get_snapshot(first.receipt.snapshot)['controller'] != partial['controller']:
                        raise ValueError('A sibling mutated the partial thinking controller')
                arm['elapsed_seconds'] = time.monotonic() - started
                if probe.store.get_snapshot(root.receipt.snapshot) != parent:
                    raise ValueError('Thinking changed the saved root')
                with payload_path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != parent_sha:
                        raise ValueError('Thinking changed the saved root checkpoint bytes')
                arm['parent_bytes_unchanged'] = True
                probe.publish()
                if args.until_natural_end:
                    outcome = natural_end_outcome(arm)
                    if outcome is not None:
                        row['escalation_outcome'] = outcome
                        probe.publish()
                        break
            if args.until_natural_end and 'escalation_outcome' not in row:
                row['escalation_outcome'] = dict(status='NATURAL_END_NOT_REACHED', largest_budget=args.budgets[-1])
                raise ValueError('Natural thought closure not reached within the explicitly bounded sweep')
        probe.record['status'] = 'BOUNDED_THINKING_CLI_Q4_PASS_QUALITY_REPORTED_SEPARATELY'
    except BaseException as exc:
        error = exc
        probe.record['status'] = 'THINKING_CLI_FAIL'
        probe.record.setdefault('failures', []).append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        if probe.service is not None:
            try:
                probe.stop_service()
            except BaseException as exc:
                error = error or exc
                probe.record['status'] = 'CLEANUP_FAIL'
                probe.record.setdefault('failures', []).append(dict(type=type(exc).__name__, message=str(exc)))
        probe.publish()
        if probe.tracker:
            probe.tracker.finish(probe.record['status'], None if error is None else type(error).__name__)
            probe.tracker.close()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
