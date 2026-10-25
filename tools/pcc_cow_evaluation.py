"""Bounded evaluation through the same consumed COW requests used by BW.

This is an execution adapter, not a scorer, trainer, or workflow controller.
The caller supplies an already pinned native profile and holds the GPU lease.
"""
from pathlib import Path
import hashlib
import os


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_sources(profile, runtime_root):
    """Check the existing COW profile closure, including this execution adapter."""
    closure = profile['native_build']['selected_runtime_source_closure']['files_sha256']
    required = ('tools/pcc_cow_evaluation.py',
                'experimental/branching_context/src/context_time_travel/real.py',
                'experimental/branching_context/src/context_time_travel/qwen35_backend.py',
                'experimental/branching_context/src/context_time_travel/llama_dense_backend.py')
    if any(name not in closure for name in required):
        raise ValueError('COW evaluation requires its complete pinned source closure')
    base = Path(runtime_root).resolve(strict=True)
    for name, expected in closure.items():
        path = (base / name).resolve(strict=True)
        if not path.is_relative_to(base) or sha(path) != expected:
            raise ValueError('COW source closure differs: ' + name)


class CowEvaluationSession:
    """One loaded model; one consumed native request per independent case.

    No fresh-oracle replacement, duplicated KV cache, or bespoke sampler.
    COW owns history, controller, checkpoint, import/export and native selection.
    Only RAM materializations are evicted between cases; evidence stays on disk.
    """
    def __init__(self, profile, directory, runtime_root):
        self.profile = profile
        self.runtime_root = Path(runtime_root).resolve(strict=True)
        verify_sources(profile, self.runtime_root)
        if profile.get('backend') != 'llama-dense-llama.cpp':
            raise ValueError('This bounded integration currently qualifies dense MiniCPM only')
        for name, expected in profile['execution']['environment'].items():
            if os.environ.get(name) != expected:
                raise ValueError('COW execution environment differs: ' + name)
        from context_time_travel.real import BranchingModel
        self.backend = BranchingModel.load_backend(profile)
        try:
            self.model = BranchingModel(directory, self.backend, profile)
        except BaseException:
            self.backend.close()
            raise
        self.closed = False
        self.failed = False

    def complete(self, prefix, *, maximum, stream):
        e = self.profile['execution']
        if self.closed or self.failed:
            raise ValueError('Closed or failed COW evaluation session; no implicit retry')
        if (type(prefix) not in (list, tuple) or not prefix
                or any(type(t) is not int or not 0 <= t < self.backend.vocab_size for t in prefix)
                or type(maximum) is not int or not 1 <= maximum <= 4096
                or len(prefix) + maximum > e['capacity']
                or type(stream) is not str or not stream):
            raise ValueError('Explicit bounded COW prefix, output and stream required')
        before = self.backend.work_counters()
        try:
            root, reused = self.model.create_with_reuse(tuple(prefix))
            # BW's COW server dispatches to this same request implementation.
            result = self.model.run_greedy_request(root, (), maximum, stream=stream)
            tokens = result['tokens']
            expected_history = tuple(prefix) + tuple(tokens)
            if (not tokens or len(tokens) > maximum
                    or self.model.history(result['snapshot']) != expected_history
                    or self.model.cache[result['snapshot']].consumed != len(expected_history)):
                raise ValueError('COW request omitted the consumed output boundary')
            eos = set(e['eos_tokens'])
            if any(t in eos for t in tokens[:-1]):
                raise ValueError('COW request continued after an EOS token')
            stopped = tokens[-1] in eos
            if stopped != result['stopped'] or (not stopped and len(tokens) != maximum):
                raise ValueError('COW stopping receipt differs from the sampled boundary')
            after = self.backend.work_counters()
            delta = {k: after[k] - before[k] for k in before}
            expected_work = len(tokens) + (0 if reused else len(prefix))
            if delta['consumed_tokens'] != expected_work:
                raise ValueError('COW consumed-work receipt differs')
            return dict(**result, root_snapshot=root, prefix_reused=reused,
                        counters=delta, stop_reason='eos' if stopped else 'length',
                        stop_token_id=tokens[-1] if stopped else None,
                        execution_path='cow-consumed-request-v1')
        except BaseException:
            self.failed = True
            raise
        finally:
            # This benchmark has independent cases. Keep canonical history and
            # controller records, but bound rebuildable binary RAM to one case.
            self.model.evict_all()

    def close(self):
        if not self.closed:
            self.closed = True
            self.backend.close()
            verify_sources(self.profile, self.runtime_root)
            if sha(self.profile['model_path']) != self.profile['model_weights_sha256']:
                raise ValueError('COW evaluation model changed during execution')
