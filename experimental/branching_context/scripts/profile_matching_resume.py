"""Diagnostic wall-time attribution around the unchanged lean runner.

Inclusive and exclusive times use a shared nested call stack. Native request
times include waiting/synchronization; they are not GPU kernel timings.
"""
import functools
import json
from pathlib import Path
import sys
import time


class Timings:
    def __init__(self):
        self.stack = []
        self.rows = {}
        self.phase = None

    def wrap(self, label, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            name = self.phase+'/'+label if self.phase else label
            frame = [time.perf_counter_ns(), 0]
            self.stack.append(frame)
            try: return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns()-frame[0]
                self.stack.pop()
                if self.stack: self.stack[-1][1] += elapsed
                row = self.rows.setdefault(name, dict(calls=0,inclusive_ns=0,exclusive_ns=0))
                row['calls'] += 1
                row['inclusive_ns'] += elapsed
                row['exclusive_ns'] += elapsed-frame[1]
        return timed


def main():
    # Same arguments and acceptance checks as lean_matching_resume.py.
    api = Path(sys.argv[sys.argv.index('--api')+1])
    output = Path(sys.argv[sys.argv.index('--output')+1])
    sys.path.insert(0,str(api/'src'))
    from context_time_travel.real import BranchingModel
    from context_time_travel.store import Store
    from context_time_travel.strata_backend import StrataBackend
    from context_time_travel import controller
    import context_time_travel.store as store_module
    import lean_matching_resume
    timings = Timings()
    targets = [(BranchingModel,n) for n in ('history','_save','_completed_schedule','_input_identity')]
    targets += [(Store,n) for n in ('append','get_snapshot','lineage')]
    targets += [(controller,n) for n in ('choose','consume','_validated_state')]
    targets += [(store_module,'_controller')]
    targets += [(StrataBackend,n) for n in ('restore','export','park','logits','consume','consume_generated','position')]
    originals = []
    for cls,name in targets:
        original = cls.__dict__[name]
        function = getattr(cls,name)
        wrapped = timings.wrap(cls.__name__+'.'+name,function)
        if isinstance(original,staticmethod): wrapped = staticmethod(wrapped)
        originals.append((cls,name,original))
        setattr(cls,name,wrapped)
    request = StrataBackend._request
    @functools.wraps(request)
    def timed_request(self, command):
        return timings.wrap('native.'+command.split()[0], request)(self,command)
    StrataBackend._request = timed_request
    stream = BranchingModel.stream
    def timed_stream(self, *args, **kwargs):
        iterator = stream(self, *args, **kwargs)
        advance = timings.wrap('API.generation_step', lambda: next(iterator))
        try:
            while True:
                previous = timings.phase
                timings.phase = 'generation'
                try:
                    try: item = advance()
                    except StopIteration: return
                finally: timings.phase = previous
                yield item
        finally: iterator.close()
    BranchingModel.stream = timed_stream
    try: lean_matching_resume.main()
    finally:
        StrataBackend._request = request
        BranchingModel.stream = stream
        for cls,name,original in originals: setattr(cls,name,original)
        if output.exists():
            (output/'wall-attribution.json').write_text(json.dumps(dict(
                scope='Entire runner including setup and final validation; nested wall times, not GPU events',
                measurements=timings.rows),indent=2)+'\n')


if __name__ == '__main__': main()
