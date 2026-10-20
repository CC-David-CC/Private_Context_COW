"""Compact measured evidence and extra prefix/controller audits; no inference."""
from collections import Counter
import codecs
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def summarize(matrix,output):
    record=json.loads(matrix.read_text())
    report=dict(status=record['status'],scope='greedy matching-operator synthetic-token workloads; CLI, not web serving',
        GPU=record['GPU'],cases=[],original_gates='G0 PASS; original G1-G5 not qualified')
    for job in record['jobs']:
        if job['status']!='PASS': continue
        case=matrix.parent/job['name']
        result=json.loads((case/'result.json').read_text())
        prepare=json.loads((case/'prepare.json').read_text())
        resume=json.loads((case/'resume.json').read_text())
        profile=json.loads((case/'profile.json').read_text())
        # Read tokenizer metadata only. No model/native process is instantiated.
        spec=importlib.util.spec_from_file_location('pool_summary_gguf',profile['gguf_reader'])
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        tokens=module.GGUFFile(Path(profile['model_path'])).metadata['tokenizer.ggml.tokens']
        direct=list(range(33,127))+list(range(161,173))+list(range(174,256))
        missing=[b for b in range(256) if b not in direct]
        byte_map=dict(zip(map(chr,direct+list(range(256,256+len(missing)))),direct+missing))
        first={r['operation'].split('-tree-')[1].split('-')[0]:r for r in prepare['requests'] if '-tree-' in r['operation']}
        second={r['operation'].split('-tree-')[1].split('-')[0]:r for r in resume['requests']}
        checks=[]
        for entry in result['comparisons']:
            tree=entry['tree'];a,b=first[str(tree)],second[str(tree)]
            root=prepare['roots'][tree%result['sizes']['roots']]
            width=4*248320
            fresh_prefix=(case/('fresh-tree-'+str(tree)+'.f32.requests.f32')).read_bytes()[:width]
            if hashlib.sha256(fresh_prefix).hexdigest()!=root['frontier_sha256']:
                raise AssertionError('Saved prefix differs from uncached matching reference')
            ca,cb=a['state']['controller'],b['state']['controller']
            for key in ('schema','rng_algorithm','stream_key','next_draw','policy','sampler_order','grammar_state','output_decoder','stop_tail'):
                if ca[key]!=cb[key]: raise AssertionError('Controller changed across restart: '+key)
            added=b['answer']['tokens']
            counts=Counter(ca['seen'])
            counts.update(str(t) for t in added)
            if dict(counts)!=cb['seen'] or ca['accepted_in_scope']+len(added)!=cb['accepted_in_scope']:
                raise AssertionError('Controller token accounting changed')
            decoder=codecs.getincrementaldecoder('utf8')(errors='replace')
            decoder.setstate((bytes.fromhex(ca['output_pending_hex']),0))
            for token in added: decoder.decode(bytes(byte_map[c] for c in tokens[token]),final=False)
            if decoder.getstate()[0].hex()!=cb['output_pending_hex']:
                raise AssertionError('Pending output decoder state changed')
            status='stopped' if ca['status']=='stopped' or (added and added[-1] in profile['execution']['eos_tokens']) else 'active'
            if cb['status']!=status: raise AssertionError('Stopping state changed')
            expected=a['state']['history']+added
            if expected!=b['state']['history']: raise AssertionError('Continuation duplicated/omitted input')
            checks.append(dict(tree=tree,prefix_full_frontier='PASS',controller='PASS',history='PASS',
                final_controller_sha256=digest(cb)))
        requests=[r for phase in (prepare,resume) for r in phase['requests'] if r.get('mode') is None]
        rows=[]
        for r in requests:
            answer=r['answer'];n=len(answer['tokens']);duration=answer.get('generation_seconds',0)
            rows.append(dict(operation=r['operation'],outputs=n,
                generation_seconds=duration,generation_tps=n/duration if n and duration else None,
                request_seconds=answer.get('request_seconds'),capture_seconds=answer.get('capture_seconds'),
                counter_deltas=answer.get('counter_deltas'),controller_sha256=digest(r['state']['controller']),
                payload_sha256=r['state']['manifest']['payload_sha256']))
        report['cases'].append(dict(name=job['name'],status='PASS',sizes=result['sizes'],
            numerical_max_error=result['numerical_max_error'],comparisons=result['comparisons'],
            additional_checks=checks,requests=rows,elapsed_seconds=job['seconds'],
            phases=result['phases'],
            native_exit_observed=[prepare['native_exit_observed'],resume['native_exit_observed']],
            ram_peak_bytes=max(prepare['pool']['ram']['peak_accounted_bytes'],resume['pool']['ram']['peak_accounted_bytes']),
            ram_limit=prepare['pool']['ram']['byte_limit'],
            clean_evictions=prepare['pool']['ram']['clean_evictions']+resume['pool']['ram']['clean_evictions'],
            disk_file_bytes=resume['pool']['disk']['file_bytes'],
            actual_outputs=sum(e['outputs'] for e in result['comparisons']),
            continued_trees=sum(bool(e['resumed_outputs']) for e in result['comparisons']),
            profile_id=result['profile_id'],model_contract_id=result['model_contract_id'],
            model_weights_sha256=result['model_weights_sha256'],hardware=result['hardware'],build_sha256=result['build_sha256'],
            parent_bytes_unchanged=resume['parent_bytes_unchanged'],
            original_result_sha256=hashlib.sha256((case/'result.json').read_bytes()).hexdigest()))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],cases=[dict(name=c['name'],outputs=c['actual_outputs'],
        continued_trees=c['continued_trees'],max_error=c['numerical_max_error'],evictions=c['clean_evictions']) for c in report['cases']])))


if __name__=='__main__': summarize(Path(sys.argv[1]),Path(sys.argv[2]))
