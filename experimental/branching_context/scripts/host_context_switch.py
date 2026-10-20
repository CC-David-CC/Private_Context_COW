"""Real-GPU three-context host parking, disk fallback and fresh-reference audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from durable_resume import open_model, publish, sha
from native_request_probe import reference, frontier
from resident_resume import save


def state(model, sid):
    from context_time_travel.store import _json_bytes
    return dict(history_sha256=hashlib.sha256(_json_bytes(list(model.history(sid)))).hexdigest(),
        controller_sha256=hashlib.sha256(_json_bytes(model.controller(sid))).hexdigest(),
        frontier_sha256=hashlib.sha256(frontier(model.next_logits(sid))).hexdigest(),
        consumed=model.consumed(sid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix',type=int,default=128)
    parser.add_argument('--suffix',type=int,default=4)
    parser.add_argument('--generated',type=int,default=8)
    args = parser.parse_args()
    if min(args.prefix,args.suffix,args.generated) <= 0: parser.error('Positive token counts required')
    args.output.mkdir(parents=True,exist_ok=False)
    profile = json.loads(args.profile.read_text())
    if args.prefix+args.suffix+args.generated > profile['execution']['capacity']:
        raise ValueError('Fixture exceeds pinned context capacity')
    publish(args.output/'profile.json',profile)
    report = dict(status='RUNNING',scope='three synthetic contexts, one active request; not gate certification',
        profile_id=profile['execution_profile_id'],harness_sha256=sha(__file__),contexts=[],switches=[],
        checks=[],numerical_max_error=0.)
    model = None
    offline = args.output/'offline-checkpoints'
    try:
        model,startup = open_model(SimpleNamespace(profile=args.profile,output=args.output,phase='host-switch'))
        report.update(startup_seconds=startup,native_pid=model.backend.process.pid,
            staging_storage=model.backend.state_staging_storage,staging_directory=str(model.backend._directory))
        for index in range(3):
            prefix = [100+index*40+n%17 for n in range(args.prefix)]
            suffix = [220+index*40+n%11 for n in range(args.suffix)]
            sid = model.create(prefix)
            branch = 'context-'+str(index)
            model.store.fork(branch,sid)
            publication = save(model,branch,sid)
            initial = state(model,sid)
            answer = model.run_greedy_request(sid,suffix,args.generated,stream='host-'+str(index))
            expected = state(model,answer['snapshot'])
            model.cache.pop(answer['snapshot'],None)
            parked = model.park_checkpoint(sid)
            if type(model.cache[sid].blob) is not bytes:
                raise AssertionError('Host parking retained a GPU image instead of immutable bytes')
            report['contexts'].append(dict(root=sid,branch=branch,prefix=prefix,suffix=suffix,
                publication=publication,initial=initial,expected=expected,tokens=answer['tokens'],parked=parked))
            publish(args.output/'result.json',report)
        # These files belong only to this new test store. Make them inaccessible
        # to materialization so the RAM result cannot secretly use durable files.
        model.checkpoint_directory.rename(offline)
        model.checkpoint_directory.mkdir(mode=0o700)
        # Python image finalizers queue native releases. Drain that queue through
        # the existing owner-serialized lifecycle API before testing RAM-only.
        model.backend.release_checkpoints()
        parked_stats = model.backend.work_counters()
        if parked_stats['handle_live_entries'] != 0:
            raise AssertionError('A retained native snapshot could conceal RAM restoration')
        report['all_host_parked_counters'] = parked_stats
        original = {p.name:sha(p) for p in offline.iterdir() if p.is_file()}
        for index in (0,1,2,0,2,1):
            entry = report['contexts'][index]
            sid = entry['root']
            if type(model.cache[sid].blob) is not bytes:
                raise AssertionError('Inactive context is not host-backed')
            before = model.backend.work_counters()
            answer = model.run_greedy_request(sid,entry['suffix'],args.generated,stream='host-'+str(index))
            after = model.backend.work_counters()
            if (after['forward_tokens']-before['forward_tokens'] != len(entry['suffix'])+len(answer['tokens'])
                    or after['cold_restore_copy_bytes'] <= before['cold_restore_copy_bytes']):
                raise AssertionError('Host switch replayed a prefix or did not transfer host state')
            actual = state(model,answer['snapshot'])
            if answer['tokens'] != entry['tokens'] or actual != entry['expected']:
                raise AssertionError('Host-backed continuation changed consumed state')
            model.cache.pop(answer['snapshot'],None)
            parked = model.park_checkpoint(sid)
            if parked['state_sha256'] != entry['parked']['state_sha256']:
                raise AssertionError('Restored prefix serialized bytes changed')
            report['switches'].append(dict(context=index,outputs=len(answer['tokens']),
                generation_seconds=answer['generation_seconds'],request_seconds=answer['request_seconds'],
                forward_tokens=after['forward_tokens']-before['forward_tokens'],
                host_upload_bytes=after['cold_restore_copy_bytes']-before['cold_restore_copy_bytes'],
                native_staging_file_bytes=after['loaded_file_bytes']-before['loaded_file_bytes']))
            if any(model.checkpoint_directory.iterdir()):
                raise AssertionError('RAM-only experiment published or loaded durable checkpoint files')
            publish(args.output/'result.json',report)
        # Audit the suffix boundary separately; this rerun is not speed timing.
        for index,entry in enumerate(report['contexts']):
            sid = model.feed(entry['root'],entry['suffix'])
            entry['suffix_frontier_sha256'] = state(model,sid)['frontier_sha256']
            model.cache.pop(sid,None)
            model.park_checkpoint(entry['root'])
        if original != {p.name:sha(p) for p in offline.iterdir() if p.is_file()}:
            raise AssertionError('Durable parent checkpoint bytes changed')
        model.checkpoint_directory.rmdir()
        offline.rename(model.checkpoint_directory)
        model.cache.clear()
        disk = model.restore_checkpoint(report['contexts'][0]['root'])
        if disk['forward_tokens'] != 0 or disk['counter_deltas']['loaded_file_bytes'] <= 0:
            raise AssertionError('Disk fallback did not perform strict measured restore')
        if state(model,report['contexts'][0]['root']) != report['contexts'][0]['initial']:
            raise AssertionError('Disk fallback changed the prefix frontier')
        report['disk_fallback'] = disk
        process = model.backend.process
        model.backend.close()
        report['native_exit_observed'] = process.poll() is not None
        if not report['native_exit_observed']: raise AssertionError('Native worker survived into fresh reference')
        model = None
        for index,entry in enumerate(report['contexts']):
            fresh = reference(profile,args.output,'fresh-'+str(index),entry['prefix']+entry['suffix'],
                [len(entry['prefix']),len(entry['suffix'])],args.generated)
            if (fresh['tokens'] != entry['tokens']
                    or hashlib.sha256(fresh['prompt_frontiers'][0]).hexdigest() != entry['initial']['frontier_sha256']
                    or hashlib.sha256(fresh['prompt_frontiers'][1]).hexdigest() != entry['suffix_frontier_sha256']
                    or hashlib.sha256(fresh['generated_frontiers'][-1]).hexdigest() != entry['expected']['frontier_sha256']):
                raise AssertionError('Host/disk state differs from fresh matching-schedule reference')
            report['checks'].append(dict(context=index,prefix='PASS',suffix='PASS',final='PASS',
                tokens='PASS',actual_outputs=len(fresh['tokens']),native_work=fresh['work']))
            publish(args.output/'result.json',report)
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL',error=str(error))
        raise
    finally:
        if model is not None:
            if offline.exists() and model.checkpoint_directory.exists() and not any(model.checkpoint_directory.iterdir()):
                model.checkpoint_directory.rmdir();offline.rename(model.checkpoint_directory)
            model.backend.close()
        publish(args.output/'result.json',report)


if __name__ == '__main__': main()
