"""Read progress without interpreting a missing published PID as a startup stall."""
import argparse
import json
from pathlib import Path


def tail_records(path):
    if not path.exists(): return []
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-65536))
        lines=stream.read().splitlines()
    records=[]
    for line in lines:
        try: records.append(json.loads(line))
        except (ValueError,UnicodeDecodeError): pass
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    worker=json.loads((root/'worker.json').read_text())
    outer=worker.get('phase');record=dict(worker_status=worker['status'],case=outer,
        worker_pid=worker['pid'],elapsed_seconds=worker.get('elapsed_seconds'),
        completed_phases=worker['phases'])
    phases=tail_records(root/(str(outer)+'.stdout'))
    phases=[item for item in phases if 'phase' in item]
    if phases:
        current=phases[-1];record['phase']=current['phase'];record['python_pid']=current['pid']
        native=tail_records(root/str(outer)/(current['phase']+'-native.jsonl'))
        for item in reversed(native):
            stats=item.get('response',item.get('counters',{}))
            if 'forward_tokens' in stats:
                record.update(native_pid=item['pid'],last_native_event_time=item.get('time'),
                    consumed_position=stats.get('consumed'),forward_tokens=stats['forward_tokens'],
                    verifier_decode_tokens=stats.get('decode_timed_tokens'),
                    integrity_mismatches=stats.get('integrity_mismatches'),native_closed=item.get('closed',False))
                break
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
