"""Real process-separated fixture checks; independent CPU state/frontier oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from fixture_reference import byte_at

SEED=0x4d465f434f575f31
def expected_frontier(ids):
    r=SEED&65535
    for token in ids:r=(r*257+token+1)&65535
    return r,[(r^(k*197))&65535 for k in range(64)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--devices',default='0')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    devices=list(map(int,a.devices.split(',')));results=[];processes=[]
    def check(name,value,**detail):
        results.append(dict(id=name,status='PASS' if value else 'FAIL',**detail))
        if not value:raise AssertionError(name)
    def start():
        proc=subprocess.Popen([str(a.binary),'--gpu',a.devices,'--capacity','12288','--seed',hex(SEED)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(a.output/('native-'+str(len(processes))+'.stderr')).open('w'),text=True)
        processes.append(proc);return proc
    def call(proc,line):
        proc.stdin.write(line+'\n');proc.stdin.flush();raw=proc.stdout.readline()
        with (a.output/'protocol.jsonl').open('a') as out:out.write(json.dumps(dict(pid=proc.pid,command=line,response=raw))+'\n')
        value=json.loads(raw)
        if not value['ok']:raise RuntimeError(value['error'])
        return value
    status='FAIL'
    try:
        proc=start();call(proc,'ready');call(proc,'reset');ids=[i%64 for i in range(128)]
        call(proc,'consume 128 '+' '.join(map(str,ids)))
        check('MF-06-INDEPENDENT-FRONTIER',call(proc,'frontier')['logits']==expected_frontier(ids)[1])
        image=call(proc,'capture')['image'];saved=a.output/'prefix.state';call(proc,f'serialize {image} {saved}')
        raw=saved.read_bytes();offset=0
        def unpack(fmt):
            nonlocal offset
            v=struct.unpack_from('<'+fmt,raw,offset);offset+=struct.calcsize('<'+fmt);return v
        check('MF-06-IMAGE-HEADER',raw[:4]==b'CFB1');offset=4
        seed,capacity,parts,consumed=unpack('QQIQ')
        check('MF-06-IMAGE-IDENTITY',(seed,capacity,parts,consumed)==(SEED,12288,len(devices),128))
        check('MF-06-IMAGE-TOPOLOGY',list(unpack('i'*parts))==devices)
        check('MF-06-IMAGE-HISTORY',list(unpack('i'*consumed))==ids)
        for layer in range(parts):
            extent,r=unpack('QI');payload=raw[offset:offset+extent];offset+=extent
            expected=bytearray(extent)
            for pos,t in enumerate(ids):
                for b in range(32):expected[pos*32+b]=byte_at(SEED,0,layer,t,b,pos*32+b)
            check('MF-06-INDEPENDENT-PART-'+str(layer),payload==expected and r==expected_frontier(ids)[0],extent=extent)
        check('MF-06-IMAGE-COMPLETE',offset==len(raw),saved_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        call(proc,'consume 4 3 4 5 6');call(proc,f'restore {image}')
        check('MF-06-HOT-REWIND',call(proc,'frontier')['logits']==expected_frontier(ids)[1])
        first_pid=proc.pid;call(proc,'quit');proc.wait(timeout=20)
        check('MF-06-PRODUCER-EXIT',proc.returncode==0)
        proc=start();call(proc,'ready');before=call(proc,'counters')['forward_tokens'];reply=call(proc,f'import {saved}')
        check('MF-06-NEW-PROCESS',proc.pid!=first_pid,pids=[first_pid,proc.pid])
        check('MF-06-STRICT-ZERO-PREFIX-WORK',call(proc,'counters')['forward_tokens']==before)
        check('MF-06-RESTORED-FRONTIER',reply['history']==ids and reply['logits']==expected_frontier(ids)[1])
        for label,suffix in [('A',[3,4,5,6]),('B',[7,8,9,10])]:
            call(proc,f'import {saved}');call(proc,'consume 4 '+' '.join(map(str,suffix)))
            check('MF-06-BRANCH-'+label,call(proc,'frontier')['logits']==expected_frontier(ids+suffix)[1])
        check('MF-06-PARENT-BYTES-UNCHANGED',saved.read_bytes()==raw)
        call(proc,'quit');proc.wait(timeout=20);check('MF-06-CONSUMER-EXIT',proc.returncode==0);status='PASS'
    except Exception as e:results.append(dict(id='MF-06-EXCEPTION',status='FAIL',error=repr(e)))
    finally:
        for proc in processes:
            if proc.poll() is None:proc.kill();proc.wait(timeout=20)
        report=dict(status=status,scope='actual GPU fixture memory; not model inference',devices=devices,seed=SEED,
            tests=results,failures=sum(x['status']=='FAIL' for x in results),skips=0,process_exit_codes=[p.returncode for p in processes],time=time.time())
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return status!='PASS'
if __name__=='__main__':raise SystemExit(main())
