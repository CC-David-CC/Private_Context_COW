"""Owner-run Wen code-completion study with fresh native inference and held-out checks.

Uses the same lesson/job controller as BW, without starting an inference server.
This is a bounded generated-code benchmark, not general agent qualification.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
import math
import os
import select
import signal
import struct
import subprocess
import time
from tools.pcc_lessons import LessonService,NativeRunner,digest
from tools.pcc_native_training import sha256
from tools.pcc_learning_cases import cases,check
from tools.strata_tokenizer import Tokenizer

def receive(proc,count,timeout=120):
    result=bytearray();deadline=time.monotonic()+timeout
    while len(result)<count:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Owned native oracle timed out')
        ready,_,_=select.select([proc.stdout],[],[],remaining)
        if not ready:raise TimeoutError('Owned native oracle output timed out')
        part=os.read(proc.stdout.fileno(),count-len(result))
        if not part:raise RuntimeError('Native oracle exited; inspect its private log')
        result.extend(part)
    return bytes(result)

@contextmanager
def oracle_process(binary,model,gpu_uuid,log,adapter=None):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    selected=[p for line in rows.splitlines() if (p:=[v.strip() for v in line.split(',')])[1]==gpu_uuid]
    if len(selected)!=1 or int(selected[0][2])<48*1024 or int(selected[0][3])!=0:
        raise RuntimeError('Evaluation GPU lacks the admitted idle headroom')
    locks=Path.home()/'.local/state/context-cow';locks.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (locks/('gpu-'+selected[0][0]+'.lock')).open('a+') as lease,Path(log).open('wb') as err:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        environment=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu_uuid,
            GGML_CUDA_F32_ACTIVATIONS='1',COW_QWEN35_F32_ATTENTION='1',NVIDIA_TF32_OVERRIDE='0',GGML_NO_BACKTRACE='1')
        argv=[str(binary),str(model),'256','128','8','CUDA0']
        if adapter is not None:argv+=['--lora',str(adapter),'1']
        proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=environment,start_new_session=True,pass_fds=(lease.fileno(),))
        try:yield proc
        finally:
            if proc.poll() is None:
                proc.stdin.close()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def completion(proc,tok,prompt,maximum=24):
    import numpy as np  # Binary-output decoding only, never the training runtime.
    prefix=tok.encode(prompt);generated=[]
    for _ in range(maximum):
        tokens=prefix+generated
        packet=dict(schema=1,spans=[dict(kind='text',tokens=tokens)],chunks=[[0,len(tokens)]],embedding_paths={})
        proc.stdin.write(('M '+json.dumps(packet)+'\n').encode());proc.stdin.flush()
        vocab,work,position=struct.unpack('<iQi',receive(proc,16))
        if work!=len(tokens) or position!=len(tokens)-1:raise ValueError('Oracle did not prove fresh full-prefill work')
        raw=receive(proc,vocab*4);logits=np.frombuffer(raw,dtype='<f4')
        if not np.isfinite(logits).all():raise ValueError('Nonfinite independent oracle logits')
        generated.append(int(logits.argmax()))
        text=tok.decode(generated)
        if '\n' in text:return text.split('\n',1)[0]
        if '<|im_end|>' in text or '<|endoftext|>' in text:return text
    return tok.decode(generated)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config',type=Path,required=True)
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--evaluation-gpu-uuid',required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--examples',type=int,default=1000)
    parser.add_argument('--checkpoints',default='1,32,256,1000')
    parser.add_argument('--rank',type=int,default=8);parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--learning-rate',type=float,default=.001)
    args=parser.parse_args();root=args.workdir.absolute();root.mkdir(mode=0o700,parents=True,exist_ok=False)
    service=None
    def phase(name,**extra):
        value=dict(phase=name,time=time.time(),**extra);temporary=root/'phase.tmp';temporary.write_text(json.dumps(value)+'\n');temporary.replace(root/'phase.json')
        print(json.dumps(value),flush=True)
    try:
        config=json.loads(args.training_config.read_bytes());runner_config=config['runner']
        checkpoints=[int(v) for v in args.checkpoints.split(',')]
        if not 1<=args.examples<=1000 or checkpoints!=sorted(set(checkpoints)) or checkpoints[-1]!=args.examples or checkpoints[0]<1:
            raise ValueError('Declare increasing checkpoints ending at the requested owned-example count')
        if args.batch_size<1:raise ValueError('Positive batch size required')
        phase('freeze-data-and-policy')
        tokenizer=Tokenizer.from_gguf(runner_config['model']);pool=cases(count=2000)
        training=[];evaluation=[r for r in pool if r['split'] in ('held_out','preservation')];blind=[r for r in pool if r['split']=='blind_test']
        for row in pool:
            if row['split']!='train':continue
            prefix=tokenizer.encode(row['prompt']);full=tokenizer.encode(row['prompt']+row['completion'])
            if (full[:len(prefix)]==prefix and 2<=len(full)-1<=config['max_tokens'] and len(training)<args.examples):training.append(row)
        if len(training)!=args.examples or not all(check(r,r['completion'])['passed'] for r in training+evaluation+blind):
            raise ValueError('Owned labels failed independent checks or admitted token bounds')
        probes=training[:64];observed=probes+evaluation
        policy=dict(repair='initial training-pool failure must pass',held_out_minimum_accuracy=.90,
            held_out_minimum_repaired_error_fraction=.25,preservation='all baseline-passing controls must remain passing',
            preservation_minimum_baseline_controls=8,
            candidate_selection='eligible checkpoints: most held-out passes, then fewer examples; otherwise final candidate for diagnosis only',
            blind_final='untouched until selection; at least90% and no fewer passes than original model',
            generation='greedy return-expression completion, newline stop,24-token cap; fresh full-prefill oracle',
            no_automatic_promotion=True)
        manifest=dict(policy=policy,training_sha256=digest(training),evaluation_sha256=digest(evaluation),blind_sha256=digest(blind),
            training_count=len(training),evaluation_count=len(evaluation),checkpoints=checkpoints,
            native_executable_sha256=sha256(runner_config['executable']),oracle_sha256=sha256(args.oracle),
            base_sha256=runner_config['base_sha256'],rank=args.rank,batch_size=args.batch_size,learning_rate=args.learning_rate)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (root/'data-private.json').write_text(json.dumps(dict(training=training,evaluation=evaluation,blind=blind),indent=2)+'\n')
        def evaluate(label,adapter=None,selected=None):
            records=[];phase('fresh-native-evaluation',checkpoint=label)
            selected=observed if selected is None else selected
            with oracle_process(args.oracle,runner_config['model'],args.evaluation_gpu_uuid,root/(label+'-oracle.log'),adapter) as proc:
                for index,row in enumerate(selected):
                    generated=completion(proc,tokenizer,row['prompt']);verdict=check(row,generated)
                    records.append(dict(id=row['id'],split=row['split'],completion=generated,verdict=verdict))
                    if index%16==15:phase('fresh-native-evaluation',checkpoint=label,cases=index+1,total=len(selected))
            (root/(label+'-private.json')).write_text(json.dumps(records,indent=2)+'\n')
            return records
        baseline=evaluate('baseline')
        failures=[r for r in baseline if r['split']=='train' and not r['verdict']['passed']]
        if not failures:
            phase('baseline-no-observed-training-failure');(root/'result.json').write_text(json.dumps(dict(status='NO_OBSERVED_TRAINING_FAILURE',trained=False,promoted=False))+'\n')
            return 4
        anchor=next(r for r in training if r['id']==failures[0]['id'])
        training=[anchor]+[r for r in training if r['id']!=anchor['id']]
        (root/'training-order-private.json').write_text(json.dumps(training,indent=2)+'\n')
        contract=digest(dict(owner_benchmark='signed-ceiling-v1',base_sha256=runner_config['base_sha256']))
        service=LessonService(root/'lesson-store',contract,tokenizer.encode,NativeRunner(runner_config),max_tokens=config['max_tokens'],max_steps=config['max_steps'])
        phase('admit-independently-checked-training-variations',count=len(training))
        ids=[];failure_hash=digest(failures[0]);checker_hash=sha256(Path(__file__).with_name('pcc_learning_cases.py'))
        for index,row in enumerate(training):
            lesson=dict(schema=1,title='Signed ceiling division variation '+str(index),model_contract_id=contract,
                prompt=row['prompt'],completion=row['completion'],scope='owned-training',
                rationale='Owned variation of a captured native completion failure; exact signed-integer checks validate the correction. Evaluation cases are excluded.',
                evidence=dict(method='host-verified-correction-v1',failed_result_sha256=failure_hash,
                    corrected_result_sha256=digest(dict(prompt=row['prompt'],completion=row['completion'])),
                    verification_sha256=digest(dict(case=row['id'],checker_sha256=checker_hash,verdict=check(row,row['completion'])))))
            ids.append(service.submit('/cow/training/lessons',dict(operation_id='lesson-'+row['id'],lesson=lesson))['lesson_id'])
        base_by_id={r['id']:r for r in baseline};held=[r for r in baseline if r['split']=='held_out'];controls=[r for r in baseline if r['split']=='preservation' and r['verdict']['passed']]
        baseline_held=sum(r['verdict']['passed'] for r in held);required_gain=max(1,math.ceil((len(held)-baseline_held)*.25))
        results=[];prior=None;start=0
        for count in checkpoints:
            selected=ids[start:count];batch=min(args.batch_size,len(selected));steps=(len(selected)+batch-1)//batch;operation='quality-through-'+str(count)
            request=dict(operation_id=operation,lesson_ids=selected,mode='lora',batch_size=batch,steps=steps,learning_rate=args.learning_rate,rank=args.rank,parent_job_id=prior)
            phase('native-standard-lora-update',through_examples=count,new_examples=len(selected),steps=steps)
            service.submit('/cow/training/jobs',request)
            deadline=time.monotonic()+3700
            while True:
                job=service.job(operation)
                if job['state'] not in ('queued','running'):break
                if time.monotonic()>deadline:raise TimeoutError('Native update has no terminal proof; reconcile its original run')
                time.sleep(3)
            if job['state']!='complete':raise RuntimeError('Native quality update failed; preserve original job evidence')
            candidate=service._read(service.root/'operations'/(operation+'.json'))['candidate']
            records=evaluate('after-'+str(count),candidate['candidate_path']);now={r['id']:r for r in records}
            held_pass=sum(now[r['id']]['verdict']['passed'] for r in held)
            gates=dict(repairs=now[anchor['id']]['verdict']['passed'],held_out=(held_pass/len(held)>=.9 and held_pass>=baseline_held+required_gain),
                preservation=(len(controls)>=8 and all(now[r['id']]['verdict']['passed'] for r in controls)))
            row=dict(examples=count,job_id=operation,candidate_sha256=candidate['candidate_sha256'],held_out_passed=held_pass,held_out_total=len(held),
                baseline_held_out_passed=baseline_held,preserved=sum(now[r['id']]['verdict']['passed'] for r in controls),preservation_baseline_passed=len(controls),
                original_failure_repaired=gates['repairs'],gates=gates,learning_pass=all(gates.values()),promoted=False)
            results.append(row);(root/'checkpoints.json').write_text(json.dumps(results,indent=2)+'\n');phase('checkpoint-evaluated',**row)
            prior=operation;start=count
        unchanged=sha256(runner_config['model'])==runner_config['base_sha256']
        if not unchanged:raise ValueError('Original model changed during quality study')
        eligible=[r for r in results if r['learning_pass']]
        selected=max(eligible,key=lambda r:(r['held_out_passed'],-r['examples'])) if eligible else results[-1]
        candidate=service._read(service.root/'operations'/(selected['job_id']+'.json'))['candidate']
        original_blind=evaluate('blind-original',selected=blind)
        candidate_blind=evaluate('blind-selected',candidate['candidate_path'],selected=blind)
        baseline_blind_passed=sum(r['verdict']['passed'] for r in original_blind)
        blind_passed=sum(r['verdict']['passed'] for r in candidate_blind)
        blind_gate=blind_passed/len(blind)>=.9 and blind_passed>=baseline_blind_passed
        passed=bool(eligible) and blind_gate
        result=dict(status='BOUNDED_LEARNING_PASS' if passed else 'NO_QUALIFIED_GAIN',promoted=False,
            base_unchanged=unchanged,manifest=manifest,original_failure_id=anchor['id'],checkpoints=results,
            selected_candidate_sha256=selected['candidate_sha256'],selected_examples=selected['examples'],
            blind=dict(baseline_passed=baseline_blind_passed,candidate_passed=blind_passed,total=len(blind),passed=blind_gate),
            qualification='generated return-expression benchmark only; not general BW task competence or sustained load')
        (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');phase('complete',status=result['status']);return 0 if passed else 3
    except BaseException as error:
        phase('failed',error=type(error).__name__+': '+str(error));raise
    finally:
        if service is not None:service.close()

if __name__=='__main__':raise SystemExit(main())
