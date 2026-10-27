"""Owner-bound high-rank LoRA through authenticated PCC HTTP and native jobs."""
from pathlib import Path
import argparse,json,threading,time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from tools.pcc_lessons import LessonService,NativeRunner
from tools.pcc_native_training import sha256
from tools.pcc_optimizer_profiles import identity
from experimental.native_training.make_llama_fixture import make_fixture


def run(executable,output,recovery=False):
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(root/'base.gguf');rows=[]
    for kind in ('adamw','muon'):
        profile=dict(schema=4,title='Owned high-rank fixture',optimizer=kind,mode='lora',rank=4096,
            learning_rate=.001,optimizer_settings=dict(schema=1),state_policy='fresh',
            training_budget_seconds=None,lora_update_space='factorwise-v1',lora_rank_profile='extended-4096-v1')
        key=identity(profile)
        config=dict(model=str(model),base_sha256=sha256(model),executable=str(executable),devices=['CPU'],
            device_budget_bytes={'CPU':4<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20,
            gpu_uuids=[],sequence_profile='dense-llama-short-v1',memory_profile='allocator-v1',
            timeout_seconds=60,optimizer_profiles={key:profile})
        if recovery:config['completion_recovery']=dict(schema=1,source_closure={str(executable):sha256(executable)})
        runner=NativeRunner(config)
        service=LessonService(root/kind,'a'*64,lambda text:[int(c) for c in text],runner,
                              max_tokens=512,max_steps=8,sequence_profile='dense-llama-short-v1')
        from serve.cow_http import make_server
        server=make_server(object(),api_key='fixture-key',training_service=service)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            def call(path,body=None,key='fixture-key'):
                req=Request('http://127.0.0.1:'+str(server.server_port)+path,
                    data=None if body is None else json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with urlopen(req,timeout=20) as response:return json.load(response)
            lesson=dict(schema=1,title='Owned fixture',model_contract_id='a'*64,prompt='01',completion='234',
                rationale='Generated mechanics fixture',scope='owned-training',
                evidence=dict(method='host-verified-correction-v1',failed_result_sha256='b'*64,
                              corrected_result_sha256='c'*64,verification_sha256='d'*64))
            lesson_id=call('/cow/training/lessons',dict(operation_id='lesson',lesson=lesson))['lesson_id']
            request=dict(operation_id='first',lesson_ids=[lesson_id],mode='lora',rank=4096,batch_size=1,
                         steps=1,learning_rate=.001,parent_job_id=None,optimizer_profile_id=key)
            accepted=call('/cow/training/jobs',request)
            deadline=time.monotonic()+60
            while True:
                job=call('/cow/training/jobs/first')
                if job['state'] not in ('queued','running'):break
                if time.monotonic()>deadline:raise TimeoutError('Original job not settled')
                time.sleep(.02)
            assert job['state']=='complete',job
            record=service._read(service.root/'operations/first.json')
            checkpoint=record['candidate']['receipt']['optimizer_checkpoint']
            assert checkpoint['completed_steps']==1
            if recovery:
                assert 'completion_recovery' in record['candidate']
                gate=json.loads((service.root/'runs/first/completion-gate.json').read_bytes())
                assert gate['passed'] is True and gate['absolute_error']<=gate['absolute_bound']==.0002
            assert any(g['optimizer']=='muon' for g in checkpoint['groups'])==(kind=='muon')
            assert call('/cow/training/jobs',request)==accepted
            negative=0
            for change in (dict(optimizer_profile_id='f'*64),dict(rank=2048),dict(optimizer_profile_id=None)):
                try:call('/cow/training/jobs',dict(request,operation_id='refused-'+str(negative),**change))
                except HTTPError as error:assert error.code==400;error.close();negative+=1
                else:raise AssertionError('Unbound high-rank request admitted')
            rows.append(dict(optimizer=kind,status='PASS',rank=4096,completed_updates=1,
                             replay_reused_original_operation=True,refused_requests=negative))
        finally:server.shutdown();thread.join();server.server_close()
    result=dict(status='PASS',completion_recovery=recovery,scope='Generated tiny-model HTTP/native integration, not MiniCPM learning',cases=rows)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--output',required=True)
    p.add_argument('--recovery',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.executable,a.output,a.recovery)))
