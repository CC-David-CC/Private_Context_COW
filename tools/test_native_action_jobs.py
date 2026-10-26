"""Actual longer-context CPU teaching jobs through PCC HTTP and completion gates.

Generated tiny model and synthetic digit-token teaching fixtures only. This does
not assert S3 lesson approval, MiniCPM improvement or a production model identity.
"""
from pathlib import Path
import argparse,json,math,os,threading,time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from experimental.native_training.make_llama_fixture import make_fixture
from tools.pcc_native_training import sha256
from tools.pcc_lessons import LessonService,NativeRunner
from tools.pcc_optimizer_profiles import identity
from tools.pcc_candidate_parent import resolve_parent
from tools.pcc_cow_profile import bind_dense_profile,bind_dense_adapter
from experimental.native_training.test_llama_action_profile import tensor_arrays

PROFILE='dense-llama-action-8k-v1'

def run(executable,output,*,profile=PROFILE):
    PROFILE=profile
    executable=Path(executable).resolve(strict=True);root=Path(output).absolute()
    root.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(root/'tiny.gguf',dtype='f32',layers=1,context_length=8192);base=sha256(model)
    cases=[]
    for mode,kind in [('full','adamw'),('full','muon'),('lora','adamw')]:
        folder=root/(mode+'-'+kind);folder.mkdir(mode=0o700)
        fresh=dict(schema=2,title='Synthetic long-context native fixture',mode=mode,rank=1 if mode=='lora' else None,
            optimizer=kind,learning_rate=.001,optimizer_settings=dict(schema=1),state_policy='fresh',training_budget_seconds=None)
        resume=dict(fresh,state_policy='resume');fresh_id=identity(fresh);resume_id=identity(resume)
        config=dict(model=str(model),base_sha256=base,executable=str(executable),devices=['CPU'],gpu_uuids=[],
            device_budget_bytes={'CPU':12<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20,
            sequence_profile=PROFILE,memory_profile='allocator-v1',timeout_seconds=120,
            optimizer_profiles={fresh_id:fresh,resume_id:resume},
            completion_recovery=dict(schema=1,source_closure={str(executable):sha256(executable)}))
        runner=NativeRunner(config);service=LessonService(folder/'service','a'*64,lambda s:[int(c) for c in s],
            runner,max_tokens=8192,max_steps=4,sequence_profile=PROFILE)
        from serve.cow_http import make_server
        server=make_server(object(),api_key='controlled-native-action-fixture',training_service=service)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_port)
            def call(path,value=None):
                request=Request(url+path,data=None if value is None else json.dumps(value).encode(),
                    headers={'Authorization':'Bearer controlled-native-action-fixture','Content-Type':'application/json'})
                with urlopen(request,timeout=30)as response:return json.load(response)
            def wait(name):
                deadline=time.monotonic()+180
                while time.monotonic()<deadline:
                    value=call('/cow/training/jobs/'+name)
                    if value['state']not in ('queued','running'):return value
                    time.sleep(.01)
                raise RuntimeError('Original job still unresolved; inspect it, do not redispatch')
            lessons=[]
            for index in range(3):
                lesson=dict(schema=2,title='Synthetic digit teaching fixture '+str(index),model_contract_id='a'*64,
                    target_policy='actions_answer',scope='owned-training',
                    evidence=dict(method='sss-verified-episode-v1',archive_sha256='b'*64,verification_sha256='c'*64,
                        renderer_sha256='d'*64,tokenizer_sha256='e'*64),
                    segments=[dict(role='system',kind='instruction',rendered=str(index)*513,verified=False,scope='practice'),
                              dict(role='user',kind='question',rendered='12',verified=False,scope='practice'),
                              dict(role='assistant',kind='action',rendered='34',verified=True,scope='practice')])
                lessons.append(call('/cow/training/lessons',dict(operation_id='lesson-'+str(index),lesson=lesson))['lesson_id'])
            assert all(len(service.lesson(i)['example']['tokens'])>512 for i in lessons)
            request=dict(operation_id='first',lesson_ids=lessons[:2],mode=mode,rank=1,batch_size=2,steps=1,
                learning_rate=.001,parent_job_id=None,optimizer_profile_id=fresh_id)
            accepted=call('/cow/training/jobs',request);first=wait('first');assert first['state']=='complete',first
            operation=service._read(service.root/'operations/first.json');candidate=operation['candidate']
            assert candidate['receipt']['sequence_profile']==PROFILE
            gradients=candidate['receipt']['steps'][0]['gradients']
            assert all(math.isfinite(row['l2'])for row in gradients)and any(row['l2']>0 for row in gradients)
            first_arrays=tensor_arrays(candidate['candidate_path'])
            if mode=='full':
                base_arrays=tensor_arrays(model)
                assert first_arrays.keys()==base_arrays.keys()and any((first_arrays[name]!=base_arrays[name]).any()for name in base_arrays)
            else:
                assert any((values!=0).any()for name,values in first_arrays.items()if name.endswith('.lora_b'))
            gate_path=Path(candidate['candidate_path']).parent/'completion-gate.json'
            gate=json.loads(gate_path.read_bytes());assert gate['gate_id']==candidate['completion_recovery']['gate_id']
            assert first['optimizer_completed_steps']==1 and first['consumed_example_presentations']==2
            before={p:p.read_bytes() for p in (service.root/'operations').glob('*.json')}
            assert call('/cow/training/jobs',request)==accepted
            assert before=={p:p.read_bytes() for p in (service.root/'operations').glob('*.json')}
            # Resolve the actual long-example optimizer stream using the same
            # owner parent resolver used by the learning driver.
            template=dict(backend='llama-dense-llama.cpp',native_build=dict(sha256={},lora_binding='llama-adapter-per-lease-v1'),
                hardware=dict(gpu_uuid='GPU-identity-fixture'),execution=dict(state_format='llama-dense-frontier-v1',kv_dtype='FP32',
                    device_binding='cuda-single-device-v1',environment=dict(CUDA_VISIBLE_DEVICES='0'),GPU=[0]))
            profile=folder/'identity-template.json';profile.write_text(json.dumps(template))
            bound=bind_dense_profile(template,model_path=Path(candidate['candidate_path'])if mode=='full'else model,
                model_sha256=candidate['candidate_sha256']if mode=='full'else base,gpu_index=0,gpu_uuid='GPU-identity-fixture')
            adapters=[]
            if mode=='lora':
                bound=bind_dense_adapter(bound,adapter_path=candidate['candidate_path'],adapter_sha256=candidate['candidate_sha256'])
                adapters=[dict(sha256=candidate['candidate_sha256'],scale=1.)]
            reference=dict(binding_kind='measured-composite',base_model_sha256=candidate['candidate_sha256']if mode=='full'else base,
                candidate_model_contract_id=bound['model_contract_id'],candidate_adapters=adapters)
            version=Path(candidate['candidate_path']).parent/'versions'/(candidate['candidate_sha256']+'.json')
            resolved=resolve_parent(reference,version,profile,artifact_roots=[root],state_policy='resume',
                base_model=model if mode=='lora'else None,examples=[service.lesson(i)['example']for i in lessons[:2]])
            assert resolved['optimizer_state_fields']['example_cursor']==2
            continued=dict(request,operation_id='continued',parent_job_id='first',optimizer_profile_id=resume_id)
            call('/cow/training/jobs',continued);second=wait('continued');assert second['state']=='complete',second
            assert second['optimizer_completed_steps']==2 and second['example_cursor']==4
            try:
                call('/cow/training/jobs',dict(continued,operation_id='wrong-data',lesson_ids=lessons[2:]))
            except HTTPError as error:assert error.code==400;error.close()
            else:raise AssertionError('Changed long-example stream silently reused optimizer state')
            second_candidate=service._read(service.root/'operations/continued.json')['candidate']
            second_arrays=tensor_arrays(second_candidate['candidate_path'])
            assert first_arrays.keys()==second_arrays.keys()and any((first_arrays[name]!=second_arrays[name]).any()for name in first_arrays)
            assert second_candidate['receipt']['parent_adapter_sha256']==candidate['candidate_sha256'] if mode=='lora'else second_candidate['base_sha256']==candidate['candidate_sha256']
            assert second_candidate['completion_recovery']['gate_id']
            cases.append(dict(mode=mode,optimizer=kind,status='PASS',actual_native_updates=2,example_presentations=4,
                input_tokens=len(service.lesson(lessons[0])['example']['tokens']),
                schema2_action_mask=True,automatic_independent_reload=True,original_completion_intent=True,
                duplicate_operation_did_not_retrain=True,long_parent_state_resolved=True,changed_stream_resume_refused=True,
                nonzero_gradients_observed=True,saved_weights_changed=True,continued_weights_changed=True))
        finally:
            server.shutdown();server.server_close();thread.join();service.close()
    assert sha256(model)==base
    result=dict(schema=1,status='PASS',scope='Synthetic digit-token tiny CPU jobs through actual PCC HTTP/lesson/native/reload boundaries; not model learning',
        profile=PROFILE,cases=cases,native_executable_sha256=sha256(executable),test_source_sha256=sha256(__file__),
        source_base_unchanged=True,gpu_used=False,promoted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.executable,args.output)))
