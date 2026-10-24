"""Real CPU native optimizer jobs through PCC HTTP/operation/parent boundaries."""
from pathlib import Path
import argparse,copy,json,os,time,threading,subprocess,sys
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from experimental.native_training.make_llama_fixture import make_fixture
from experimental.native_training.test_optimizer_training import state_arrays
from experimental.native_training.llama_reference import read_gguf
from tools.pcc_lessons import LessonService,NativeRunner
from tools.pcc_optimizer_profiles import identity
from tools.pcc_native_training import execute_native,sha256

def run(executable,output):
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(root/'base.gguf');base=sha256(model)
    common=dict(model=str(model),base_sha256=base,executable=str(executable),devices=['CPU'],device_budget_bytes={'CPU':4<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20,gpu_uuids=[],sequence_profile='dense-llama-short-v1',memory_profile='allocator-v1',timeout_seconds=60)
    negative=0;results=[]
    for kind in ('adamw','muon'):
        directory=root/kind;directory.mkdir()
        fresh=dict(schema=1,title=kind+' fresh whole-update fixture',optimizer=kind,learning_rate=.001,optimizer_settings=dict(schema=1),state_policy='fresh',training_budget_seconds=1e-12)
        resume=dict(fresh,title=kind+' same-data resume fixture',state_policy='resume')
        fresh_id=identity(fresh);resume_id=identity(resume)
        config=dict(common,optimizer_profiles={fresh_id:fresh,resume_id:resume})
        runner=NativeRunner(config);service=LessonService(directory/'store','a'*64,lambda text:[int(c) for c in text],runner,max_tokens=512,max_steps=8,sequence_profile='dense-llama-short-v1')
        from serve.cow_http import make_server
        server=make_server(object(),api_key='private-fixture-key',training_service=service);thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_port)
            def call(path,data=None,key='private-fixture-key'):
                with urlopen(Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}),timeout=20) as response:return json.load(response)
            def wait(op):
                for _ in range(1000):
                    v=call('/cow/training/jobs/'+op)
                    if v['state'] not in ('queued','running'):return v
                    time.sleep(.02)
                raise TimeoutError('Original native job has not settled')
            capabilities=call('/cow/training/optimizer-profiles');assert {p['id'] for p in capabilities['profiles']}=={fresh_id,resume_id}
            assert all(p['may_stop_before_all_selected_lessons'] for p in capabilities['profiles'])
            try:call('/cow/training/optimizer-profiles',key='wrong')
            except HTTPError as error:assert error.code==401;error.close();negative+=1
            else:raise AssertionError('Unauthenticated profile discovery')
            lessons=[]
            for index,(prompt,answer) in enumerate([('01','234'),('12','345'),('23','456')]):
                lesson=dict(schema=1,title='Owned tiny fixture '+str(index),model_contract_id='a'*64,prompt=prompt,completion=answer,rationale='Synthetic CPU mechanics fixture, no capability claim',scope='owned-training',evidence=dict(method='host-verified-correction-v1',failed_result_sha256='b'*64,corrected_result_sha256='c'*64,verification_sha256='d'*64))
                lessons.append(call('/cow/training/lessons',dict(operation_id='lesson-'+str(index),lesson=lesson))['lesson_id'])
            request=dict(operation_id='first',lesson_ids=lessons[:2],mode='full',batch_size=1,steps=4,learning_rate=.001,rank=4,parent_job_id=None,optimizer_profile_id=fresh_id)
            accepted=call('/cow/training/jobs',request);first=wait('first');assert first['state']=='complete',first
            assert first['optimizer']==kind and first['optimizer_completed_steps']==1 and first['consumed_example_presentations']==1 and first['selected_lessons']==2
            assert first['training_budget']['stop_reason']=='training_budget'
            assert first['consumed_lesson_ids_this_job']==lessons[:1] and first['unconsumed_lesson_ids_this_job']==lessons[1:2]
            assert call('/cow/training/jobs',request)==accepted
            assert call('/cow/training/operations/first')==accepted
            first_record=service._read(service.root/'operations/first.json');first_candidate=first_record['candidate']
            continued=dict(request,operation_id='continued',parent_job_id='first',optimizer_profile_id=resume_id)
            call('/cow/training/jobs',continued);second=wait('continued');assert second['state']=='complete',second
            assert second['optimizer_completed_steps']==2 and second['example_cursor']==2 and second['optimizer_state_policy']=='resume'
            second_candidate=service._read(service.root/'operations/continued.json')['candidate']
            assert second_candidate['base_sha256']==first_candidate['candidate_sha256']
            # Independent uninterrupted native request must match both weights
            # and every saved optimizer moment across the two actual API jobs.
            examples=[service.lesson(i)['example'] for i in lessons[:2]]
            direct=dict(model=str(model),base_sha256=base,mode='full',devices=['CPU'],sequence_profile='dense-llama-short-v1',memory_profile='allocator-v1',device_budget_bytes={'CPU':4<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20,examples=examples,optimizer=kind,optimizer_settings=dict(schema=1),optimizer_state_mode='fresh',learning_rate=.001,batch_size=1,steps=2,deadline_seconds=60)
            straight=execute_native(executable,direct,directory/'straight',timeout=75)
            a,_=read_gguf(second_candidate['candidate_path']);b,_=read_gguf(straight['candidate_path']);assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
            am,aa=state_arrays(second_candidate['optimizer_state_path']);bm,ba=state_arrays(straight['optimizer_state_path']);assert aa.keys()==ba.keys() and all(np.array_equal(aa[k],ba[k]) for k in aa)
            # New curriculum explicitly retains learned weights, resets moments.
            new=dict(request,operation_id='new-block',lesson_ids=lessons[2:],parent_job_id='continued')
            client_args=[sys.executable,'-m','tools.cow_cli','--url',url,'--state-dir',str(directory/'cli-journal'),'lesson']
            environment=dict(os.environ,COW_API_KEY='private-fixture-key')
            discovered=subprocess.run([*client_args,'optimizers'],capture_output=True,text=True,env=environment,timeout=20)
            assert discovered.returncode==0 and {p['id'] for p in json.loads(discovered.stdout)['profiles']}=={fresh_id,resume_id},discovered.stderr
            command=[*client_args,'train',lessons[2],'--operation-id','new-block','--parent-job-id','continued','--mode','full','--learning-rate','.001','--steps','4','--batch-size','1','--optimizer-profile-id',fresh_id]
            submitted=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=20)
            assert submitted.returncode==0,submitted.stderr
            third=wait('new-block');assert third['state']=='complete',third
            duplicate=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=20)
            assert duplicate.returncode==0 and json.loads(duplicate.stdout)==json.loads(submitted.stdout),duplicate.stderr
            third_candidate=service._read(service.root/'operations/new-block.json')['candidate'];assert third_candidate['base_sha256']==second_candidate['candidate_sha256'] and third['optimizer_completed_steps']==1 and third['example_cursor']==1
            notes=directory/'notes.txt';notes.write_text('Tiny native CPU workflow fixture, not a model learning claim.\n')
            preserve_command=[*client_args,'preserve','new-block','--operation-id','preserve','--name','Tiny-'+kind,'--expected-sha256',third_candidate['candidate_sha256'],'--notes-file',str(notes),'--report-sha256','e'*64]
            saved=subprocess.run(preserve_command,capture_output=True,text=True,env=environment,timeout=20);assert saved.returncode==0,saved.stderr
            repeated=subprocess.run(preserve_command,capture_output=True,text=True,env=environment,timeout=20);assert repeated.returncode==0 and json.loads(repeated.stdout)==json.loads(saved.stdout)
            listed=call('/cow/training/exports');assert len(listed['exports'])==1 and listed['exports'][0]['weights_published'] is False
            export=service.root/'exports'/('Tiny-'+kind+'.gguf');assert sha256(export)==third_candidate['candidate_sha256'] and export.stat().st_ino==Path(third_candidate['candidate_path']).stat().st_ino
            manifest=service.root/'exports'/('Tiny-'+kind+'.json');saved_manifest=manifest.read_bytes();corrupt=json.loads(saved_manifest);corrupt['owner_notes']='Changed without provenance update';manifest.write_text(json.dumps(corrupt))
            try:call('/cow/training/exports')
            except HTTPError as error:assert error.code==400;error.close();negative+=1
            else:raise AssertionError('Corrupt export provenance accepted')
            manifest.write_bytes(saved_manifest)
            preserve_body=dict(operation_id='bad-export',job_id='new-block',name='Tiny-'+kind,expected_sha256=third_candidate['candidate_sha256'],notes=notes.read_text(),report_sha256='e'*64)
            for change in [dict(name='../escape'),dict(expected_sha256='f'*64),dict(notes='Conflicting immutable notes')]:
                try:call('/cow/training/exports',dict(preserve_body,operation_id='bad-export-'+str(negative),**change))
                except HTTPError as error:assert error.code==400;error.close();negative+=1
                else:raise AssertionError('Unsafe/conflicting export admitted')
            for change in [dict(optimizer_profile_id=None),dict(optimizer_profile_id='f'*64),dict(learning_rate=.002),dict(mode='lora'),dict(executable='/client/forbidden'),dict(optimizer_profile_id=resume_id,parent_job_id=None)]:
                try:call('/cow/training/jobs',dict(request,operation_id='refused-'+str(negative),**change))
                except HTTPError as error:assert error.code==400;error.close();negative+=1
                else:raise AssertionError('Unregistered profile/resource/state variation admitted')
            # Dataset mismatch cannot silently reuse moments; preserve failure.
            changed=dict(request,operation_id='wrong-data-resume',lesson_ids=lessons[2:],parent_job_id='continued',optimizer_profile_id=resume_id)
            call('/cow/training/jobs',changed);failed=wait('wrong-data-resume');assert failed['state']=='failed' and 'candidate' not in failed
            assert not (service.root/'runs/wrong-data-resume').exists();negative+=1
            results.append(dict(optimizer=kind,status='PASS',actual_http_native_jobs=3,global_resume_steps=second['optimizer_completed_steps'],completed_resume_weights_and_moments_bit_exact=True,new_block_keeps_weights_and_resets_moments=True,partial_budget_coverage_exposed=True,duplicate_operation_not_retrained=True,actual_cli_discovery_and_new_block=True,private_named_gguf_preservation_idempotent=True,promoted=False))
        finally:server.shutdown();server.server_close();thread.join();service.close()
    assert sha256(model)==base
    result=dict(status='PASS',scope='Real tiny CPU native AdamW/Muon jobs via existing authenticated PCC training service, not model learning',cases=results,negative_cases=negative,weights_published=False,promoted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.executable,a.output)))
