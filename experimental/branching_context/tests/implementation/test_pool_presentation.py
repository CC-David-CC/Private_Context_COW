"""Presentation-bound coordinator intents; synthetic scheduler, not GPU evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.store import Conflict, IntegrityError, InvalidOperation, _json_bytes
import test_pool_coordinator as pool_fixtures


def request(kind, namespace, *, messages=(), output='a' * 32, renderer='b' * 64):
    return dict(schema=1, namespace_id=namespace, kind=kind,
                messages=[dict(message) for message in messages],
                renderer_id=renderer,
                output_message_id=None if kind == 'root' else output)


class PresentationCoordinatorTests(unittest.TestCase):
    def start(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        ids=(0,); trace=[]; models={}
        config=PoolConfig(gpus=ids,ram_bytes=4096,disk_bytes=16384,min_available_bytes=0)
        def factory(gpu):
            model=pool_fixtures.Model(temp.name,gpu,ids,trace,__import__('threading').Event())
            model.gate.set()
            # Synthetic implementation follows the real optional plumbing API.
            original=model.save_prepared_branch
            def save(name,revision,prepared,operation,*,presentation=None):
                model.checkpoint_directory.joinpath(prepared.checkpoint.checksum+'.state').write_bytes(
                    prepared.checkpoint.blob+b'\0\0\x80?')
                model.checkpoint_directory.joinpath(prepared.snapshot+'.json').write_bytes(prepared.envelope)
                if presentation is None:
                    head=model.store.move(name,revision,prepared.snapshot,operation)
                else:
                    head=model.store.move(name,revision,prepared.snapshot,operation,presentation=presentation)
                return dict(snapshot=prepared.snapshot,branch=__import__('dataclasses').asdict(head),
                    payload_sha256=prepared.checkpoint.checksum,model_contract_id='m',execution_profile_id='e')
            model.save_prepared_branch=save
            # Correct the synthetic generated controller to a real active/stopped scope.
            old_run=model.run_greedy_request
            def run(sid,input_ids,count,*,stream,on_token):
                model.gate.wait(5)
                model.stats['forward_tokens'] += len(input_ids)+count
                outputs=list(range(count))
                for token in outputs: on_token(token)
                target=model.store.append(sid,tuple(input_ids)+tuple(outputs),
                    dict(status='active' if stream == 'active-answer' else 'stopped',
                         stream_key=hashlib.sha256(stream.encode()).hexdigest(),
                         policy={'kind':'greedy'},accepted_in_scope=count))
                model.image(target)
                return dict(snapshot=target,tokens=outputs,request_seconds=.001,generation_seconds=.001)
            model.run_greedy_request=run
            models[gpu]=model
            return model
        coordinator=ContextCoordinator(temp.name,config,factory)
        self.addCleanup(coordinator.close); coordinator.ready(5)
        return coordinator,models

    def test_create_binds_detached_root_presentation_and_namespace(self):
        c,models=self.start(); namespace=c.store.create_namespace('test chat','namespace-op').namespace_id
        messages=[dict(id='2'*32,role='user',content='hello')]
        value=request('root',namespace,messages=messages,output=None)
        create=c.create('chat',(10,11),'root-op',presentation=value)
        messages[0]['content']='mutated after submit'
        result=create.durable.result(5)
        self.assertEqual(c.store.branch_namespace('chat'),namespace)
        intent=json.loads((c._intents/'root-op.json').read_text())['intent']
        self.assertEqual(intent['schema'],2)
        request_digest=hashlib.sha256(_json_bytes(intent['presentation'])).hexdigest()
        expected_signature=hashlib.sha256(_json_bytes(dict(branch='chat',ids=[10,11],max_steps=0,
            kind='create',expected_revision=None,principal=None,
            presentation_digest=request_digest))).hexdigest()
        self.assertEqual(intent['signature'],expected_signature)
        root_operation='root-'+hashlib.sha256(b'root-op').hexdigest()[:32]
        receipt=c.store.presentation_receipt(root_operation)
        self.assertEqual(receipt['presentation']['messages'][0]['content'],'hello')
        self.assertEqual(receipt['presentation']['input_span']['count'],2)
        self.assertEqual(result['branch']['revision'],1)

    def test_first_question_uses_only_suffix_from_idle_codebase_prefix(self):
        c,models=self.start();ns=c.store.create_namespace('Codebase','namespace-op').namespace_id
        prepared=c.create('chat',(10,11),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='system',content='codebase')])).durable.result(5)
        before=models[0].stats['forward_tokens']
        result=c.submit('chat',(12,),1,'first-question',expected_revision=1,
            presentation=request('initial_turn',ns,
                messages=[dict(id='3'*32,role='user',content='question')])).durable.result(5)
        self.assertEqual(models[0].stats['forward_tokens']-before,2)
        self.assertEqual(c.store.lineage(result['snapshot'])[1],[10,11,12,0])
        self.assertEqual(c.store.get_snapshot(prepared['snapshot'])['inputs'],[10,11])
        self.assertEqual(c.store.presentation_receipt('first-question')['presentation']['kind'],'initial_turn')

    def test_foreign_namespace_rejected_before_worker_dispatch(self):
        c,models=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        foreign=c.store.create_namespace('other chat','foreign-namespace-op').namespace_id
        c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None)).durable.result(5)
        calls=[]; models[0].run_greedy_request=lambda *a,**k: calls.append('ran')
        with self.assertRaises(InvalidOperation):
            c.submit('chat',(12,),1,'foreign',expected_revision=1,
                presentation=request('answer',foreign,output='4'*32))
        self.assertEqual(calls,[])

    def test_same_operation_id_binds_presentation_text_namespace_and_kind(self):
        c,_=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        root=c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None))
        root.durable.result(5)
        with self.assertRaises(ValueError):
            c.create('chat',(10,),'root-op',presentation=request('root',ns,
                messages=[dict(id='2'*32,role='user',content='different text')],output=None))
        value=request('answer',ns,output='4'*32)
        job=c.submit('chat',(),1,'answer-op',expected_revision=1,presentation=value)
        self.assertEqual(c.submit('chat',(),1,'answer-op',expected_revision=1,presentation=value),job)
        with self.assertRaises(ValueError):
            c.submit('chat',(),1,'answer-op',expected_revision=1,
                presentation=request('answer',ns,output='4'*32,renderer='c'*64))
        with self.assertRaises(ValueError):
            c.submit('chat',(),1,'answer-op',expected_revision=1,
                presentation=request('answer',ns,output='5'*32))
        job.durable.result(5)

    def test_uncertain_store_commit_retry_reuses_same_annotation(self):
        c,models=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None)).durable.result(5)
        model=models[0]; original=model.save_prepared_branch; calls=[]
        def commit_then_raise(*args,**kwargs):
            receipt=original(*args,**kwargs); calls.append(receipt)
            raise OSError('ack lost after branch receipt')
        model.save_prepared_branch=commit_then_raise
        job=c.submit('chat',(),1,'answer-op',expected_revision=1,
            presentation=request('answer',ns,output='4'*32))
        job.snapshot.result(5)
        with self.assertRaisesRegex(OSError,'ack lost'):
            job.durable.result(5)
        trace_count=len([entry for entry in model.trace if entry[0]=='start'])
        model.save_prepared_branch=original
        recovered=c.retry_failed('answer-op').result(5)
        self.assertEqual(len(calls),1)
        self.assertEqual(recovered['branch']['revision'],2)
        self.assertEqual(len([entry for entry in model.trace if entry[0]=='start']),trace_count)
        bound=c.store.presentation_receipt('answer-op')['presentation']
        self.assertEqual(bound['output_span']['message_id'],'4'*32)

    def test_failure_before_publication_leaves_head_unchanged_then_retries(self):
        c,models=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None)).durable.result(5)
        model=models[0]; old=c.store.head('chat'); original=model.save_prepared_branch
        model.save_prepared_branch=lambda *a,**k: (_ for _ in ()).throw(OSError('before branch publication'))
        job=c.submit('chat',(),1,'answer-op',expected_revision=1,
            presentation=request('answer',ns,output='4'*32))
        job.snapshot.result(5)
        with self.assertRaisesRegex(OSError,'before branch'):
            job.durable.result(5)
        self.assertEqual(c.store.head('chat'),old)
        model.save_prepared_branch=original
        result=c.retry_failed('answer-op').result(5)
        self.assertEqual(result['branch']['revision'],2)
        self.assertEqual(c.store.presentation_receipt('answer-op')['presentation']['kind'],'answer')

    def test_restart_retry_uses_original_source_and_presentation_receipt(self):
        c,models=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None)).durable.result(5)
        expected=c.submit('chat',(),1,'answer-op',expected_revision=1,
            presentation=request('answer',ns,output='4'*32)).durable.result(5)
        directory,config,factory=c.store.root,c.config,c._factory
        count=sum(row[0]=='start' for model in models.values() for row in model.trace)
        c.close()
        resumed=ContextCoordinator(directory,config,factory); self.addCleanup(resumed.close)
        resumed.ready(5)
        retry=resumed.submit('chat',(),1,'answer-op',expected_revision=1,
            presentation=request('answer',ns,output='4'*32))
        self.assertEqual(retry.durable.result(5),expected)
        self.assertEqual(sum(row[0]=='start' for model in models.values() for row in model.trace),count)

    def test_missing_or_corrupt_intent_cannot_recover_or_rebind_committed_operation(self):
        c,models=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None)).durable.result(5)
        c.submit('chat',(),1,'answer-op',expected_revision=1,
            presentation=request('answer',ns,output='4'*32)).durable.result(5)
        directory,config,factory=c.store.root,c.config,c._factory
        count=sum(row[0]=='start' for model in models.values() for row in model.trace)
        c.close()
        resumed=ContextCoordinator(directory,config,factory); self.addCleanup(resumed.close)
        resumed.ready(5)
        intent=resumed._intents/'answer-op.json'
        envelope=json.loads(intent.read_text()); envelope['checksum']='0'*64
        intent.write_text(json.dumps(envelope),encoding='utf-8')
        with self.assertRaises(IntegrityError):
            resumed.submit('chat',(),1,'answer-op',expected_revision=1,
                presentation=request('answer',ns,output='4'*32))
        intent.unlink()
        with self.assertRaises(Conflict):
            resumed.submit('chat',(),1,'answer-op',expected_revision=1,
                presentation=request('answer',ns,output='4'*32))
        self.assertEqual(sum(row[0]=='start' for model in models.values() for row in model.trace),count)

    def test_intent_schema_rejects_boolean_integer_alias(self):
        c,_=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None)).durable.result(5)
        path=c._intents/'root-op.json'
        envelope=json.loads(path.read_text()); envelope['intent']['schema']=True
        envelope['checksum']=hashlib.sha256(_json_bytes(envelope['intent'])).hexdigest()
        path.write_text(json.dumps(envelope),encoding='utf-8')
        with self.assertRaises(IntegrityError):
            c._intent('root-op',envelope['intent']['signature'],None,
                request('root',ns,messages=[dict(id='2'*32,role='user',content='hello')],output=None))

    def test_malformed_kind_and_reused_input_output_message_id_rejected(self):
        from context_time_travel.presentation_request import detach_request
        from context_time_travel.store import InvalidOperation
        ns='a'*32
        value=request('answer',ns,output='4'*32)
        value['kind']=[]
        with self.assertRaises(InvalidOperation): detach_request(value)
        value=request('user_turn',ns,messages=[dict(id='4'*32,role='user',content='next')],output='4'*32)
        with self.assertRaises(InvalidOperation): detach_request(value)

    def test_resume_output_id_must_match_active_presentation_before_dispatch(self):
        c,_=self.start(); ns=c.store.create_namespace('test chat','namespace-op').namespace_id
        c.create('chat',(10,),'root-op',presentation=request('root',ns,
            messages=[dict(id='2'*32,role='user',content='hello')],output=None)).durable.result(5)
        c.submit('chat',(),1,'active-answer',expected_revision=1,
            presentation=request('answer',ns,output='4'*32)).durable.result(5)
        with self.assertRaises(InvalidOperation):
            c.submit('chat',(),1,'resume-bad',expected_revision=2,
                presentation=request('resume',ns,output='5'*32))

    def test_legacy_without_presentation_keeps_schema_one_intent(self):
        c,_=self.start(); c.create('legacy',(10,),'legacy-root').durable.result(5)
        path=c._intents/'legacy-root.json'
        value=json.loads(path.read_text())['intent']
        self.assertEqual(set(value),{'schema','signature','head'})
        self.assertEqual(value['schema'],1)
        expected=hashlib.sha256(_json_bytes(dict(branch='legacy',ids=[10],max_steps=0,
            kind='create',expected_revision=None,principal=None))).hexdigest()
        self.assertEqual(value['signature'],expected)


if __name__ == '__main__': unittest.main()
