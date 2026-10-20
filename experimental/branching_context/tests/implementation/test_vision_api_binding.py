"""Synthetic API/receipt tests, separate from executed real GPU evidence."""
from contextlib import contextmanager
import tempfile
import unittest

from context_time_travel.input_spans import InputHistory
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, InvalidOperation
from context_time_travel.vision_execution import RESUME_CONTRACT, POSITION_CONVENTION
from test_vision_execution import spans, receipt
from test_root_computation_reuse import CPUBackend


class MixedCPUBackend(CPUBackend):
    def contract_fields(self):
        return dict(positions=POSITION_CONVENTION,input_kind='canonical-mixed-spans-v1')

    @contextmanager
    def lease(self):
        yield {'inputs':InputHistory([])}

    def consume_inputs(self,lease,inputs,*,embeddings):
        prior=lease['inputs']
        lease['inputs']=InputHistory(prior.spans+inputs)
        self.stats['forward_tokens']+=len(lease['inputs'])-len(prior)
        self.last_execution_receipt=receipt(lease['inputs'],position=len(prior))

    def consume(self,lease,ids):
        self.consume_inputs(lease,[dict(kind='text',tokens=list(ids))],embeddings={})

    def export(self,lease): return lease['inputs'].digest.encode()
    def logits(self,lease): return tuple(float(n) for n in range(self.vocab_size))
    def position(self,lease): return len(lease['inputs'])-1
    def piece(self,token): return bytes([65+token])
    def is_eog(self,token): return False

    def restore(self,lease,blob,*,history,frontier):
        if blob!=history.digest.encode(): raise IntegrityError('Incorrect mixed input binding')
        lease['inputs']=history
        return blob


class NoArtifacts:
    def __init__(self,store): self.store=store
    @contextmanager
    def lease(self,span,**kwargs):
        from types import SimpleNamespace
        yield SimpleNamespace(embedding=object())


class VisionAPIBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-exec',
            model_weights_sha256='a'*64,execution=dict(resume_contract=RESUME_CONTRACT,
                prefill_chunk=128,embedding_width=4,capacity=128))
        self.backend=MixedCPUBackend(self.profile)
        self.model=BranchingModel(self.tmp.name,self.backend,self.profile)
        self.artifacts=NoArtifacts(self.model.store)

    def create(self,source=None):
        return self.model.create_inputs(spans() if source is None else source,artifacts=self.artifacts)

    def test_canonical_root_identity_and_reuse_include_image(self):
        first=self.create(); work=self.backend.stats['forward_tokens']
        self.assertEqual(first,self.create())
        self.assertEqual(work,self.backend.stats['forward_tokens'])
        self.assertNotEqual(first,self.create(spans(embedding_sha256='f'*64)))
        self.assertEqual(self.model.store.get_snapshot(first)['schema'],2)

    def test_feed_publishes_mixed_child_and_preserves_image_identity(self):
        root=self.create(); child=self.model.feed(root,[4,5])
        self.assertEqual(self.model.history(child),InputHistory(spans())+(4,5))
        self.assertEqual(self.model.history(root),InputHistory(spans()))
        self.assertEqual(self.model.store.get_snapshot(child)['parent'],root)
        self.assertEqual(len(self.model.cache[child].execution_schedule),2)

    def test_materialize_refuses_altered_receipt_before_native_restore(self):
        from dataclasses import replace
        root=self.create(); saved=self.model.cache[root]
        changed=dict(saved.execution_schedule[0],native_input_sha256='0'*64)
        self.model.cache[root]=replace(saved,execution_schedule=(changed,))
        self.backend.restore=lambda *a,**k:self.fail('Invalid receipt reached native restore')
        with self.assertRaises(IntegrityError):
            with self.model.materialize(root): pass

    def test_missing_checkpoint_does_not_silently_replay(self):
        root=self.create(); before=self.backend.stats['forward_tokens']; self.model.cache.clear()
        with self.assertRaises(IntegrityError):
            with self.model.materialize(root): pass
        self.assertEqual(before,self.backend.stats['forward_tokens'])

    def test_repetition_controller_counts_text_and_not_image_pad_rows(self):
        root=self.create(); active=self.model.start_generation(root,dict(temperature=0),'vision')
        self.assertEqual(self.model.controller(active)['seen'],{'1':1,'2':1,'3':1})
        self.assertEqual(self.model.history(active),self.model.history(root))

    def test_incomplete_image_boundary_is_rejected_before_forward(self):
        with self.assertRaises(InvalidOperation): self.create(spans()[:-1])
        self.assertEqual(self.backend.stats['forward_tokens'],0)


if __name__=='__main__': unittest.main()
