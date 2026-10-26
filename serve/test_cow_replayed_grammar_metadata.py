"""Receipt replay must retain parser completion, including accepting active scopes."""
import unittest
from serve import test_cow_grammar_canonical_flow as fixture
from context_time_travel.store import InvalidOperation

class ReplayedGrammarMetadataTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture.CowCanonicalGrammarFlowTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture._canonical_forked_root()

    def compare(self,state,maximum,*,stops=None):
        f=self.fixture
        request=f.text.answer('BranchA',0,'replay-metadata',maximum,grammar_state=state,
            settings=dict(temperature=0.,min_p=0.),**({}if stops is None else dict(text_stops=stops)))
        request.durable.result(5);raw=request.snapshot.result(5)
        source=f.coordinator.committed_generation('replay-metadata')
        before=f.backends[0].work_counters()
        replay=f.text._replay_committed(source)['answer']
        self.assertEqual(replay['tokens'],raw['tokens'])
        self.assertEqual(replay['grammar_complete'],raw['grammar_complete'])
        self.assertEqual(replay['stopped'],raw['stopped'])
        self.assertEqual(f.backends[0].work_counters(),before)
        return raw,source

    def test_complete_named_scope_keeps_completion_and_stopping(self):
        raw,_=self.compare('READ',3)
        self.assertTrue(raw['grammar_complete']);self.assertTrue(raw['stopped'])

    def test_incomplete_active_scope_is_not_promoted(self):
        raw,_=self.compare('READ',1)
        self.assertFalse(raw['grammar_complete']);self.assertFalse(raw['stopped'])

    def test_accepting_scope_before_eos_is_still_active(self):
        raw,_=self.compare('TEST',1)
        self.assertTrue(raw['grammar_complete']);self.assertFalse(raw['stopped'])

    def test_text_stop_does_not_make_an_incomplete_grammar_complete(self):
        raw,_=self.compare('READ',3,stops=('A',))
        self.assertFalse(raw['grammar_complete']);self.assertTrue(raw['stopped'])

    def test_eos_cannot_replace_an_unavailable_saved_grammar(self):
        _,source=self.compare('READ',3)
        self.fixture.text.grammar_registry=None
        with self.assertRaises(InvalidOperation):self.fixture.text._replay_committed(source)

    def coordinator_compare(self,state,maximum):
        f=self.fixture;raw,_=self.compare(state,maximum)
        f.coordinator.close()
        from context_time_travel.coordinator import ContextCoordinator
        from context_time_travel.pools import PoolConfig
        from context_time_travel.real import BranchingModel
        from serve.cow_adapter import CowAdapter
        from serve.cow_text_adapter import CowTextAdapter
        def factory(gpu):
            backend=fixture.GrammarCPUBackend(f.profile,f.tokenizer);f.backends[gpu]=backend
            return BranchingModel(f.tmp.name,backend,f.profile)
        f.coordinator=ContextCoordinator(f.tmp.name,PoolConfig(gpus=(0,),ram_bytes=64<<20,disk_bytes=256<<20,min_available_bytes=0),factory)
        self.addCleanup(f.coordinator.close);f.coordinator.ready(5)
        adapter=CowAdapter(f.coordinator,grammar_registry=f.registry)
        text=CowTextAdapter(adapter,f.tokenizer,fixture._render,
            lambda messages,rendered,add_generation_prompt:fixture._encode(f.tokenizer,messages,rendered,add_generation_prompt),
            initial_root_mode=fixture._initial_root_mode,renderer_id='b'*64,grammar_registry=f.registry)
        before=f.backends[0].work_counters()
        replay=text.answer('BranchA',0,'replay-metadata',maximum,grammar_state=state,settings=dict(temperature=0.,min_p=0.))
        replay.durable.result(5);answer=replay.snapshot.result(5)
        for key in ('tokens','grammar_complete','stopped'):self.assertEqual(answer[key],raw[key])
        self.assertEqual(before,f.backends[0].work_counters())

    def test_coordinator_restart_replays_complete_grammar_metadata_without_forward(self):self.coordinator_compare('READ',3)
    def test_coordinator_restart_replays_incomplete_active_metadata_without_forward(self):self.coordinator_compare('READ',1)
    def test_coordinator_restart_replays_accepting_active_metadata_without_forward(self):self.coordinator_compare('TEST',1)

if __name__=='__main__':unittest.main()
