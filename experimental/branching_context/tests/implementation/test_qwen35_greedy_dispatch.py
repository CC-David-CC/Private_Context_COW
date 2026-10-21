"""CPU dispatch regressions, not native/GPU qualification."""
import unittest
from unittest.mock import Mock
from concurrent.futures import CancelledError
from context_time_travel.real import BranchingModel

class Qwen35GreedyDispatchTests(unittest.TestCase):
    def model(self):
        m=object.__new__(BranchingModel)
        m.profile={'backend':'qwen35-llama.cpp','execution':{}}
        m.store=Mock();m.run_sampled_request=Mock(return_value={'snapshot':'consumed'})
        return m

    def test_plain_greedy_keeps_mixed_artifacts_and_reencode(self):
        m=self.model();delta=object();payloads={'image':object()}
        reencode=lambda *args: None;cancelled=lambda:False;progress=lambda t:None
        result=m.run_greedy_request('parent',delta,8,stream='answer',on_token=progress,
            cancelled=cancelled,artifacts=payloads,reencode=reencode)
        self.assertEqual(result,{'snapshot':'consumed'})
        m.run_sampled_request.assert_called_once()
        args,kw=m.run_sampled_request.call_args
        self.assertEqual(args,('parent',delta,8))
        self.assertEqual(kw['sampling']['temperature'],0)
        self.assertIs(kw['artifacts'],payloads);self.assertIs(kw['reencode'],reencode)
        self.assertIs(kw['cancelled'],cancelled);self.assertIs(kw['on_token'],progress)
        m.store.authorize.assert_called_once_with('parent','append')

    def test_cancelled_greedy_never_enters_native_request(self):
        m=self.model()
        with self.assertRaises(CancelledError):m.run_greedy_request('parent',(),8,stream='answer',cancelled=lambda:True)
        m.run_sampled_request.assert_not_called();m.store.authorize.assert_not_called()
