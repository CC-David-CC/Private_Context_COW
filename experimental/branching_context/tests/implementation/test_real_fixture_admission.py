"""Model/input admission before spawning inference; no GPU qualification."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import resident_resume


class RealFixtureAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'fixture.json'
        self.profile = dict(model_contract_id='model',execution=dict(capacity=32))
        self.fixture = dict(model_contract_id='model',prefix=[1,2],A=[3],B=[4,5],
                            scope='real source/chat-template fixture; no inference executed')

    def read(self):
        self.path.write_text(json.dumps(self.fixture))
        return resident_resume.read_fixture(self.path,self.profile,8)

    def test_different_suffix_lengths_are_preserved_as_real_native_requests(self):
        actual = self.read()
        self.assertEqual(actual['prefix']+actual['A'],[1,2,3])
        self.assertEqual(actual['prefix']+actual['B'],[1,2,4,5])

    def test_wrong_model_empty_or_identical_branches_and_invalid_tokens_fail_admission(self):
        for key,value in [('model_contract_id','other'),('prefix',[]),('B',[3]),
                          ('A',[True]),('A',[-1]),('A',[248320])]:
            with self.subTest(key=key):
                original = self.fixture[key]
                self.fixture[key] = value
                with self.assertRaises(ValueError): self.read()
                self.fixture[key] = original

    def test_context_budget_rejects_input_before_creating_native_process(self):
        self.profile['execution']['capacity'] = 10
        with self.assertRaises(ValueError): self.read()


if __name__ == '__main__': unittest.main()
