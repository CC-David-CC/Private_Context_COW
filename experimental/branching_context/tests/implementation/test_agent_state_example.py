"""Host state transitions are independent of model text and grammar completion."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('grammar_first_agent',
    Path(__file__).resolve().parents[2] / 'examples/grammar_first_agent.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class HostStateExampleTests(unittest.TestCase):
    def test_denial_never_advances_any_state(self):
        for state in example.STATES:
            self.assertEqual(example.next_state(state, {'accepted': False, 'applied': True, 'passed': True}), state)

    def test_accepted_results_choose_only_host_defined_edges(self):
        cases = [('READ', {}, 'EDIT'), ('EDIT', {'applied': False}, 'EDIT'),
                 ('EDIT', {'applied': True}, 'TEST'), ('TEST', {'passed': False}, 'EDIT'),
                 ('TEST', {'passed': True}, 'FINISH'), ('FINISH', {}, None)]
        for state, fields, expected in cases:
            self.assertEqual(example.next_state(state, {'accepted': True, **fields}), expected)

    def test_truthy_strings_and_model_claims_do_not_authorize(self):
        for state, result in [('READ', {'accepted': 'true'}), ('TEST', {'accepted': True}),
            ('TEST', {'accepted': True, 'passed': 'verified'}), ('EDIT', {'accepted': True, 'applied': 1}),
            ('execute-anything', {'accepted': True})]:
            with self.assertRaises(ValueError):
                example.next_state(state, result)

    def test_static_grammars_are_separate_and_host_named(self):
        grammars = example.load_grammars()
        self.assertEqual(tuple(grammars), example.STATES)
        self.assertTrue(all(row['format'] == 'gbnf' for row in grammars.values()))
        self.assertIn('read_file', grammars['READ']['source'])
        self.assertNotIn('propose_patch', grammars['READ']['source'])


if __name__ == '__main__':
    unittest.main()
