"""CPU protocol checks for the actual selected-token native receipt layout."""
import threading
import unittest
from context_time_travel.strata_backend import StrataBackend, _Lease, _validate_selected_token_capability
from context_time_travel.store import IntegrityError


class SelectedTokenReceiptTests(unittest.TestCase):
    def test_grammar_tokenizer_is_bound_to_model_not_just_vocab_length(self):
        from tools.strata_tokenizer import Tokenizer
        from context_time_travel.grammar_first import GrammarEngine
        backend = object.__new__(StrataBackend)
        backend._tokens = ['A', '<eos>']
        backend._metadata = {'tokenizer.ggml.tokens': backend._tokens, 'tokenizer.ggml.merges': [],
            'tokenizer.ggml.token_type': [1, 3], 'tokenizer.ggml.eos_token_id': 1}
        def engine(letter):
            tokenizer = Tokenizer([letter, '<eos>'], [], [1, 3],
                                  special_ids={'tokenizer.ggml.eos_token_id': 1})
            return GrammarEngine(tokenizer, {'state': {'format': 'gbnf', 'source': 'root ::= "' + letter + '"'}},
                                 eos_tokens=[1])
        backend.validate_grammar_engine(engine('A'))
        with self.assertRaises(IntegrityError):
            backend.validate_grammar_engine(engine('B'))

    def test_capability_must_be_explicit_and_match_ready_frame(self):
        profile = {'execution': {'selected_token_protocol': 'consumed-selected-receipt-v1'}}
        for ready in ({}, {'selected_token_protocol': 'different'}):
            with self.assertRaises(IntegrityError):
                _validate_selected_token_capability(profile, ready)
        _validate_selected_token_capability(profile, {'selected_token_protocol': 'consumed-selected-receipt-v1'})
        _validate_selected_token_capability({'execution': {}}, {})  # Existing greedy profile.

    def backend(self, operator='normal-verifier-decode', position=4):
        backend = object.__new__(StrataBackend)
        backend.profile = {'execution': {'generated_operator': 'normal-verifier-decode',
            'resume_contract': 'identical-native-operators-v1', 'prefill_chunk': 8, 'capacity': 64}}
        backend.vocab_size = 8
        counters = iter([{'forward_tokens': 4}, {'forward_tokens': 5}])
        backend._refresh_stats = lambda: next(counters)
        backend._request = lambda command: {'status': 'ok', 'consumed': 5, 'operator': operator,
            'position': position, 'count': 1, 'chunks': [[position, 1]]}
        lease = _Lease(backend, bound=True, consumed=4, state='ACTIVE', _owner_thread=threading.get_ident())
        backend._active = lease
        return backend, lease

    def test_receipt_matches_actual_selected_operator_and_position(self):
        backend, lease = self.backend()
        backend.consume_generated(lease, (2,))
        self.assertEqual(lease.consumed, 5)
        self.assertEqual(backend.last_execution_receipt['chunks'], [[4, 1]])
        self.assertEqual(backend.last_execution_receipt['operator'], 'normal-verifier-decode')

    def test_wrong_operator_or_position_is_not_relabelled(self):
        for operator, position in [('batched-prefill', 4), ('normal-verifier-decode', 5)]:
            backend, lease = self.backend(operator, position)
            with self.assertRaises(IntegrityError):
                backend.consume_generated(lease, (2,))
            self.assertFalse(hasattr(backend, 'last_execution_receipt'))


if __name__ == '__main__':
    unittest.main()
