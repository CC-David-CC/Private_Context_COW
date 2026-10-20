"""Startup-loaded immutable GBNF registry tests (real GrammarEngine, CPU)."""
import json
import tempfile
from pathlib import Path
import unittest

from context_time_travel.store import IntegrityError, InvalidOperation
from test_grammar_first import TinyByteTokenizer
from serve.cow_grammar import GrammarRegistry


class CowGrammarRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer = TinyByteTokenizer()
        self.eos = [TinyByteTokenizer.EOS]

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self, states, name="registry.json"):
        path = self.root / name
        path.write_text(json.dumps({"schema": 1, "states": states}), encoding="utf-8")
        return path

    @staticmethod
    def states(source='root ::= "A"'):
        return {"answer": {"format": "gbnf", "source": source}}

    def load(self, path):
        return GrammarRegistry.load(path, self.tokenizer, self.eos)

    def test_inline_registry_compiles_and_exposes_only_host_state_names(self):
        registry = self.load(self.write_config(self.states()))
        self.assertEqual(registry.states, ("answer",))
        state = registry.start("answer", "registry-test", [], len(self.tokenizer.tokens))
        session = registry.restore(state, [])
        self.assertEqual(session.engine.engine_id, registry.engine_id)
        with self.assertRaises((AttributeError, TypeError)):
            registry.states += ("client-added",)
        with self.assertRaises(InvalidOperation):
            registry.start("client-added", "registry-test", [], len(self.tokenizer.tokens))

    def test_changed_definition_gets_new_engine_identity_and_old_resume_fails(self):
        first = self.load(self.write_config(self.states('root ::= "A"')))
        state = first.start("answer", "resume-test", [], len(self.tokenizer.tokens))
        second = self.load(self.write_config(self.states('root ::= "B"'), "changed.json"))
        self.assertNotEqual(first.spec_sha256, second.spec_sha256)
        self.assertNotEqual(first.engine_id, second.engine_id)
        with self.assertRaises(IntegrityError):
            second.restore(state, [])

    def test_loaded_definition_is_immutable_after_config_file_changes(self):
        path = self.write_config(self.states('root ::= "A"'))
        registry = self.load(path)
        identity = registry.engine_id
        path.write_text(json.dumps({"schema": 1, "states": self.states('root ::= "B"')}),
                        encoding="utf-8")
        state = registry.start("answer", "immutable-test", [], len(self.tokenizer.tokens))
        self.assertEqual(registry.engine_id, identity)
        self.assertEqual(state["constraint"]["engine_id"], identity)

    def test_malformed_duplicate_unknown_and_path_based_definitions_reject(self):
        cases = [
            b'{"schema":1,"schema":1,"states":{"answer":{"format":"gbnf","source":"root ::= \\\"A\\\""}}}',
            json.dumps({"schema": 2, "states": self.states()}).encode(),
            json.dumps({"schema": 1, "states": self.states(), "extra": True}).encode(),
            json.dumps({"schema": 1, "states": {"answer": {
                "format": "gbnf", "source": 'root ::= "A"', "path": "/etc/passwd"}}}).encode(),
            json.dumps({"schema": 1, "states": {"answer": {
                "format": "json_schema", "source": '{}'}}}).encode(),
            b'\xff\xfe',
        ]
        for index, raw in enumerate(cases):
            path = self.root / f"invalid-{index}.json"
            path.write_bytes(raw)
            with self.subTest(index=index), self.assertRaises(InvalidOperation):
                self.load(path)

    def test_registry_size_is_bounded_before_json_parse(self):
        path = self.root / "oversized.json"
        path.write_bytes(b" " * (GrammarRegistry.MAX_CONFIG_BYTES + 1))
        with self.assertRaisesRegex(InvalidOperation, "exceeds"):
            self.load(path)

    def test_state_count_and_total_source_limits_are_bounded(self):
        too_many = {
            f"state-{index}": {"format": "gbnf", "source": 'root ::= "A"'}
            for index in range(65)
        }
        with self.assertRaisesRegex(InvalidOperation, "1..64"):
            self.load(self.write_config(too_many, "too-many.json"))

        large = 'x' * (800 * 1024)
        too_much_source = {
            f"state-{index}": {"format": "gbnf", "source": large}
            for index in range(4)
        }
        with self.assertRaisesRegex(InvalidOperation, "Total grammar source"):
            self.load(self.write_config(too_much_source, "too-much-source.json"))

    def test_symlinked_registry_is_rejected(self):
        target = self.write_config(self.states(), "target.json")
        link = self.root / "link.json"
        link.symlink_to(target)
        with self.assertRaisesRegex(InvalidOperation, "regular file"):
            self.load(link)


if __name__ == "__main__":
    unittest.main(verbosity=2)
