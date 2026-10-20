"""JSON Schema uses the canonical grammar controller, never a prompt fallback."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.store import IntegrityError, InvalidOperation
from tests.implementation.test_grammar_first import TinyByteTokenizer
from serve.cow_grammar import GrammarRegistry


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["read_file", "run_tests"]},
        "arguments": {
            "type": "object",
            "properties": {"path": {"type": "string"},
                           "limit": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": ["path", "limit"], "additionalProperties": False,
        },
    },
    "required": ["tool", "arguments"], "additionalProperties": False,
}


class JsonSchemaGrammarTests(unittest.TestCase):
    def engine(self, schema=TOOL_SCHEMA):
        return GrammarEngine(TinyByteTokenizer(), {
            "ACTION": {"format": "json_schema", "source": schema},
            "GBNF": {"format": "gbnf", "source": 'root ::= "A" | "B"'},
        }, [TinyByteTokenizer.EOS])

    def start(self, engine, name="ACTION"):
        state = engine.start(name, "json-schema-test", [], engine.vocab_size)
        return state, engine.restore(state, [])

    def consume(self, engine, session, state, text):
        for byte in text.encode("utf-8"):
            state = session.consume(state, byte, engine.tokenizer.token_bytes(byte), False)
        return state

    def accepted(self, engine, value):
        state, session = self.start(engine)
        # Compile and consume a single canonical property order, also used by
        # the schema identity; no dependence on mutable dictionary insertion order.
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.consume(engine, session, state, text)
        self.assertTrue(session.is_complete())

    def rejected(self, engine, text):
        state, session = self.start(engine)
        with self.assertRaises(IntegrityError):
            self.consume(engine, session, state, text)

    def test_required_fields_tool_enum_and_typed_arguments(self):
        engine = self.engine()
        for tool in ("read_file", "run_tests"):
            self.accepted(engine, {"tool": tool, "arguments": {"path": "src/main.py", "limit": 4}})
        invalid = [
            {"arguments": {"path": "x", "limit": 1}},
            {"tool": "read_file"},
            {"tool": "delete_everything", "arguments": {"path": "x", "limit": 1}},
            {"tool": "read_file", "arguments": {"path": 3, "limit": 1}},
            {"tool": "read_file", "arguments": {"path": "x", "limit": "1"}},
            {"tool": "read_file", "arguments": {"path": "x", "limit": True}},
            {"tool": "read_file", "arguments": {"path": "x", "limit": 0}},
            {"tool": "read_file", "arguments": {"path": "x", "limit": 9}},
            {"tool": "read_file", "arguments": {"path": "x"}},
            {"tool": "read_file", "arguments": {"path": "x", "limit": 1, "shell": "rm"}},
            {"tool": "read_file", "arguments": {"path": "x", "limit": 1}, "extra": 0},
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.rejected(engine, json.dumps(value, sort_keys=True, separators=(",", ":")))

    def test_boolean_null_array_and_unicode_string_types(self):
        schema = {"type": "object", "properties": {
            "a": {"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 2},
            "b": {"type": "null"}, "s": {"type": "string"}},
            "required": ["a", "b", "s"], "additionalProperties": False}
        engine = self.engine(schema)
        self.accepted(engine, {"a": [True, False], "b": None, "s": 'quote " slash \\ café'})
        self.rejected(engine, '{"a":[1],"b":null,"s":"x"}')
        self.rejected(engine, '{"a":[],"b":null,"s":"x"}')

    def test_json_syntax_only_schema_accepts_object_but_not_broken_json(self):
        engine = self.engine({"type": "object"})
        self.accepted(engine, {"arbitrary": [1, "text", False, None]})
        for text in ('{"x":}', '{"x":NaN}', '{"x":1,}', 'not JSON'):
            with self.subTest(text=text):
                self.rejected(engine, text)

    def test_illegal_raw_argmax_cannot_bypass_schema_and_greedy_draws_stay_zero(self):
        engine = self.engine({"type": "object"})
        state, session = self.start(engine)
        logits = [0.0] * engine.vocab_size
        logits[ord("Z")] = 1000.0
        logits[ord("{")] = 5.0
        self.assertEqual(session.choose(logits), ord("{"))
        self.assertEqual(state["next_draw"], 0)
        state = session.consume(state, ord("{"), b"{", False)
        self.assertEqual(state["next_draw"], 0)

    def test_schema_snapshot_is_detached_and_key_order_has_one_identity(self):
        source = copy.deepcopy(TOOL_SCHEMA)
        first = self.engine(source)
        reordered = dict(reversed(list(source.items())))
        reordered["properties"] = dict(reversed(list(source["properties"].items())))
        second = self.engine(reordered)
        self.assertEqual(first.engine_id, second.engine_id)
        self.assertEqual(first.spec_sha256, second.spec_sha256)
        source["properties"]["tool"]["enum"][:] = ["forbidden"]
        value = {"tool": "read_file", "arguments": {"path": "x", "limit": 1}}
        self.accepted(first, value)
        self.accepted(second, value)

    def test_partial_progress_forks_and_restores_independently(self):
        engine = self.engine({"type": "object", "properties": {"answer": {"enum": ["A", "B"]}},
                              "required": ["answer"], "additionalProperties": False})
        state, session = self.start(engine)
        prefix = '{"answer":"'
        state = self.consume(engine, session, state, prefix)
        saved = copy.deepcopy(state)
        history = list(prefix.encode())
        branch_a = engine.restore(saved, history)
        branch_b = engine.restore(saved, history)
        self.consume(engine, branch_a, copy.deepcopy(saved), 'A"}')
        self.assertTrue(branch_a.is_complete())
        self.assertFalse(branch_b.is_complete())
        self.consume(engine, branch_b, copy.deepcopy(saved), 'B"}')
        self.assertTrue(branch_b.is_complete())
        self.assertEqual(state, saved)
        incompatible = self.engine({"type": "object"})
        with self.assertRaises(IntegrityError):
            incompatible.restore(saved, history)

    def test_invalid_or_unsupported_schema_does_not_fall_back_to_unconstrained(self):
        for schema in ('{}', [], {"type": "not_a_type"},
                       {"type": "object", "unsupportedKeyword": True},
                       {"type": "number", "minimum": float("nan")},
                       {"type": "object", "x-guidance": {"lenient": True}, "unsupportedKeyword": True},
                       {"$ref": "https://invalid.example/schema.json"}):
            with self.subTest(schema=schema), self.assertRaises(InvalidOperation):
                self.engine(schema)

    def test_registry_mixes_gbnf_and_schema_and_rejects_duplicate_schema_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.json"
            path.write_text(json.dumps({"schema": 1, "states": {
                "JSON": {"format": "json_schema", "source": {"type": "object"}},
                "TEXT": {"format": "gbnf", "source": 'root ::= "A"'}}}), encoding="utf-8")
            registry = GrammarRegistry.load(path, TinyByteTokenizer(), [TinyByteTokenizer.EOS])
            self.assertEqual(registry.states, ("JSON", "TEXT"))
            state = registry.start("JSON", "registry", [], registry.engine.vocab_size)
            self.assertEqual(registry.restore(state, []).engine.engine_id, registry.engine_id)
            path.write_text('{"schema":1,"states":{"JSON":{"format":"json_schema",'
                            '"source":{"type":"object","type":"string"}}}}', encoding="utf-8")
            with self.assertRaises(InvalidOperation):
                GrammarRegistry.load(path, TinyByteTokenizer(), [TinyByteTokenizer.EOS])


if __name__ == "__main__":
    unittest.main()
