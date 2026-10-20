"""JSON Unicode representations must preserve value, schema and controller identity."""
import copy
import hashlib
import json
import unittest

from tools.strata_tokenizer import Tokenizer
from test_grammar_first import TinyByteTokenizer
from context_time_travel.grammar_first import GrammarEngine, LLGUIDANCE_VERSION
from context_time_travel.store import IntegrityError, InvalidOperation

class JsonUnicodeEscapeTests(unittest.TestCase):
    @staticmethod
    def make_tokenizer():
        base = TinyByteTokenizer()
        return Tokenizer(base.tokens + ["</think>"], [], base.token_types + [4],
                         special_ids=base.special_ids)

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = cls.make_tokenizer()
        cls.eos = [cls.tokenizer.special_ids["eos_token_id"]]

    def engine(self, schema, *, thinking=False):
        source = dict(format="json_schema", source=schema)
        if thinking:
            source = dict(format="thinking", source=dict(schema=1,max_tokens=4096,final=source))
        return GrammarEngine(self.tokenizer, {"answer": source}, self.eos)

    def legal(self, engine, text, *, thinking=False):
        text = "CPU audit.</think>" + text if thinking else text
        tokens = self.tokenizer.encode(text, parse_special=thinking)
        self.assertEqual(b"".join(self.tokenizer.token_bytes(t) for t in tokens), text.encode("utf-8"))
        matcher = engine._matcher("answer")
        for token in tokens:
            mask = matcher.compute_bitmask()
            if not mask[token >> 3] & (1 << (token & 7)):
                return False
            if not matcher.consume_token(token):
                return False
        return matcher.is_accepting()

    @staticmethod
    def report_schema():
        return dict(type="object",properties={"stdout":dict(type="string")},
                    required=["stdout"],additionalProperties=False)

    def test_literal_and_standard_escaped_unicode_have_same_report_value(self):
        text = chr(0xe9)+" e"+chr(0x301)+" "+chr(0x394)+" "+chr(0x1f42e)+"\n"
        engine = self.engine(self.report_schema())
        for ascii_only in (False, True):
            encoded = json.dumps(dict(stdout=text), ensure_ascii=ascii_only, sort_keys=True)
            with self.subTest(ascii_only=ascii_only):
                self.assertEqual(json.loads(encoded)["stdout"], text)
                self.assertTrue(self.legal(engine, encoded))

    def test_uppercase_and_lowercase_four_hex_escapes_are_legal(self):
        engine = self.engine(self.report_schema())
        for encoded in (r'{"stdout":"\u00e9\u0394\ud83d\udc2e"}',
                        r'{"stdout":"\u00E9\u0394\uD83D\uDC2E"}'):
            with self.subTest(encoded=encoded):
                self.assertTrue(self.legal(engine, encoded))

    def test_thinking_final_uses_the_same_unicode_rules(self):
        engine = self.engine(self.report_schema(), thinking=True)
        for encoded in (json.dumps(dict(stdout=chr(0x1f42e)), ensure_ascii=False),
                        r'{"stdout":"\ud83d\udc2e"}'):
            with self.subTest(encoded=encoded):
                self.assertTrue(self.legal(engine, encoded, thinking=True))

    def test_surrogate_pair_is_one_character_for_exact_length(self):
        engine = self.engine(dict(type="string",minLength=1,maxLength=1))
        for encoded in (json.dumps(chr(0x1f42e),ensure_ascii=False), r'"\ud83d\udc2e"'):
            with self.subTest(encoded=encoded):
                self.assertTrue(self.legal(engine,encoded))
        for encoded in ('""',r'"\ud83d\udc2ex"',json.dumps(chr(0x1f42e)+"x",ensure_ascii=False)):
            with self.subTest(encoded=encoded):
                self.assertFalse(self.legal(engine,encoded))

    def test_combining_mark_remains_a_separate_length_character(self):
        engine = self.engine(dict(type="string",minLength=2,maxLength=2))
        self.assertTrue(self.legal(engine,r'"e\u0301"'))
        self.assertFalse(self.legal(engine,r'"\u00e9"'))

    def test_invalid_escapes_and_unpaired_surrogates_are_rejected(self):
        engine = self.engine(dict(type="string"))
        for encoded in (r'"\ud800"',r'"\udc00"',r'"\ud83dx"',r'"\ud83d\u00e9"',
                        r'"\u0e9"',r'"\U0001f42e"', '"'+chr(1)+'"'):
            with self.subTest(encoded=encoded):
                self.assertFalse(self.legal(engine,encoded))

    def test_required_types_and_additional_field_rules_still_reject(self):
        engine = self.engine(self.report_schema())
        for encoded in ('{}','{"stdout":false}','{"stdout":1}',
                        '{"stdout":"okay","extra":"unexpected"}',
                        '{"stdout":"okay"}garbage'):
            with self.subTest(encoded=encoded):
                self.assertFalse(self.legal(engine,encoded))

    def test_tool_enum_path_const_and_argument_types_still_hold(self):
        schema=dict(type="object",properties={
            "arguments":dict(type="object",properties={
                "path":dict(const="src/main.py"),
                "query":dict(type="string",minLength=1,maxLength=4)},
                required=["path","query"],additionalProperties=False),
            "tool":dict(enum=["read_file"])},
            required=["arguments","tool"],additionalProperties=False)
        engine=self.engine(schema)
        valid=dict(arguments=dict(path="src/main.py",query=chr(0x1f42e)),tool="read_file")
        for ascii_only in (False,True):
            self.assertTrue(self.legal(engine,json.dumps(valid,ensure_ascii=ascii_only,sort_keys=True)))
        invalid=[
            dict(valid,tool="delete_file"),
            dict(valid,arguments=dict(path="../secret",query="x")),
            dict(valid,arguments=dict(path="src/main.py",query=False)),
            dict(valid,arguments=dict(path="src/main.py",query="12345")),
            dict(valid,arguments=dict(path="src/main.py",query="x",extra="no")),
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertFalse(self.legal(engine,json.dumps(value,sort_keys=True)))

    def test_regex_path_constraints_cannot_be_bypassed(self):
        engine=self.engine(dict(type="string",pattern=r"^src/[a-z]+[.]py$"))
        self.assertTrue(self.legal(engine,'"src/main.py"'))
        for encoded in ('"../secret"','"src/main.txt"',r'"\u002e\u002e/secret"'):
            with self.subTest(encoded=encoded):
                self.assertFalse(self.legal(engine,encoded))

    def test_padded_escape_is_valid_json_but_not_equal_observed_unicode(self):
        encoded=r'{"stdout":"\u000e9"}'
        self.assertNotEqual(json.loads(encoded)["stdout"],chr(0xe9))
        self.assertTrue(self.legal(self.engine(self.report_schema()),encoded))

    def test_json_compiler_options_change_engine_identity_and_old_resume_rejects(self):
        engine=self.engine(self.report_schema())
        identity=dict(llguidance_version=LLGUIDANCE_VERSION,
            tokenizer_sha256=engine.tokenizer_sha256,spec_sha256=engine.spec_sha256,
            eos_tokens=list(engine.eos_tokens))
        legacy=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True,
            separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
        self.assertNotEqual(engine.engine_id,legacy)
        state=engine.start("answer","unicode-compatibility",[],len(self.tokenizer.tokens))
        old=copy.deepcopy(state)
        old["constraint"]["engine_id"]=legacy
        with self.assertRaises(IntegrityError):
            engine.restore(old,[])

    def test_gbnf_only_grammar_and_engine_identity_are_unchanged(self):
        import llguidance
        source='root ::= "A" | "B"'
        engine=GrammarEngine(self.tokenizer,{"answer":dict(format="gbnf",source=source)},self.eos)
        identity=dict(llguidance_version=LLGUIDANCE_VERSION,
            tokenizer_sha256=engine.tokenizer_sha256,spec_sha256=engine.spec_sha256,
            eos_tokens=list(engine.eos_tokens))
        legacy=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True,
            separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(engine.engine_id,legacy)
        self.assertEqual(engine._states["answer"]["grammar"],llguidance.grammar_from("gbnf",source))
        self.assertTrue(self.legal(engine,"A"))
        self.assertFalse(self.legal(engine,"C"))

    def test_client_schema_compiler_overrides_remain_rejected(self):
        schema=dict(self.report_schema(),**{"x-guidance":{"lenient":True}})
        with self.assertRaises(InvalidOperation):
            self.engine(schema)

if __name__ == "__main__":
    unittest.main()
