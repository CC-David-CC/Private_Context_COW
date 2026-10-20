"""Check prompt literal boundaries with an actual exported tokenizer (no GPU)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
from serve.server import Service, MockEngine
from serve.test_literal_boundaries import LITERALS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    vocab = json.loads((a.tokenizer / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for text, i in vocab.items():
        tokens[i] = text
    types = json.loads((a.tokenizer / 'token_type.json').read_text())
    tok = Tokenizer(tokens, (a.tokenizer / 'merges.txt').read_text(encoding='utf-8').split('\n'), types)
    template_path = a.tokenizer / 'chat_template.jinja'
    if not template_path.exists():
        template_path = ROOT / 'serve/chat_template.jinja'
    svc = Service(MockEngine(tok, 'ok'), tok, ChatTemplate(template_path))
    specials = set(tok.special_tokens.values())
    own = svc.encode_prompt([{'role':'user','content':'plain'}], None, {})
    control = [i for i in own if i in specials]
    rows = []
    for text in LITERALS:
        ids = svc.encode_prompt([{'role':'user','content':text}], None, {})
        assert text in tok.decode(ids), text
        assert [i for i in ids if i in specials] == control, text
        raw = tok.encode(text, parse_special=True)
        plain = tok.encode(text, parse_special=True, plain=[(0,len(text))])
        rows.append({'literal':text,'raw_ids':raw,'plain_ids':plain,'status':'PASS',
                     'raw_special_ids':[i for i in raw if i in specials]})
    files = ['vocab.json','merges.txt','token_type.json']
    profile = {name:hashlib.sha256((a.tokenizer/name).read_bytes()).hexdigest() for name in files}
    report = {'status':'PASS','scope':'actual tokenizer prompt boundaries, no inference',
              'fingerprints':profile,'template_sha256':hashlib.sha256(template_path.read_bytes()).hexdigest(),
              'special_literals':len(tok.special_tokens),'cases':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS:',len(rows),'literal cases with actual model tokenizer')


if __name__ == '__main__':
    main()
