"""Exact GGUF token-ID and bounded timing qualification; no GPU allocation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from tools.fastokens_tokenizer import FastokensTokenizer
    from tools.strata_tokenizer import Tokenizer
    import fastokens._native as native
    reference=Tokenizer.from_gguf(args.model);fast=FastokensTokenizer.from_gguf(args.model)
    randomizer=random.Random(20261023)
    alphabet='abC_Z09 ,.:;\t\n\r\x00\x01é中界🙂\u0301\u200d\U0001e4d0\U0001e5d0'
    texts=['','hello world',"we're THEY'LL",'a\u0301_j','\r\n \n  \t',''.join(map(chr,range(256))),
           '中文かな 한글 '*1000,'<|im_start|>system\n<tool_call>literal</tool_call><|im_end|>',
           'def double(value):\n    return value * 2\n', 'é\u0301\u200d🙂'*500]
    texts.extend(''.join(randomizer.choice(alphabet) for _ in range(randomizer.randrange(1,500))) for _ in range(400))
    literals=list(reference.special_tokens)
    literals.extend(t for t in reference.tokens if '<' in t and '>' in t and t not in reference.special_tokens)
    for literal in literals:texts.extend([literal,'before '+literal+' after'])
    prefix='def helper(value):\n    return value + 1 # é中\n'*200
    texts.extend(prefix[:cut]+' and different continuation' for cut in (1,2,3,7,31,127,511,len(prefix)-1,len(prefix)))
    checked=0;failures=[]
    for index,text in enumerate(texts):
        for special in (False,True):
            for plain in ((),[(0,len(text))]):
                expected=reference.encode(text,parse_special=special,plain=plain)
                actual=fast.encode(text,parse_special=special,plain=plain)
                checked+=1
                if actual!=expected:
                    failures.append(dict(case=index,parse_special=special,plain=bool(plain),
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        reference_tokens=len(expected),candidate_tokens=len(actual)))
    # Use the actual COW chat-template/literal-span renderer as a separate boundary.
    from serve.cow_http import model_text_components
    components=model_text_components(args.model)
    encoder=components['render'].__self__
    for text in texts[:30]:
        messages=[dict(role='system',content='Assist with Python code and writing.'),dict(role='user',content=text)]
        rendered=encoder.render(messages,add_generation_prompt=True)
        expected=encoder.encode_rendered(messages,rendered,True)
        original=encoder.tok;encoder.tok=fast
        try:actual=encoder.encode_rendered(messages,rendered,True)
        finally:encoder.tok=original
        checked+=1
        if actual!=expected:failures.append(dict(boundary='chat-template',text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    timings=[]
    if not failures:
        payload=''.join(f'def method_{i}(value):\n    return value + {i} # αβ中🙂\n' for i in range(1800))
        for label,text in [('code',payload),('long-unbroken','中文編碼测试'*3000),('shared-prefix',payload+payload[:1000])]:
            row=dict(case=label,utf8_bytes=len(text.encode()))
            for name,tokenizer in [('python',reference),('fastokens',fast)]:
                samples=[]
                for _ in range(3):
                    tokenizer._piece_ids.clear();started=time.perf_counter();ids=tokenizer.encode(text)
                    cold=time.perf_counter()-started
                    started=time.perf_counter();again=tokenizer.encode(text);warm=time.perf_counter()-started
                    if again!=ids:raise AssertionError('Warm piece cache changed IDs')
                    samples.append((cold,warm))
                row[name]=dict(cold_seconds=statistics.median(v[0] for v in samples),
                               warm_seconds=statistics.median(v[1] for v in samples),tokens=len(ids))
            timings.append(row)
    identity=hashlib.sha256(json.dumps(dict(tokens=reference.tokens,merges=list(reference.ranks),
        token_types=reference.token_types,pre=reference.pre),ensure_ascii=False).encode()).hexdigest()
    report=dict(schema=1,status='PASS' if not failures else 'FAIL',checked=checked,failures=failures,
        model_name=args.model.name,tokenizer_identity=identity,vocabulary=len(reference.tokens),
        fastokens_version=importlib.metadata.version('fastokens'),
        native_library_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        timings=timings,scope='Exact token IDs, literal controls, protected spans, Unicode, prefix cuts and COW chat rendering')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
