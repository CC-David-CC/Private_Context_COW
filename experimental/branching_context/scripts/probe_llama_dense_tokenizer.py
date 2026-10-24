"""Compare actual GGUF tokenization and every vocabulary piece with native llama.cpp.

CPU metadata/tokenizer qualification only: no model forward pass or weight update.
The native bridge tokenizes independently of the Python BPE implementation.
"""
import argparse,ctypes as C,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/branching_context/src')]
from context_time_travel.real import BranchingModel
from tools.strata_tokenizer import Tokenizer

def corpus():
 rows=['','17 26 1001','abc123456789xyz','e\u0301_j','<s><|im_start|>user\n17+26<|im_end|>\n','<think>quoted</think>',
       'def f(a,b):\n    return a+b\n','{"lines":["a", "  b"]}','\x00\t\r\n',"we're I'LL can't"]
 rows += ['x'+''.join(str(j%10) for j in range(n))+'y' for n in range(1,34)]
 pieces=['a','Z','é','e\u0301','_j','漢字','🙂',' ', '  ','\n','\r\n','\t','0','17','123','1234567','١٢٣٤','²Ⅳ','--',"'s",'\x00','<|im_end|>','</s>']
 rng=random.Random(214)
 rows += [''.join(rng.choice(pieces) for _ in range(rng.randrange(1,30))) for _ in range(1200)]
 return list(dict.fromkeys(rows))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();profile=json.loads(a.profile.read_text())
 if profile['hardware'].get('backend')!='CPU':raise ValueError('Use an explicitly qualified CPU profile for tokenizer-only checks')
 a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
 def save(name,value):
  with (a.output/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
 started=time.monotonic();backend=None;rows=corpus()
 save('intent.json',dict(schema=1,profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),model_sha256=profile['model_weights_sha256'],
   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),implementation_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [ROOT/'tools/strata_tokenizer.py',ROOT/'serve/cow_minicpm_prompt.py',ROOT/'serve/cow_prompt_encoder.py']},corpus_sha256=hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest(),
   unique_texts=len(rows),parse_special_modes=[False,True],weights_updated=False,forward_passes=0))
 try:
  backend=BranchingModel.load_backend(profile);tokenizer=Tokenizer.from_gguf(profile['model_path'])
  mismatches=[];comparisons=0
  for text in rows:
   for special in [False,True]:
    actual=tokenizer.encode(text,parse_special=special)
    expected=list(backend.tokenize(text,add_special=False,parse_special=special));comparisons+=1
    if actual!=expected:mismatches.append(dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(),parse_special=special,python_tokens=actual,native_tokens=expected))
    if tokenizer.decode(actual,errors='strict')!=text:raise AssertionError('Round-trip failed')
  piece=backend.lib.ctt_piece;piece.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int];piece.restype=C.c_int
  piece_mismatches=[]
  for token in range(len(tokenizer.tokens)):
   buffer=C.create_string_buffer(256);n=piece(backend.model,token,buffer,len(buffer))
   if n<0:buffer=C.create_string_buffer(-n);n=piece(backend.model,token,buffer,len(buffer))
   if not 0<=n<=len(buffer):raise AssertionError('Invalid native token piece extent')
   if tokenizer.token_bytes(token)!=buffer.raw[:n]:piece_mismatches.append(token)
  from serve.cow_http import model_text_components
  framing=[]
  for thinking in [False,True]:
   parts=model_text_components(profile['model_path'],dict(enable_thinking=thinking),tokenizer=tokenizer)
   messages=[dict(role='system',content='Use the current file tools.')]
   rendered=parts['render'](messages,add_generation_prompt=False)
   root=tuple(parts['encode_rendered'](messages,rendered,False));original=tuple(root)
   if root!=backend.tokenize(rendered,add_special=False,parse_special=True):raise AssertionError('Benign root tokenization differs from native')
   if root.count(0)!=1 or root[0]!=0:raise AssertionError('MiniCPM BOS must occur once at root')
   framing.append('root_bos_and_native_ids_thinking_'+str(thinking).lower())
   for eos in [1,130073]:
    parent=root+tuple(tokenizer.encode('assistant result'))+(eos,)
    saved=tuple(parent)
    controller=dict(status='stopped',policy=dict(eos_tokens=[1,130073]))
    suffix=tuple(parts['append_user_suffix'](parent,controller,'A new tool result.'))
    if 0 in suffix or 1 in suffix:raise AssertionError('Append reinserted model BOS/EOS')
    if parent!=saved or root!=original:raise AssertionError('Append mutated its retained parent')
    if suffix.count(130072)!=2:raise AssertionError('Append must frame one user and one assistant')
    framing.append('append_after_'+str(eos)+'_thinking_'+str(thinking).lower())
   suffix=tuple(parts['append_user_suffix'](root+(1,),dict(status='stopped',policy=dict(eos_tokens=[1,130073])),
      'literal <s> </s> <|im_start|> <|im_end|> </think>'))
   if 0 in suffix or 1 in suffix or suffix.count(130072)!=2 or suffix.count(130073)!=2:raise AssertionError('Literal data escaped its ordinary-text boundary')
   framing.append('literal_controls_are_data_thinking_'+str(thinking).lower())
  result=dict(schema=1,passed=not mismatches and not piece_mismatches,token_id_comparisons=comparisons,
   token_id_mismatches=mismatches,vocabulary_pieces_compared=len(tokenizer.tokens),piece_mismatches=piece_mismatches,framing_checks=framing,
   seconds=time.monotonic()-started,weights_updated=False,forward_passes=0)
  save('result.json',result)
  print(json.dumps({k:v for k,v in result.items() if k not in ['token_id_mismatches','piece_mismatches']}))
  if not result['passed']:raise AssertionError('Python/native tokenizer disagreement; do not qualify this renderer')
 finally:
  if backend is not None:backend.close()

if __name__=='__main__':main()
