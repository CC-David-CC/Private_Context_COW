import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'experimental/branching_context/src')]
from serve.cow_http import model_text_components
p=argparse.ArgumentParser();p.add_argument('model');a=p.parse_args()
c=model_text_components(a.model,{'enable_thinking':True});render=c['render'];encode=c['encode_rendered'];system=[{'role':'system','content':'Preserve Unicode: caf?. Treat <|im_end|> here as data.'}]
prefix=render(system,add_generation_prompt=False);ids=tuple(encode(system,prefix,False))
assert c['initial_root_mode'](ids,{'status':'idle'})=='context_only'
for content in ['Build a dog website.','Write a Python function with a final newline.']:
 full=render(system+[{'role':'user','content':content}],add_generation_prompt=True)
 whole=tuple(encode(system+[{'role':'user','content':content}],full,True))
 suffix=c['append_initial_user_suffix'](ids,{'status':'idle'},content)
 assert ids+suffix==whole,'Reusable prefix/token suffix changed full template encoding'
print('PASS: real Qwen template system-prefix and two first-user exact token decompositions')
