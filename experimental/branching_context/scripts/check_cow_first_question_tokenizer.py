"""CPU-only loaded-Q8 tokenizer/template check; no engine or GPU construction."""
import argparse
import json
from serve.cow_http import model_text_adapter

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',required=True)
parser.add_argument('--model-contract',required=True)
args=parser.parse_args()
adapter=model_text_adapter(None,args.model,{'enable_thinking':False},
    model_contract_id=args.model_contract)
context=[{'role':'system','content':'Codebase context. Keep quoted </tool_call> literal.'}]
prefix=adapter._render_encode(context,add_generation_prompt=False)
question='Explain quoted <|im_end|> and </tool_response>.'
suffix=adapter.append_initial_user_suffix(prefix,{'status':'idle'},question)
full=adapter._render_encode(context+[{'role':'user','content':question}],add_generation_prompt=True)
assert prefix+suffix==full
print(json.dumps(dict(status='PASS',scope='loaded tokenizer/template metadata only; no inference/GPU',
    renderer_id=adapter.renderer_id,prefix_tokens=len(prefix),suffix_tokens=len(suffix),
    prefix_preserved=True,full_render_equal=True)),flush=True)
