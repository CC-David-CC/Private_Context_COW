"""Known BW failures, light guidance, and empty logical gaps; no training yet.

All task content and generated answers are private operational artifacts.
This one-prefix hybrid continuation is not independent multi-Writer composition.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from probe import Probe
from bw_validation import verify

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.gguf_reader import GGUFFile
from tools.strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--bw-source',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--ids',default='train-writing-tool,train-writing-bytes,heldout-writing-tool,heldout-writing-bytes,preserve-read-notes,preserve-ascii,preserve-identity')
    parser.add_argument('--distances',default='0')
    parser.add_argument('--max-new-tokens',type=int,default=256)
    args = parser.parse_args()
    sys.path.insert(0,str(args.bw_source.resolve()))
    args.run.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.run/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    selected = args.ids.split(',')
    cases = [c for c in cases if c['id'] in selected]
    assert set(selected) == {c['id'] for c in cases}
    metadata = GGUFFile(args.model).metadata
    tokenizer = Tokenizer.from_gguf(args.model)
    eos = {value for key,value in metadata.items() if key.startswith('tokenizer.ggml.') and key.endswith(('eos_token_id','eot_token_id','eom_token_id'))}
    template_path = args.run/'chat-template.jinja'
    template_path.write_text(metadata['tokenizer.chat_template'],encoding='utf-8',newline='')
    template = ChatTemplate(template_path)
    profile = dict(schema=1,rope_factor=4.,rope_original_context=metadata['wen1.context_length'],
                   physical_capacity=8192,prefill_chunk=64,kv_dtype='F16',flash_attention=True,
                   semantics='single native hybrid prefix; recurrent state retained; empty logical gaps have no recurrent steps')
    distances = [int(value) for value in args.distances.split(',')]
    assert distances and len(distances) == len(set(distances)) and min(distances) >= 0 and max(distances) <= 1048576
    rows = []
    prepared = []
    for case in cases:
        for guided in (False,True) if case['family'] == 'bounded-writing' else (False,):
            messages = copy.deepcopy(case['messages'])
            if guided:
                # A procedural hint, not the reference answer or paragraph.
                messages[-1]['content'] += ('\n\nCheck before answering: count whitespace-separated words and revise the paragraph to the requested count. '
                    'Use one create_text_file JSON proposal. Close the entire outer JSON object before ending. '
                    'The write is still pending; do not return finish.')
            text = template.render(messages,enable_thinking=False)
            ids = tokenizer.encode(text,parse_special=True)
            prefix = tokenizer.encode(template.render(messages,add_generation_prompt=False,enable_thinking=False),parse_special=True)
            assert ids[:len(prefix)] == prefix, 'Generation boundary crosses a token; choose an explicit compiled boundary'
            if not guided: assert ids == case['prompt_ids'], 'Existing BW prompt IDs changed'
            prepared.append((case,guided,ids,len(prefix)))
    save('manifest.json',dict(model_file=args.model.name,expected_model_sha256='49425fd802b04b400a1725e8afe527c405bb15e828d0275ff295e38396839f34',
         source_commit=__import__('subprocess').check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
         cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),profile=profile,
         distances=distances,training=False,scope='known BW light-guidance evaluation; not independent multi-chunk composition'))
    with Probe(args.binary,args.model,args.run/'process',profile,args.gpu) as probe:
        # Same-schedule instrumentation must not manufacture different logits.
        case,guided,ids,boundary = prepared[0]
        assert probe.tokenize(template.render(case['messages'],enable_thinking=False)) == ids
        traced,_,_ = probe.forward(ids)
        untraced,_,_ = probe.forward(ids,capture=False)
        controls = dict(traced_untraced_exact=bool(np.array_equal(traced,untraced)),
                        traced_untraced_max_abs=float(np.max(np.abs(traced-untraced))))
        save('controls.json',controls)
        assert controls['traced_untraced_exact'], 'Trace perturbed native logits'
        for case,guided,ids,boundary in prepared:
            for distance in distances:
                positions = [i+(distance if i >= boundary else 0) for i in range(len(ids))]
                started = time.monotonic()
                result = probe.generate(ids,positions,maximum=args.max_new_tokens)
                text = tokenizer.decode([token for token in result['tokens'] if token not in eos])
                label = case['id']+('_guided' if guided else '_base')+'_d'+str(distance)
                verdict = verify(case,text,args.run/'checks'/label)
                row = dict(id=case['id'],split=case['split'],family=case['family'],guided=guided,
                           distance=distance,physical_prompt_tokens=len(ids),logical_prompt_end=positions[-1],
                           generated_tokens=result['tokens'],text=text,verdict=verdict,
                           seconds=time.monotonic()-started)
                rows.append(row); save('results.json',rows)
                print(json.dumps({k:row[k] for k in ('id','guided','distance','physical_prompt_tokens','logical_prompt_end','verdict','seconds')}),flush=True)
    save('summary.json',dict(training=False,process_closed=True,total=len(rows),passed=sum(row['verdict']['passed'] for row in rows),
         groups=[dict(guided=guided,distance=distance,total=len(group),passed=sum(row['verdict']['passed'] for row in group))
                 for guided in (False,True) for distance in distances
                 if (group:=[row for row in rows if row['guided']==guided and row['distance']==distance])]))
    print('GUIDANCE_EVALUATION_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
