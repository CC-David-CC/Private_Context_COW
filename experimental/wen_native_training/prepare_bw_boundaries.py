"""Prepare private boundary lessons from actual native BW failure receipts.

No arithmetic examples, invented failures, new skills or bundled answers.
The syntax target closes one observed missing brace; it does not correct prose.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.gguf_reader import GGUFFile
from tools.strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate


def main():
    parser = argparse.ArgumentParser()
    for name in ('model','cases','observations','manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    cases = {c['id']:c for c in json.loads(args.cases.read_text(encoding='utf-8'))}
    observations = json.loads(args.observations.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text())
    metadata = GGUFFile(args.model).metadata
    tokenizer = Tokenizer.from_gguf(args.model)
    # Reuse the exact extracted, immutable template from the observed run.
    template = ChatTemplate(args.manifest.parent/'chat-template.jinja')
    assert template.source == metadata['tokenizer.chat_template']
    eos = metadata['tokenizer.ggml.eos_token_id']
    close = tokenizer.encode('}',parse_special=True)
    assert len(close) == 1
    def boundary(case):
        prefix = tokenizer.encode(template.render(case['messages'],add_generation_prompt=False,enable_thinking=False),parse_special=True)
        assert case['prompt_ids'][:len(prefix)] == prefix
        return len(prefix)
    rows = []
    for row in observations:
        if row['guided'] or row['distance'] != 0: continue
        case = cases[row['id']]
        if case['family'] != 'bounded-writing': continue
        assert not row['verdict']['passed'] and case['split'] in ('train','heldout')
        assert row['generated_tokens'][-1] == eos
        response = row['generated_tokens'][:-1]
        assert tokenizer.decode(response) == row['text']
        corrected = row['text']+'}'
        action = json.loads(corrected)  # Reject any repair requiring more than the observed missing brace.
        assert action['tool'] == 'create_text_file' and action['arguments']['path'] == case['path']
        assert tokenizer.decode(response+close) == corrected
        paragraph = ' '.join(action['arguments'].get('lines',[]))
        rows.append(dict(id=case['id']+'_close',split=case['split'],family='observed-outer-JSON-closure',
                         source_case=case['id'],tokens=case['prompt_ids']+response,
                         boundary=boundary(case),target=close[0],
                         observed_full_task_pass=False,closure_target_valid_json=True,
                         observed_words=len(paragraph.split()),requested_words=case['word_count']))
    assert sum(r['split']=='train' for r in rows) == 2
    assert sum(r['split']=='heldout' for r in rows) == 2
    # Positive examples with a different target prevent a universal-brace lesson.
    # These are already reviewed passing skill families from the original train split.
    for key in ('train-unicode-lf','train-times-three'):
        case = cases[key]; assert case['split'] == 'train'
        json.loads(case['correction'])
        response = tokenizer.encode(case['correction'],parse_special=True)
        rows.append(dict(id=key+'_finish',split='train',family='finish-valid-JSON',source_case=key,
                         tokens=case['prompt_ids']+response,boundary=boundary(case),target=eos))
    for row in observations:
        if row['guided'] or row['distance'] != 0 or row['split'] != 'preserve':continue
        case = cases[row['id']]
        assert row['verdict']['passed'] and row['generated_tokens'][-1] == eos
        rows.append(dict(id=case['id']+'_finish',split='preserve',family='finish-valid-JSON',source_case=case['id'],
                         tokens=case['prompt_ids']+row['generated_tokens'][:-1],boundary=boundary(case),target=eos))
    assert sum(r['split']=='preserve' for r in rows) == 3
    result = dict(schema=1,objective='observed-BW-boundary-repair',
                  model_sha256=manifest['expected_model_sha256'],vocab=len(metadata['tokenizer.ggml.tokens']),
                  profile=manifest['profile'],
                  observations_sha256=hashlib.sha256(args.observations.read_bytes()).hexdigest(),rows=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(prepared=len(rows),ids=[(r['id'],r['split'],len(r['tokens'])) for r in rows])),flush=True)


if __name__ == '__main__': main()
