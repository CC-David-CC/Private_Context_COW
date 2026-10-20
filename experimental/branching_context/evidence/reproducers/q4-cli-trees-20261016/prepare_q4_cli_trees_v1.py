"""Prepare CLI-system-role source fixtures under the current execution profile."""
import hashlib,json,sys
from pathlib import Path
root=Path('/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1')
runtime=Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime')
profile_path=runtime.parent/'profile/profile.json'
old=Path('/home/dflanag3/spatial-artifacts/cow-client-pipeline-integration-20261015-v1/fixtures-q4-cold-cache-v1')
output=root/'fixtures-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(profile_path)=='a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755'
profile=json.loads(profile_path.read_bytes())
sys.path[:0]=[str(runtime/'tools'),str(runtime),str(runtime/'experimental/branching_context/src')]
from gguf_reader import GGUFFile
from tools.strata_tokenizer import Tokenizer
from tools.cow_cli import _messages_file
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.frontend import ChatTemplate
from serve.server import MockEngine,Service
from serve.cow_prefix import exact_append_suffix
modules={}
for name in ('tools.cow_cli','tools.strata_tokenizer','serve.cow_prompt_encoder','serve.frontend','serve.cow_prefix'):
    module=sys.modules[name];path=Path(module.__file__).resolve();assert path.is_relative_to(runtime)
    modules[name]=dict(path=str(path),sha256=sha(path))
manifest=json.loads((old/'fixture-set.json').read_bytes())
assert manifest['schema']=='cow-browser-context-set-v1' and len(manifest['contexts'])==3
metadata=GGUFFile(profile['model_path']).metadata;template=metadata['tokenizer.chat_template']
tokenizer=Tokenizer.from_gguf(profile['model_path'])
output.mkdir(mode=0o700,parents=True)
template_path=output/'chat_template.jinja';template_path.write_text(template,encoding='utf-8',newline='\n')
encoder=CowPromptEncoder(tokenizer,ChatTemplate(template_path),profile['execution']['text_template_kwargs'])
service=Service(MockEngine(tokenizer,'fixture-tokenization-only'),tokenizer,ChatTemplate(template_path))
rows=[]
def encode(messages,gen):
    rendered=encoder.render(messages,add_generation_prompt=gen)
    tokens=tuple(encoder.encode_rendered(messages,rendered,gen))
    ordinary=tuple(service.encode_prompt(messages,None,{**profile['execution']['text_template_kwargs'],'add_generation_prompt':gen}))
    assert ordinary==tokens
    return list(tokens)
for i,entry in enumerate(manifest['contexts']):
    original=(old/entry['fixture_file']).resolve();assert original.is_relative_to(old)
    assert sha(original)==entry['fixture_sha256']
    data=json.loads(original.read_bytes());name=entry['name']
    assert name==data['name'] and len(data['root_messages'])==1
    case=output/(str(i)+'-'+name);case.mkdir(mode=0o700)
    context=case/'context.txt';context.write_text(data['root_messages'][0]['content'],encoding='utf-8',newline='\n')
    messages=_messages_file(context);assert messages==[dict(role='system',content=context.read_text())]
    base=encode(messages,False);assert 6000<=len(base)<=12000
    branches={}
    for label in ('A','B'):
        question=data['branches'][label]['question']
        full=encode(messages+[dict(role='user',content=question)],True)
        suffix=list(encoder.append_initial_user_suffix(tuple(base),dict(status='idle',policy={'eos_tokens':profile['execution']['eos_tokens']}),question))
        assert full==base+suffix and list(exact_append_suffix(base,full))==suffix
        assert 400<=len(suffix)<=700
        branches[label]=dict(question=question,suffix_token_ids=suffix,full_token_ids=full,suffix_tokens=len(suffix))
    assert branches['A']['suffix_token_ids']!=branches['B']['suffix_token_ids']
    row=dict(name=name,index=i,branch='Tree'+str(i),context_file=str(context),context_sha256=sha(context),
        root_messages=messages,root_token_ids=base,root_tokens=len(base),branches=branches,
        source_fixture_path=str(original),source_fixture_sha256=sha(original),source_receipts=data['source_receipts'])
    with (case/'fixture.json').open('x') as f:json.dump(row,f,sort_keys=True,indent=2);f.write('\n')
    rows.append(dict(name=name,index=i,fixture=str(case/'fixture.json'),fixture_sha256=sha(case/'fixture.json'),root_tokens=len(base),suffix_tokens={k:v['suffix_tokens'] for k,v in branches.items()}))
record=dict(schema='cow-cli-three-source-fixtures-v1',status='PREPARED_NOT_RUN',scope='Actual CLI system-context framing, CPU tokenizer/template parity only; no native state or model inference',
    profile_path=str(profile_path),profile_sha256=sha(profile_path),execution_profile_id=profile['execution_profile_id'],
    model_contract_id=profile['model_contract_id'],contexts=rows,imports=modules,
    source_scope='Three distinct Strata source bundles, not three unrelated repositories; old source text reused as data and retokenized, no old native checkpoint imported',
    builder_sha256=sha(Path(__file__)),template_sha256=sha(template_path))
with (output/'fixture-set.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('imports',)},sort_keys=True))
