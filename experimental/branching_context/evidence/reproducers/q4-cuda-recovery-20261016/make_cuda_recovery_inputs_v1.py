"""Create token fixtures from canonical token IDs only; never import a checkpoint."""
import hashlib,json,os
from pathlib import Path
root=Path('/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1')
source=Path('/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/input-store-v1')
cursor='808cc601aae3e0a9f6f8abeafc04a4e600ef1d04c8af261280aff57c25687c85'
seen=set();chunks=[]
while cursor is not None:
    assert cursor not in seen
    seen.add(cursor)
    raw=(source/'objects'/(cursor+'.json')).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==cursor
    value=json.loads(raw);chunks.append(value['inputs']);cursor=value['parent']
ids=[token for chunk in reversed(chunks) for token in chunk]
assert len(ids)==7999
inputs=root/'inputs'
inputs.mkdir(mode=0o700)
receipt={}
for name,tokens in [('tiny',[100+i%17 for i in range(128)]),('target',ids[:7995])]:
    fixture=dict(base_tokens=tokens,generated_prefix=4,followup_outputs=8,stream='cuda-recovery-fixture',
        token_provenance='deterministic 100+i%17' if name=='tiny' else 'first7995 canonical input IDs from verified old Q4 snapshot; numerical state NOT imported')
    raw=(json.dumps(fixture,sort_keys=True,separators=(',',':'))+'\n').encode()
    path=inputs/(name+'.json')
    with path.open('xb') as f:f.write(raw)
    os.chmod(path,0o400)
    receipt[name]=dict(path=str(path),base_tokens=len(tokens),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
print(json.dumps(receipt,sort_keys=True))
with (inputs/'receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')