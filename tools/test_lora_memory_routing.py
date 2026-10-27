"""Check persistent admission bytes against completed native tensor artifacts."""
from pathlib import Path
import argparse,json,subprocess
from tools.gguf_reader import GGUFFile


def run(executable,completed,output):
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False);rows=[]
    for kind in ('adamw','muon'):
        prior=Path(completed)/kind/'first'
        request=json.loads((prior/'request-private.json').read_bytes())
        result=subprocess.run([str(executable),'--admit-training',str(prior/'request-private.json')],
                              capture_output=True,text=True,timeout=120)
        (root/(kind+'-stderr.log')).write_text(result.stderr,encoding='utf8')
        assert result.returncode==0,result.stderr
        plan=json.loads(result.stdout)
        (root/(kind+'-admission.json')).write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
        receipt=json.loads((prior/'stdout-private.json').read_bytes())
        base_bytes=sum(t.elements*4 for t in GGUFFile(request['model']).tensors)
        factor_bytes=sum(t.elements*4 for t in GGUFFile(prior/'candidate.gguf').tensors)
        state_bytes=receipt['optimizer_checkpoint']['state_bytes']
        expected=base_bytes+2*factor_bytes+state_bytes
        assert sum(row['persistent'] for row in plan['memory_plan'])==expected
        assert plan['weights_loaded'] is False and plan['weights_updated'] is False
        rows.append(dict(optimizer=kind,status='PASS',persistent_bytes=expected,
                         optimizer_state_bytes=state_bytes,model_weights_loaded=False))
    assert rows[1]['optimizer_state_bytes']<rows[0]['optimizer_state_bytes']
    result=dict(status='PASS',scope='Persistent memory routing checked against completed tiny-model artifacts',cases=rows)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True)
    p.add_argument('--completed',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.executable,a.completed,a.output)))
