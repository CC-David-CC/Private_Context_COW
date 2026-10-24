"""Offline exact-SVD diagnostics for explicitly selected saved weight matrices.

NumPy is a diagnostic reader, not a training/autograd runtime. Extraction permits
CPU analysis on another authorized Linux host without copying a whole model.
"""
from pathlib import Path
import argparse,hashlib,json,math,re,time

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def spectrum_metrics(matrix):
    import numpy as np
    a=np.asarray(matrix,dtype='float64')
    if a.ndim!=2 or not a.size or not np.isfinite(a).all():raise ValueError('A finite nonempty matrix is required')
    values=np.linalg.svd(a,compute_uv=False,full_matrices=False)
    energy=float(values@values);total=float(values.sum());top=float(values[0])
    p=values[values>0]/total if total else np.array([],dtype='float64')
    return dict(rows=a.shape[0],columns=a.shape[1],frobenius_norm=math.sqrt(energy),rms=math.sqrt(energy/a.size),spectral_norm=top,
                stable_rank=energy/(top*top) if top else 0.,entropy_effective_rank=float(np.exp(-(p*np.log(p)).sum())) if total else 0.,
                energy_participation_rank=energy*energy/float((values**4).sum()) if energy else 0.)

def extract(model,model_sha256,names,output):
    import numpy as np
    from tools.gguf_reader import GGUFFile
    model=Path(model).resolve(strict=True)
    if sha(model)!=model_sha256:raise ValueError('Pinned model differs')
    if not names or len(set(names))!=len(names) or any(not re.fullmatch('[A-Za-z0-9_.]+',name) for name in names):raise ValueError('Explicit distinct matrix names required')
    info=GGUFFile(model);by_name={t.name:t for t in info.tensors}
    tensors=[by_name[name] for name in names]
    if any(len(t.shape)!=2 or t.type_name not in ('F16','F32') for t in tensors):raise ValueError('Only complete F16/F32 matrices are qualified for extraction')
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False);rows=[]
    for index,t in enumerate(tensors):
        array=np.memmap(model,dtype='<f2' if t.type_name=='F16' else '<f4',mode='r',offset=info.data_start+t.offset,shape=tuple(reversed(t.shape)))
        if not np.isfinite(array).all():raise ValueError('Nonfinite saved matrix')
        name=str(index).zfill(3)+'.npy';target=root/name
        with target.open('xb') as stream:np.save(stream,np.asarray(array),allow_pickle=False)
        rows.append(dict(name=t.name,shape=list(array.shape),storage=t.type_name,file=name,sha256=sha(target)))
        del array
    if sha(model)!=model_sha256:raise ValueError('Model changed during extraction')
    receipt=dict(schema=1,kind='pcc.saved-matrix-subset.v1',model_sha256=model_sha256,matrices=rows,weights_updated=False)
    receipt['subset_id']=digest(receipt);write(root/'manifest.json',receipt);return receipt

def read_subset(root):
    root=Path(root);value=json.loads((root/'manifest.json').read_text())
    if value.get('schema')!=1 or value.get('kind')!='pcc.saved-matrix-subset.v1' or value.get('subset_id')!=digest({k:v for k,v in value.items() if k!='subset_id'}):raise ValueError('Pinned matrix subset differs')
    rows=value.get('matrices');names=set()
    if type(rows) is not list or not rows:raise ValueError('Nonempty matrix subset required')
    for row in rows:
        if row['name'] in names or not re.fullmatch('[0-9]+[.]npy',row['file']) or sha(root/row['file'])!=row['sha256']:raise ValueError('Matrix identity or saved bytes differ')
        names.add(row['name'])
    return value

def measure(subset,baseline_subset,output):
    import numpy as np
    subset=Path(subset);baseline_subset=Path(baseline_subset)
    current=read_subset(subset);base=read_subset(baseline_subset)
    left={r['name']:r for r in base['matrices']};right={r['name']:r for r in current['matrices']}
    if left.keys()!=right.keys() or any(left[k]['shape']!=right[k]['shape'] for k in left):raise ValueError('Compare the same complete selected matrices')
    start=time.monotonic();rows=[]
    for name,row in right.items():
        a=np.load(subset/row['file'],allow_pickle=False);b=np.load(baseline_subset/left[name]['file'],allow_pickle=False)
        if list(a.shape)!=row['shape'] or list(b.shape)!=left[name]['shape']:raise ValueError('Declared matrix shape differs')
        rows.append(dict(name=name,weights=spectrum_metrics(a),change_from_base=spectrum_metrics(a.astype('float64')-b.astype('float64'))))
    if read_subset(subset)!=current or read_subset(baseline_subset)!=base:raise ValueError('Saved diagnostic inputs changed during measurement')
    result=dict(schema=1,status='COMPLETE',method='Exact float64 SVD, all singular values of selected complete matrices',
                scope='Saved weights and accumulated change, not per-step update orthogonality or activation/feature rank',
                subset_id=current['subset_id'],baseline_subset_id=base['subset_id'],model_sha256=current['model_sha256'],baseline_model_sha256=base['model_sha256'],
                matrices=rows,wall_seconds=time.monotonic()-start,weights_updated=False,promoted=False)
    write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    e=commands.add_parser('extract');e.add_argument('--model',type=Path,required=True);e.add_argument('--model-sha256',required=True);e.add_argument('--names',nargs='+',required=True);e.add_argument('--output',type=Path,required=True)
    m=commands.add_parser('measure');m.add_argument('--subset',type=Path,required=True);m.add_argument('--baseline-subset',type=Path,required=True);m.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=extract(a.model,a.model_sha256,a.names,a.output) if a.command=='extract' else measure(a.subset,a.baseline_subset,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('matrices',)}))
