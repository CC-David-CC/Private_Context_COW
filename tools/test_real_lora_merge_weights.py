"""Independent streaming FP64 check of a real native F32 LoRA merge.

Owner-only read-only diagnostic; no training, model mutation or GPU allocation.
"""
from pathlib import Path
import argparse,hashlib,json,os,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from tools.gguf_reader import GGUFFile


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def adapter_matrices(a,b,weight_shape):
    """Recover only the unambiguous trailing rank-one dimension omitted by GGUF.

    B's explicit rank-one axis and the base matrix's input width must both
    confirm the interpretation. Do not flatten/reshape arbitrary bad geometry.
    """
    if (len(weight_shape)==2 and a.ndim==1 and b.ndim==2
            and b.shape==(weight_shape[0],1) and a.shape==(weight_shape[1],)):
        a=a.reshape(1,weight_shape[1])
    if (len(weight_shape)!=2 or a.ndim!=2 or b.ndim!=2
            or b.shape[1]!=a.shape[0] or (b.shape[0],a.shape[1])!=weight_shape):
        raise ValueError('Compatible explicit A/B and base matrix geometry required')
    return a,b


def audit(base,adapter,merged,*,base_sha256,adapter_sha256,merged_sha256,maximum_weight_error=2e-7):
    paths=[Path(base),Path(adapter),Path(merged)];expected_hashes=[base_sha256,adapter_sha256,merged_sha256]
    if any(sha(p)!=expected for p,expected in zip(paths,expected_hashes)):raise ValueError('Pinned input artifact differs')
    infos=[GGUFFile(p) for p in paths];maps=[{t.name:t for t in info.tensors} for info in infos]
    if any(len(index)!=len(info.tensors) for index,info in zip(maps,infos)) or maps[0].keys()!=maps[2].keys():
        raise ValueError('Unique complete base/merged tensor identity required')
    if any(t.type_name!='F32' for t in infos[2].tensors):raise ValueError('Merged storage must be F32')
    def view(index,name):
        tensor=maps[index][name]
        if tensor.type_name not in ('F16','F32'):raise ValueError('Floating source tensors required')
        return np.memmap(paths[index],mode='r',dtype={'F16':'<f2','F32':'<f4'}[tensor.type_name],
                         offset=infos[index].data_start+tensor.offset,shape=tuple(reversed(tensor.shape)))
    alpha=infos[1].metadata.get('adapter.lora.alpha');used=set();rows=[];elements=0;start=time.monotonic()
    for name,tensor in maps[0].items():
        if tensor.shape!=maps[2][name].shape:raise ValueError('Merged tensor geometry changed')
        w=view(0,name);actual=view(2,name);a_name=name+'.lora_a';b_name=name+'.lora_b';adapted=a_name in maps[1]
        maximum=0.;exact_frozen=True
        if adapted:
            if b_name not in maps[1] or w.ndim!=2:raise ValueError('Complete A/B matrix pair required')
            a=view(1,a_name).astype(np.float64);b=view(1,b_name)
            if alpha is None:
                raise ValueError('Explicit alpha and compatible adapter geometry required')
            a,b=adapter_matrices(a,b,w.shape)
            scale=float(alpha)/a.shape[0];used.update([a_name,b_name])
        for offset in range(0,w.shape[0],64):
            expected=w[offset:offset+64].astype(np.float64)
            if adapted:expected=expected+scale*(b[offset:offset+64].astype(np.float64)@a)
            observed=actual[offset:offset+64].astype(np.float64)
            if not np.isfinite(expected).all() or not np.isfinite(observed).all():raise ValueError('Nonfinite weight value')
            maximum=max(maximum,float(np.max(np.abs(expected-observed))))
            if not adapted:exact_frozen=exact_frozen and np.array_equal(expected,observed)
        row=dict(name=name,elements=tensor.elements,adapted=adapted,maximum_absolute_error=maximum,
                 passed=bool(maximum<=maximum_weight_error and (adapted or exact_frozen)))
        rows.append(row);elements+=tensor.elements
        del w,actual
        if adapted:del a,b
    if used!=maps[1].keys():raise ValueError('Unexpected/unconsumed adapter tensors')
    if any(sha(p)!=expected for p,expected in zip(paths,expected_hashes)):raise ValueError('Artifact changed during independent audit')
    return dict(schema=1,status='PASS' if all(r['passed'] for r in rows) else 'FAIL',
                scope='Every saved weight compared with independent FP64 W + alpha/rank * B@A; not a decoding-equivalence claim',
                base_sha256=base_sha256,adapter_sha256=adapter_sha256,merged_sha256=merged_sha256,
                tensors=len(rows),adapted_tensors=sum(r['adapted'] for r in rows),elements=elements,
                maximum_weight_error_bound=maximum_weight_error,maximum_weight_error=max(r['maximum_absolute_error'] for r in rows),
                frozen_tensors_bit_value_exact=all(r['passed'] for r in rows if not r['adapted']),checks=rows,
                wall_seconds=time.monotonic()-start,gpu_allocation_seconds=0,weights_updated=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','adapter','merged','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['base-sha256','adapter-sha256','merged-sha256']:p.add_argument('--'+name,required=True)
    a=p.parse_args();result=audit(a.base,a.adapter,a.merged,base_sha256=a.base_sha256,adapter_sha256=a.adapter_sha256,merged_sha256=a.merged_sha256)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    if result['status']!='PASS':raise SystemExit(1)
