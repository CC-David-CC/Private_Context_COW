"""Independent CPU-autograd check of the same 32-distance small graph.

PyTorch appears only as a gradient oracle here, not as native Wen's forward.
"""
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from math_reference import Schedule,Chunk,rotate,compose,compose_vjp,lora,lora_vjp,cross_entropy

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def main():
    rng = np.random.default_rng(20261022)
    schedule = Schedule.yarn([1.,.1],4.,[.2,.8])
    x = rng.normal(size=(3,6)); query = rng.normal(size=(1,6))
    wk,wv,wq,out = [rng.normal(size=s) for s in [(4,6),(3,6),(4,6),(7,3)]]
    parameters = [rng.normal(0,.1,size=s) for s in [(2,6),(4,2),(2,6),(3,2),(2,6),(4,2)]]
    distances = [0]+[int(round(v)) for v in np.geomspace(1,1048576,31)]
    assert len(set(distances)) == 32
    k,kt = lora(x,wk,*parameters[:2],2.)
    v,vt = lora(x,wv,*parameters[2:4],2.)
    q,qt = lora(query,wq,*parameters[4:],2.)
    chunk = Chunk.create(rotate(k,np.arange(len(k)),schedule),v,schedule,'test-writer')
    gradients = [np.zeros_like(p) for p in parameters]
    values = []
    for distance in distances:
        position = len(k)+distance
        answer,tape = compose(rotate(q,position,schedule),[(chunk,0)],schedule,[position],'test-writer')
        value,dlogits = cross_entropy(answer@out.T,[2])
        dq,dc = compose_vjp(tape,dlogits@out/len(distances))
        dk,dv = dc[chunk.key]
        for offset,layer_tape,g in [(0,kt,rotate(dk,-np.arange(len(k)),schedule)),
                                    (2,vt,dv),(4,qt,rotate(dq,-position,schedule))]:
            _,da,db = lora_vjp(layer_tape,g)
            gradients[offset] += da; gradients[offset+1] += db
        values.append(value)
    tensors = [torch.tensor(p,requires_grad=True) for p in parameters]
    def dense(xx,ww,aa,bb):return xx@torch.tensor(ww).T+(xx@aa.T)@bb.T
    def rope(xx,positions):
        phase = torch.as_tensor(positions)[...,None]*torch.tensor(schedule.frequencies)
        c,s = torch.cos(phase),torch.sin(phase)
        a,b = xx[...,:2],xx[...,2:]
        return torch.cat((c*a-s*b,s*a+c*b),dim=-1)
    kk = rope(dense(torch.tensor(x),wk,*tensors[:2]),torch.arange(3))
    vv = dense(torch.tensor(x),wv,*tensors[2:4])
    qq = dense(torch.tensor(query),wq,*tensors[4:])
    losses = []
    for distance in distances:
        qglobal = rope(qq,3+distance)
        p = torch.softmax(qglobal@kk.T/2.,dim=-1)
        logits = (p@vv)@torch.tensor(out).T
        losses.append(-torch.log_softmax(logits,dim=-1)[0,2])
    total = torch.stack(losses).mean();total.backward()
    np.testing.assert_allclose(float(total.detach()),np.mean(values),rtol=1e-12,atol=1e-12)
    manual = np.concatenate([g.ravel() for g in gradients])
    oracle = np.concatenate([p.grad.detach().numpy().ravel() for p in tensors])
    relative = float(np.linalg.norm(manual-oracle)/np.linalg.norm(oracle))
    cosine = float(np.dot(manual,oracle)/(np.linalg.norm(manual)*np.linalg.norm(oracle)))
    assert relative < 1e-10 and cosine > 1-1e-12
    print(json.dumps(dict(distances=32,maximum_distance=max(distances),relative_gradient_error=relative,
                         gradient_cosine=cosine,loss=float(total.detach()),device='CPU',
                         scope='same small independent Writer/Reader graph; not native Wen backward')))


if __name__ == '__main__': main()
