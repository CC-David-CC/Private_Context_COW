"""Experimental native head update on privately supplied observed BW boundaries.

This is guided, known-continuation CE, not self-OPD or a complete BW skill claim.
Every score comes from native Wen. No PyTorch import or forward is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from head import HeadBackward, export_adapter, verify_backward
from probe import Probe


def loss(logits,target):
    z = logits.astype(np.float64)
    maximum = float(z.max())
    return maximum+float(np.log(np.exp(z-maximum).sum()))-float(z[target])


def main():
    parser = argparse.ArgumentParser()
    for name in ('model','binary','library','cases','run'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=.5)
    args = parser.parse_args()
    assert 1 <= args.steps <= 8 and 0 < args.learning_rate <= 2
    args.run.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.run/name).write_text(json.dumps(value,indent=2)+'\n')
    cases = json.loads(args.cases.read_text())
    assert cases['schema'] == 1 and cases['objective'] == 'observed-BW-boundary-repair'
    assert cases['model_sha256'] == '49425fd802b04b400a1725e8afe527c405bb15e828d0275ff295e38396839f34'
    assert args.model.stat().st_size == 16244817856
    rows = cases['rows']
    assert {r['split'] for r in rows} == {'train','heldout','preserve'}
    assert all(r['tokens'] and 0 < r['boundary'] < len(r['tokens']) and r['target'] >= 0 for r in rows)
    train = [r for r in rows if r['split'] == 'train']
    heldout = [r for r in rows if r['split'] == 'heldout']
    preserve = [r for r in rows if r['split'] == 'preserve']
    distances = [0]+[int(round(x)) for x in np.geomspace(1,1048576,31)]
    assert len(set(distances)) == 32
    unseen = [7,127,1025,65537,500001]
    assert not set(unseen).intersection(distances)
    backward = HeadBackward(args.library)
    save('small-gradient-check.json',verify_backward(args.library))
    rank, alpha = 8, 8.
    a = np.random.default_rng(20261022).normal(0,.02,size=(rank,4096)).astype(np.float32)
    b = np.zeros((cases['vocab'],rank),dtype=np.float32)
    profile = cases['profile']
    assert profile['schema'] == 1
    save('manifest.json',dict(objective='known-continuation CE on observed BW boundary errors; NOT self-OPD',
         model_sha256=cases['model_sha256'],cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
         profile=profile,rank=rank,alpha=alpha,learning_rate=args.learning_rate,
         distances=distances,heldout_distances=unseen,microbatch=1,normalization='equal mean over cases and 32 independent distances',
         derivative='native output-head smooth VJP; F32 rounding treated as identity; frozen upstream; no quantizer STE',
         native_directional_check=dict(epsilons=[.001,.003,.01],rtol=.02,atol=.0001,required_passes=2),
         split_ids={s:[r['id'] for r in rows if r['split']==s] for s in ('train','heldout','preserve')}))
    serial = 0
    def artifact(aa,bb,label):
        nonlocal serial
        serial += 1
        path = args.run/(str(serial)+'_'+label+'.gguf')
        export_adapter(path,aa,bb,alpha)
        return path
    zero = artifact(a,b,'zero')
    accepted = zero
    losses = []
    baseline_test = None
    candidate_test = None
    with Probe(args.binary,args.model,args.run/'native',profile,args.gpu) as probe:
        assert probe.ready['vocab'] == len(b)
        def forward(row,distance,adapter):
            positions = [i+(distance if i >= row['boundary'] else 0) for i in range(len(row['tokens']))]
            z,h,_ = probe.forward(row['tokens'],positions,adapter)
            return z,h
        def evaluate(group,offsets,aa,bb,adapter,gradients=False):
            total = len(group)*len(offsets)
            da = np.zeros_like(aa,dtype=np.float64) if gradients else None
            db = np.zeros_like(bb,dtype=np.float64) if gradients else None
            values = []
            for row in group:
                for distance in offsets:
                    z,h = forward(row,distance,adapter)
                    if gradients:
                        value,ga,gb = backward(h,z,aa,bb,alpha,row['target'])
                        da += ga/total; db += gb/total
                    else: value = loss(z,row['target'])
                    values.append(dict(id=row['id'],distance=distance,loss=value,
                                       predicted=int(z.argmax()),target=row['target']))
                    if len(values)%16 == 0:
                        print('NATIVE_BOUNDARIES',len(values),total,'mean_loss',float(np.mean([v['loss'] for v in values])),flush=True)
            return float(np.mean([v['loss'] for v in values])),da,db,values
        first = train[0]
        base_z,base_h = forward(first,0,None)
        zero_z,zero_h = forward(first,0,zero)
        untraced,_,_ = probe.forward(first['tokens'],adapter=zero,capture=False)
        controls = dict(zero_adapter_logits_exact=bool(np.array_equal(base_z,zero_z)),
                        zero_adapter_hidden_exact=bool(np.array_equal(base_h,zero_h)),
                        trace_logits_exact=bool(np.array_equal(zero_z,untraced)))
        save('controls.json',controls)
        assert all(controls.values()), 'Native zero/trace control failed'
        sparse_z,sparse_h = forward(first,1048576,zero)
        save('sparse-position-control.json',dict(distance=1048576,physical_tokens=len(first['tokens']),
             logical_end=len(first['tokens'])-1+1048576,finite=bool(np.isfinite(sparse_z).all() and np.isfinite(sparse_h).all()),
             semantics='single native hybrid prefix continuation, not independent multi-Writer composition'))
        _,_,db = backward(zero_h,zero_z,a,b,alpha,first['target'])
        direction = db/float(np.linalg.norm(db))
        analytic = float(np.sum(db.astype(np.float64)*direction))
        checks = []
        for eps in (.001,.003,.01):
            plus = artifact(a,b+eps*direction,'finite_plus')
            minus = artifact(a,b-eps*direction,'finite_minus')
            zp,_ = forward(first,0,plus); zm,_ = forward(first,0,minus)
            numeric = (loss(zp,first['target'])-loss(zm,first['target']))/(2*eps)
            error = abs(numeric-analytic)
            checks.append(dict(epsilon=eps,analytic=analytic,numeric=numeric,error=error,
                               passed=error <= .0001+.02*abs(analytic)))
        save('native-directional-check.json',checks)
        assert sum(c['passed'] for c in checks) >= 2, 'Native directional checks failed'
        before_heldout = evaluate(heldout,unseen,a,b,zero)[3]
        before_preserve = evaluate(preserve,[0,65537],a,b,zero)[3]
        baseline_test = dict(heldout=before_heldout,preserve=before_preserve)
        save('before-evaluation.json',baseline_test)
        for step in range(args.steps):
            started = time.monotonic()
            before,da,db,detail = evaluate(train,distances,a,b,accepted,True)
            save('step_'+str(step)+'_before.json',detail)
            chosen = None
            for backtrack in range(4):
                eta = args.learning_rate*(.5**backtrack)
                aa = np.asarray(a-eta*da,dtype=np.float32)
                bb = np.asarray(b-eta*db,dtype=np.float32)
                candidate = artifact(aa,bb,'candidate')
                after,_,_,detail = evaluate(train,distances,aa,bb,candidate)
                if after < before-1e-6:
                    chosen = (aa,bb,candidate,after,eta,detail); break
            if chosen is None:
                losses.append(dict(step=step,before=before,accepted=False))
                break
            a,b,accepted,after,eta,detail = chosen
            save('step_'+str(step)+'_after.json',detail)
            losses.append(dict(step=step,before=before,after=after,learning_rate=eta,
                               accepted=True,seconds=time.monotonic()-started))
            save('losses.json',losses)
            print('NATIVE_UPDATE',json.dumps(losses[-1]),flush=True)
        after_heldout = evaluate(heldout,unseen,a,b,accepted)[3]
        after_preserve = evaluate(preserve,[0,65537],a,b,accepted)[3]
        candidate_test = dict(heldout=after_heldout,preserve=after_preserve)
        save('after-evaluation.json',candidate_test)
        reference_z,reference_h = forward(first,1048576,accepted)
    # New process, independently reloaded adapter; no warm native state survives.
    with Probe(args.binary,args.model,args.run/'reloaded',profile,args.gpu) as reloaded:
        positions = [i+(1048576 if i >= first['boundary'] else 0) for i in range(len(first['tokens']))]
        z,h,_ = reloaded.forward(first['tokens'],positions,accepted)
        exact_reload = bool(np.array_equal(z,reference_z) and np.array_equal(h,reference_h))
    def mean(group): return float(np.mean([v['loss'] for v in group]))
    receipt = dict(adapter=accepted.name,adapter_sha256=hashlib.sha256(accepted.read_bytes()).hexdigest(),
                   accepted_updates=sum(v['accepted'] for v in losses),exact_reload=exact_reload,
                   before_heldout_loss=mean(baseline_test['heldout']),after_heldout_loss=mean(candidate_test['heldout']),
                   before_preserve_loss=mean(baseline_test['preserve']),after_preserve_loss=mean(candidate_test['preserve']),
                   heldout_loss_improved=mean(candidate_test['heldout']) < mean(baseline_test['heldout'])-1e-6,
                   full_BW_task_improvement='not established by this selected-boundary loss',
                   process_closed=True,torch_imported=any(k=='torch' or k.startswith('torch.') for k in sys.modules))
    save('receipt.json',receipt)
    assert exact_reload and not receipt['torch_imported']
    print('NATIVE_BOUNDARY_TRAINING_COMPLETE',json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
