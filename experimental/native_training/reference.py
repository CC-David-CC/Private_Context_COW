"""Independent NumPy/FP64 oracle for the exported tiny layer fixtures.

This is verification, not the training runtime. It does not import PyTorch,
GGML, cached activations or any native gradient implementation.
"""
import argparse
import json
from pathlib import Path
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def silu(x):
    return x*sigmoid(x)

def layer_value(fixture, layer, weights):
    n=fixture['tokens'];hidden=len(fixture['input'])//n
    x=np.array(fixture['input'],dtype=np.float64).reshape(n,hidden).T
    upstream=np.array(fixture['upstream'],dtype=np.float64).reshape(n,hidden).T
    p=f'blk.{layer}.'
    def w(name):return weights[p+name]
    def linear(name,value):return w(name)@value
    def norm(value,name):
        return value/np.sqrt(np.mean(value*value,axis=0,keepdims=True)+1e-6)*w(name).reshape(-1,1)
    normalized=norm(x,'attn_norm.weight')
    if p+'ssm_a' in weights:
        state_size=w('ssm_norm.weight').size
        heads=w('ssm_a').size
        conv_weights=w('ssm_conv1d.weight')
        channels,kernel=conv_weights.shape
        key_heads=(channels//state_size-heads)//2
        projected=linear('attn_qkv.weight',normalized)
        convolved=np.zeros_like(projected)
        for t in range(n):
            for lag in range(kernel):
                pos=t-kernel+1+lag
                if pos>=0:convolved[:,t]+=conv_weights[:,lag]*projected[:,pos]
        convolved=silu(convolved)
        key_rows=state_size*key_heads
        q=convolved[:key_rows].reshape(key_heads,state_size,n)
        k=convolved[key_rows:2*key_rows].reshape(key_heads,state_size,n)
        v=convolved[2*key_rows:].reshape(heads,state_size,n)
        q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6)
        k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
        beta=sigmoid(linear('ssm_beta.weight',normalized))
        alpha=linear('ssm_alpha.weight',normalized)+w('ssm_dt.bias').reshape(-1,1)
        decay=np.exp(np.logaddexp(0,alpha)*w('ssm_a').reshape(-1,1))
        state=np.zeros((heads,state_size,state_size),dtype=np.float64)
        output=np.zeros((heads,state_size,n),dtype=np.float64)
        for t in range(n):
            for head in range(heads):
                key=k[head%key_heads,:,t];query=q[head%key_heads,:,t]
                state[head]*=decay[head,t]
                delta=(v[head,:,t]-state[head]@key)*beta[head,t]
                state[head]+=np.outer(delta,key)
                output[head,:,t]=state[head]@query/np.sqrt(state_size)
        output=output/np.sqrt(np.mean(output*output,axis=1,keepdims=True)+1e-6)
        output*=w('ssm_norm.weight').reshape(1,-1,1)
        gate=linear('attn_gate.weight',normalized).reshape(heads,state_size,n)
        attn=linear('ssm_out.weight',(output*silu(gate)).reshape(heads*state_size,n))
    else:
        d=w('attn_q_norm.weight').size
        heads=w('attn_q.weight').shape[0]//(2*d)
        key_heads=w('attn_k.weight').shape[0]//d
        qg=linear('attn_q.weight',normalized).reshape(heads,2*d,n)
        q=qg[:,:d].copy();gate=qg[:,d:]
        k=linear('attn_k.weight',normalized).reshape(key_heads,d,n)
        v=linear('attn_v.weight',normalized).reshape(key_heads,d,n)
        q=q/np.sqrt(np.mean(q*q,axis=1,keepdims=True)+1e-6)*w('attn_q_norm.weight').reshape(1,d,1)
        k=k/np.sqrt(np.mean(k*k,axis=1,keepdims=True)+1e-6)*w('attn_k_norm.weight').reshape(1,d,1)
        # The exported tiny fixture uses normal RoPE with rotary dimension two.
        assert d==2,'This independent oracle is explicitly scoped to the tiny fixture'
        angle=np.arange(n,dtype=np.float64);co=np.cos(angle);si=np.sin(angle)
        for a in (q,k):
            even=a[:,0,:].copy();odd=a[:,1,:].copy()
            a[:,0,:]=even*co-odd*si;a[:,1,:]=even*si+odd*co
        output=np.zeros((heads,d,n),dtype=np.float64)
        for head in range(heads):
            kh=head//(heads//key_heads)
            for t in range(n):
                scores=k[kh,:,:t+1].T@q[head,:,t]/np.sqrt(d)
                prob=np.exp(scores-np.max(scores));prob/=prob.sum()
                output[head,:,t]=v[kh,:,:t+1]@prob
        attn=linear('attn_output.weight',(output*sigmoid(gate)).reshape(heads*d,n))
    residual=x+attn
    post=norm(residual,'post_attention_norm.weight')
    result=residual+linear('ffn_down.weight',linear('ffn_up.weight',post)*silu(linear('ffn_gate.weight',post)))
    return float(np.sum(result*upstream))

def verify(report):
    results=[]
    for case in report['checks']:
        fixture=case['fixture'];layer=case['layer']
        weights={name:np.array(v['values'],dtype=np.float64).reshape(v['shape'][1],v['shape'][0])
                 for name,v in fixture['weights'].items()}
        native={d['name']:np.array(d['gradient'],dtype=np.float64).reshape(weights[d['name']].shape)
                for d in case['directions']}
        forward=layer_value(fixture,layer,weights)
        # Independent smooth FP64 derivatives, with an explicit sensitivity check.
        worst=0.;sensitivity=0.;scalars=0;failures=[]
        for name,values in weights.items():
            gradients=[]
            for epsilon in (1e-6,2e-6):
                numeric=np.zeros_like(values)
                for index in np.ndindex(values.shape):
                    original=values[index]
                    values[index]=original+epsilon;plus=layer_value(fixture,layer,weights)
                    values[index]=original-epsilon;minus=layer_value(fixture,layer,weights)
                    values[index]=original;numeric[index]=(plus-minus)/(2*epsilon)
                gradients.append(numeric)
            err=float(np.max(np.abs(gradients[0]-native[name])))
            delta=float(np.max(np.abs(gradients[0]-gradients[1])))
            worst=max(worst,err);sensitivity=max(sensitivity,delta);scalars+=values.size
            if not np.allclose(gradients[0],native[name],rtol=5e-4,atol=5e-5):
                failures.append(dict(name=name,max_absolute_error=err))
            if not np.allclose(gradients[0],gradients[1],rtol=1e-5,atol=1e-7):
                failures.append(dict(name=name,finite_difference_sensitivity=delta))
        if not np.isclose(forward,case['baseline'],rtol=5e-4,atol=5e-6):
            failures.append(dict(forward_fp64=forward,native=case['baseline']))
        results.append(dict(layer=layer,scalar_gradients=scalars,fp64_forward=forward,
                            max_absolute_gradient_error=worst,step_sensitivity=sensitivity,
                            failures=failures))
    return dict(status='PASS' if all(not r['failures'] for r in results) else 'FAIL',
                scope='independent FP64 finite differences of exported tiny hybrid layers',
                native_backend=report['backend'],checks=results)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('report',type=Path)
    args=parser.parse_args();result=verify(json.loads(args.report.read_text()))
    print(json.dumps(result,indent=2));raise SystemExit(result['status']!='PASS')
