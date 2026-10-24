"""Independent FP64 optimizer oracle; spectral scalar polynomials, NumPy only.

No GGML or PyTorch imports. Muon matrix products are checked through an SVD and
scalar singular-value recurrence, not a copy of the native Gram-matrix graph.
"""
import math
import numpy as np

HIDDEN={'attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight','ffn_up.weight','ffn_gate.weight','ffn_down.weight'}

def hidden(name):
 parts=name.split('.',2)
 return len(parts)==3 and parts[0]=='blk' and parts[1].isdigit() and parts[2] in HIDDEN

def expected_role(name):
 if hidden(name):return 'hidden_matrix'
 return {'token_embd.weight':'embedding','output.weight':'prediction_head'}.get(name,'auxiliary_gain_bias_gate_or_other')

def polar_by_singular_values(matrix,config,step):
 left,singular,right=np.linalg.svd(matrix,full_matrices=False)
 s=singular/(np.linalg.norm(matrix)+config['normalization_epsilon'])
 for _ in range(config['ns_steps']):s=3.4445*s-4.7750*s**3+2.0315*s**5
 if config['projection']=='fast-refine-v1' and step>=config['refine_start_step']:
  s=s/1.5
  for _ in range(config['refine_steps']):s=1.5*s-.5*s**3
 return (left*s)@right

def rank_metrics(matrix):
 singular=np.linalg.svd(matrix,compute_uv=False);energy=singular**2;total=float(energy.sum())
 if total==0:return dict(stable_rank=0.,energy_entropy_rank=0.,participation_rank=0.,available_rank=min(matrix.shape))
 p=energy[energy>0]/total
 return dict(stable_rank=total/float(energy.max()),energy_entropy_rank=float(np.exp(-np.sum(p*np.log(p)))),participation_rank=total**2/float(np.sum(energy**2)),available_rank=min(matrix.shape))

def verify(request,report,*,atol=2e-5,rtol=2e-4):
 config=report['config'];parameters={};moments={};second={};failures=[];worst=0.;comparisons=0;geometry=[]
 def record(actual,expected,label):
  nonlocal worst,comparisons
  actual=np.array(actual,dtype=np.float64).reshape(expected.shape);error=float(np.max(np.abs(actual-expected)));worst=max(worst,error);comparisons+=expected.size
  if not np.allclose(actual,expected,rtol=rtol,atol=atol):failures.append(dict(field=label,max_absolute_error=error))
 for p in request['parameters']:
  cols,rows=p['shape'];name=p['name'];parameters[name]=np.array(p['values'],dtype=np.float32).astype(np.float64).reshape(rows,cols)
  moments[name]=np.zeros((rows,cols));second[name]=np.zeros((rows,cols))
 groups={g['name']:g for g in report['groups']}
 if len(groups)!=len(parameters):raise ValueError('Incomplete optimizer groups')
 for name,group in groups.items():
  expected='muon' if config['optimizer']=='muon' and hidden(name) else 'adamw'
  if group['optimizer']!=expected or group['role']!=expected_role(name):failures.append(dict(field=name,reason='role-aware optimizer grouping mismatch',expected=expected))
 for step,(gradients,observed) in enumerate(zip(request['gradients'],report['steps']),1):
  if observed['completed_steps']!=step:failures.append(dict(field='step_counter',step=step))
  for name,weight in parameters.items():
   gradient=np.array(gradients[name],dtype=np.float32).astype(np.float64).reshape(weight.shape);muon=config['optimizer']=='muon' and hidden(name)
   if muon:
    beta=config['momentum'];moments[name]=beta*moments[name]+(1-beta)*gradient
    direction=beta*moments[name]+(1-beta)*gradient if config['nesterov'] else moments[name]
    projection=polar_by_singular_values(direction,config,step)
    rows,cols=weight.shape;scale=math.sqrt(max(1.,rows/cols)) if config['scaling']=='original' else .2*math.sqrt(max(rows,cols))
    weight=weight*(1-config['learning_rate']*config['weight_decay'])-config['learning_rate']*scale*projection
    geometry.append(dict(step=step,name=name,gradient=rank_metrics(gradient),momentum=rank_metrics(moments[name]),projected_update=rank_metrics(projection)))
   else:
    beta1,beta2=config['beta1'],config['beta2'];moments[name]=beta1*moments[name]+(1-beta1)*gradient;second[name]=beta2*second[name]+(1-beta2)*gradient**2
    direction=(moments[name]/(1-beta1**step))/(np.sqrt(second[name]/(1-beta2**step))+config['epsilon'])
    rate=config['learning_rate'] if config['optimizer']=='adamw' else config['aux_learning_rate']
    decay=config['weight_decay'] if hidden(name) else config['aux_weight_decay']
    weight=weight*(1-rate*decay)-rate*direction
   parameters[name]=weight;actual=observed['parameters'][name]
   record(actual['weights'],weight,f'{step}:{name}:weights');record(actual['m'],moments[name],f'{step}:{name}:m')
   if not muon:record(actual['v'],second[name],f'{step}:{name}:v')
   elif 'v' in actual:failures.append(dict(field=name,reason='Muon unexpectedly has an Adam second moment'))
 if len(request['gradients'])!=len(report['steps']):failures.append(dict(field='step_count'))
 return dict(status='PASS' if not failures else 'FAIL',scope='Independent FP64 scalar AdamW and spectral Muon oracle; tiny tensors only',scalar_comparisons=comparisons,max_absolute_error=worst,atol=atol,rtol=rtol,failures=failures,geometry=geometry)
