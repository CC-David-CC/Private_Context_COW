"""Resolve an owner-selected native candidate to exact local training artifacts.

This is an offline reader. It grants no GPU, job, workflow or promotion authority.
Inputs with paths are owner configuration, never client/model tool arguments.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, math, os, re, stat, struct

from tools.pcc_native_training import read_optimizer_state, optimizer_dataset_sha256, validate_examples
from tools.pcc_cow_profile import bind_dense_profile, bind_dense_adapter
from tools.gguf_reader import GGUFFile


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}',value) is not None


def _effective_settings(config):
    """Read native-f32-v1's canonical header as its strict input vocabulary.

    Headers include inactive defaults. Native admission forbids passing those
    knobs to an algorithm that does not use them; never silently discard a
    non-default value. The native loader still checks the entire canonical state.
    """
    defaults=dict(schema=1,optimizer='adamw',precision='native-f32-v1',learning_rate=0,
        aux_learning_rate=struct.unpack('<f',struct.pack('<f',.0003))[0],
        beta1=0,beta2=0,epsilon=0,momentum=struct.unpack('<f',struct.pack('<f',.95))[0],
        nesterov=True,ns_steps=5,projection='jordan5',scaling='original',weight_decay=0,
        aux_weight_decay=0,normalization_epsilon=struct.unpack('<f',struct.pack('<f',1e-7))[0],
        refine_start_step=1,refine_steps=5)
    if set(config)!=set(defaults) or config['schema']!=1 or config['precision']!='native-f32-v1':
        raise ValueError('Known complete native-f32-v1 optimizer header required')
    inactive=set()
    if config['optimizer']=='adamw':
        inactive.update(('aux_learning_rate','momentum','nesterov','ns_steps','projection','scaling','normalization_epsilon','refine_start_step','refine_steps'))
    elif config['optimizer']=='muon':
        if config['projection']=='jordan5':inactive.update(('refine_start_step','refine_steps'))
        elif config['projection']!='fast-refine-v1':raise ValueError('Unknown saved Muon projection')
    else:raise ValueError('Unknown adaptive optimizer header')
    if any(type(config[key]) is not type(defaults[key]) or config[key]!=defaults[key] for key in inactive):
        raise ValueError('Inactive optimizer header fields cannot hide changed settings')
    return {k:v for k,v in config.items() if k not in inactive|{'optimizer','learning_rate'}}


def validate_reference(value):
    if type(value) is not dict or not _sha(value.get('base_model_sha256')):
        raise ValueError('Explicit measured base-model identity required')
    common={'base_model_sha256','binding_kind'}
    kind=value.get('binding_kind')
    if kind=='base-weights-only':
        if set(value)!=common:raise ValueError('Legacy base-only reference cannot contain a partial composite binding')
    elif kind=='measured-composite':
        if set(value)!=common|{'candidate_model_contract_id','candidate_adapters'} or not _sha(value['candidate_model_contract_id']):
            raise ValueError('Complete measured composite reference required')
        adapters=value['candidate_adapters']
        if type(adapters) is not list or len(adapters)>1:
            raise ValueError('Native training currently qualifies one adapter or base-only weights')
        for adapter in adapters:
            if (type(adapter) is not dict or set(adapter)!={'sha256','scale'} or not _sha(adapter['sha256'])
                    or type(adapter['scale']) not in (int,float) or not math.isfinite(adapter['scale']) or adapter['scale']!=1):
                raise ValueError('Native warm start requires the exact single adapter at scale one')
    else:raise ValueError('Unknown candidate reference binding')
    return deepcopy(value)


class _Inputs:
    def __init__(self,roots):
        if not roots:raise ValueError('Explicit owner-configured artifact roots required')
        self.roots=[];self.files={}
        for value in roots:
            p=Path(value)
            if not p.is_absolute() or p.is_symlink() or not p.is_dir() or p.resolve()==Path('/'):
                raise ValueError('Absolute regular artifact directories required')
            p=p.resolve(strict=True)
            if os.name=='posix' and (p.stat().st_uid!=os.getuid() or p.stat().st_mode&0o002):
                raise ValueError('Artifact roots must be owned and not world writable')
            self.roots.append(p)

    @staticmethod
    def stamp(path):
        s=path.stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

    def path(self,value):
        if not isinstance(value,(str,Path)) or not Path(value).is_absolute():
            raise ValueError('Owner artifacts require absolute paths')
        p=Path(value)
        roots=[r for r in self.roots if p.is_relative_to(r)]
        if not roots:raise ValueError('Artifact is outside configured roots')
        root=max(roots,key=lambda r:len(r.parts))
        for node in [p,*p.parents]:
            if node.is_symlink():raise ValueError('Artifact paths cannot traverse symlinks')
            if node==root:break
        if p.resolve(strict=True)!=p or not stat.S_ISREG(p.stat().st_mode):
            raise ValueError('An existing regular owner artifact is required')
        if os.name=='posix' and p.stat().st_uid!=os.getuid():raise ValueError('Foreign-owned artifact refused')
        stamp=self.stamp(p)
        if p in self.files and self.files[p]!=stamp:raise ValueError('Artifact changed during resolution')
        self.files[p]=stamp
        return p

    def json(self,value):
        p=self.path(value)
        if p.stat().st_size>64*1024**2:raise ValueError('Bounded metadata file required')
        raw=p.read_bytes()
        return p,json.loads(raw),hashlib.sha256(raw).hexdigest()

    def pinned(self,value,expected):
        if not _sha(expected):raise ValueError('Pinned artifact SHA256 required')
        p=self.path(value)
        with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Native artifact bytes differ from their registered identity')
        return p

    def unchanged(self):
        for p,stamp in self.files.items():
            if p.is_symlink() or not p.is_file() or self.stamp(p)!=stamp:
                raise ValueError('Original artifact disappeared or changed during resolution')


def resolve_parent(reference,version_file,profile_file,*,artifact_roots,state_policy,
                   base_model=None,checkpoint_file=None,examples=None):
    """Bind verified local files without a guessed path or silent moment reset.

    Fresh returns weight fields only: the caller must choose its optimizer.
    Resume additionally returns the exact saved optimizer recipe and data cursor.
    All execution still goes through the existing PCC admission and executor.
    """
    reference=validate_reference(reference)
    if state_policy not in ('fresh','resume'):raise ValueError('Choose fresh moments or exact same-stream resume explicitly')
    if state_policy=='fresh' and examples is not None:raise ValueError('Fresh resolution does not consume a resume practice stream')
    inputs=_Inputs(artifact_roots)
    version_path,version,version_sha=inputs.json(version_file)
    profile_path,template,profile_sha=inputs.json(profile_file)
    if type(version) is not dict or type(template) is not dict:
        raise ValueError('Native version and profile metadata objects required')
    mode=version.get('mode');receipt=version.get('receipt',{})
    if (type(version.get('schema')) is not int or version['schema']!=1 or type(receipt) is not dict
            or mode not in ('full','lora') or not _sha(version.get('base_sha256'))
            or not _sha(version.get('candidate_sha256')) or receipt.get('native') is not True
            or receipt.get('mode')!=mode or receipt.get('backprop_mode')!='standard'
            or receipt.get('status')!='MECHANICS_EXECUTED'):
        raise ValueError('A completed registered native standard-backprop candidate is required')
    candidate_path=version['candidate_path'];state_path=version.get('optimizer_state_path')
    checkpoint_sha=None
    if checkpoint_file is not None:
        _,saved,checkpoint_sha=inputs.json(checkpoint_file)
        if type(saved) is not dict:raise ValueError('Checkpoint metadata object required')
        files=saved.get('files')
        if (type(saved.get('schema')) is not int or saved['schema']!=1 or saved.get('status')!='VERIFIED' or type(files) is not list
                or len(files)!=2 or any(type(row) is not dict for row in files)
                or sorted(str(row.get('role')) for row in files)!=['model','optimizer']
                or saved.get('optimizer_completed_steps')!=version.get('optimizer_completed_steps')):
            raise ValueError('Existing PCC verified model-and-moments checkpoint required')
        by_role={row['role']:row for row in files}
        if (by_role['model']['sha256']!=version['candidate_sha256']
                or by_role['optimizer']['sha256']!=version.get('optimizer_state_sha256')):
            raise ValueError('Durable checkpoint identifies a different candidate or optimizer')
        candidate_path=by_role['model']['path'];state_path=by_role['optimizer']['path']
        for role in ('model','optimizer') if state_policy=='resume' else ('model',):
            p=inputs.path(by_role[role]['path'])
            if type(by_role[role].get('bytes')) is not int or p.stat().st_size!=by_role[role]['bytes']:
                raise ValueError('Durable artifact byte count differs')
    candidate=inputs.pinned(candidate_path,version['candidate_sha256'])
    info=GGUFFile(candidate);meta=info.metadata
    if (meta.get('general.architecture')!='llama' or meta.get('pcc.training.base_sha256')!=version['base_sha256']):
        raise ValueError('Saved candidate does not identify its native Llama training base')
    if not info.tensors or any(t.type_name!='F32' for t in info.tensors):
        raise ValueError('Saved native F32 training candidate required; no implicit dequantization')
    adapters=reference.get('candidate_adapters',[])
    if mode=='full':
        if adapters or reference['base_model_sha256']!=version['candidate_sha256'] or meta.get('general.type')=='adapter':
            raise ValueError('Full parent must be the exact measured full weights, without a hat')
        if base_model is not None:raise ValueError('Full parent uses its own complete weights; no extra base path')
        model=candidate
        fields=dict(mode='full',model=str(model),base_sha256=version['candidate_sha256'])
    else:
        if (reference['binding_kind']!='measured-composite' or len(adapters)!=1
                or adapters[0]['sha256']!=version['candidate_sha256'] or reference['base_model_sha256']!=version['base_sha256']):
            raise ValueError('LoRA parent requires its exact measured base and adapter')
        if base_model is None:raise ValueError('Owner must supply the pinned original base for LoRA')
        model=inputs.pinned(base_model,version['base_sha256'])
        base_metadata=GGUFFile(model).metadata
        if base_metadata.get('general.architecture')!='llama' or base_metadata.get('general.type')=='adapter':
            raise ValueError('LoRA requires an actual complete ordinary-Llama base')
        alpha=meta.get('adapter.lora.alpha')
        if (meta.get('general.type')!='adapter' or meta.get('adapter.type')!='lora'
                or type(alpha) not in (int,float) or not math.isfinite(alpha) or not 1<=alpha<=256 or int(alpha)!=alpha):
            raise ValueError('Native LoRA rank/alpha metadata required')
        fields=dict(mode='lora',model=str(model),base_sha256=version['base_sha256'],
                    initial_adapter=str(candidate),initial_adapter_sha256=version['candidate_sha256'],rank=int(alpha))
    # Reuse PCC's COW identity implementation, never a second hash convention.
    execution=template['execution'];gpu=execution['GPU']
    if type(gpu) is not list or len(gpu)!=1:raise ValueError('Existing single-device dense identity template required')
    bound=bind_dense_profile(template,model_path=model,model_sha256=fields['base_sha256'],
                             gpu_index=gpu[0],gpu_uuid=template['hardware']['gpu_uuid'])
    if mode=='lora':bound=bind_dense_adapter(bound,adapter_path=candidate,adapter_sha256=version['candidate_sha256'],scale=1.)
    if reference['binding_kind']=='measured-composite' and bound['model_contract_id']!=reference['candidate_model_contract_id']:
        raise ValueError('Requested measured model contract differs from the actual PCC weight/adapter binding')
    state_fields={};state_summary=None
    if state_policy=='resume':
        if state_path is None or examples is None:raise ValueError('Exact optimizer state and explicit same-stream examples required')
        p=inputs.path(state_path);state=read_optimizer_state(p,version['optimizer_state_sha256'])
        sequence=receipt.get('sequence_profile','dense-llama-short-v1')
        from tools.pcc_native_training import LLAMA_DENSE_PROFILES
        if sequence not in LLAMA_DENSE_PROFILES:
            raise ValueError('Completed ordinary-Llama optimizer sequence profile required')
        checked=validate_examples(examples,sequence_profile=sequence)
        saved=receipt.get('optimizer_checkpoint',{});config=state['config']
        if type(saved) is not dict or type(config) is not dict:raise ValueError('Completed optimizer configuration required')
        if (state['model_sha256']!=version['candidate_sha256']
                or saved.get('candidate_sha256')!=version['candidate_sha256']
                or saved.get('sha256')!=version['optimizer_state_sha256']
                or state['completed_steps']!=version.get('optimizer_completed_steps')
                or saved.get('completed_steps')!=state['completed_steps']
                or state['example_cursor']!=receipt.get('example_cursor')
                or state['dataset_sha256']!=saved.get('dataset_sha256')
                or state['dataset_sha256']!=optimizer_dataset_sha256(checked)
                or config!=saved.get('config') or config.get('optimizer')!=receipt.get('optimizer')
                or config.get('optimizer') not in ('adamw','muon') or (mode=='lora' and config['optimizer']!='adamw')):
            raise ValueError('Exact optimizer candidate, recipe, practice stream and cursor must agree')
        state_fields=dict(optimizer=config['optimizer'],learning_rate=config['learning_rate'],
            optimizer_settings=_effective_settings(config),
            optimizer_state_mode='resume',initial_optimizer_state=str(p),
            initial_optimizer_state_sha256=version['optimizer_state_sha256'],example_cursor=state['example_cursor'])
        state_summary=dict(sha256=version['optimizer_state_sha256'],completed_steps=state['completed_steps'],
                           example_cursor=state['example_cursor'],dataset_sha256=state['dataset_sha256'])
    inputs.unchanged()
    result=dict(schema=1,kind='pcc.resolved-training-parent.v1',status='RESOLVED',reference=reference,
        candidate_version_sha256=version_sha,profile_template_sha256=profile_sha,
        checkpoint_receipt_sha256=checkpoint_sha,native_parent_fields=fields,
        optimizer_state_fields=state_fields,state_policy=state_policy,optimizer_state=state_summary,
        model_contract_id=bound['model_contract_id'],candidate_sha256=version['candidate_sha256'],
        requires_explicit_fresh_optimizer=state_policy=='fresh',private_paths=True,
        gpu_used=False,dispatched=False,inference_qualified=False,promoted=False)
    result['resolution_id']=digest(result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','version','profile','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--artifact-root',action='append',type=Path,required=True)
    parser.add_argument('--base-model',type=Path);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--state-policy',choices=('fresh','resume'),required=True)
    parser.add_argument('--examples',type=Path,help='Explicit native example array for same-stream resume')
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('Resolution output already exists; inspect it instead of overwriting')
    try:
        reference_bytes=args.reference.read_bytes()
        example_bytes=None if args.examples is None else args.examples.read_bytes()
        value=resolve_parent(json.loads(reference_bytes),args.version,args.profile,
            artifact_roots=args.artifact_root,state_policy=args.state_policy,base_model=args.base_model,
            checkpoint_file=args.checkpoint,examples=None if example_bytes is None else json.loads(example_bytes))
        if args.reference.read_bytes()!=reference_bytes or (args.examples is not None and args.examples.read_bytes()!=example_bytes):
            raise ValueError('Reference or resume example input changed during resolution')
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        descriptor=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    except (OSError,ValueError,KeyError,TypeError) as exc:parser.exit(2,'Cannot resolve training parent: '+str(exc)+'\n')
    print(json.dumps(dict(status=value['status'],resolution_id=value['resolution_id'],
                         state_policy=value['state_policy'],gpu_used=False,dispatched=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
