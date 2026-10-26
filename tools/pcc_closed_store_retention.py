"""Owner-side retention of archived, closed COW evaluation stores.

No inference, model training, workflow scheduling or model/state deletion.
Only the dedicated JSON/lock store beneath one completed evaluation is eligible.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,stat,tarfile,time
from tools.pcc_training_workspace import private_root,sha,write

PREFIX='development/cow-store/'
MAX_STORE_BYTES=8<<30

def read(path):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('Duplicate retention metadata field')
            value[key]=item
        return value
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs)

def sync(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)

def owned_group(path):
    """A grouping folder may inherit umask; actual stores/archives stay private."""
    path=Path(path);info=path.lstat()
    if not stat.S_ISDIR(info.st_mode)or info.st_uid!=os.getuid()or path.resolve()!=path.absolute():
        raise ValueError('Real owned retention grouping directory required')
    return path.resolve()

def durable(path,value):
    write(path,value);sync(Path(path).parent)

def owner_live(owner):
    if (type(owner)is not dict or type(owner.get('pid'))is not int or owner['pid']<=0
            or not str(owner.get('start_ticks','')).isdigit()):
        raise ValueError('Exact original evaluator process identity required')
    if owner.get('boot_id',Path('/proc/sys/kernel/random/boot_id').read_text().strip())!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return False
    try:fields=Path('/proc',str(owner['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return str(fields[19])==str(owner['start_ticks'])and fields[0]!='Z'

def inventory(store):
    if not store.exists():return {}
    private_root(store);value={}
    for path in sorted(store.rglob('*')):
        info=path.lstat()
        if stat.S_ISLNK(info.st_mode):raise ValueError('Linked COW store entry refused')
        if stat.S_ISDIR(info.st_mode):continue
        if (not stat.S_ISREG(info.st_mode)or info.st_uid!=os.getuid()or info.st_nlink!=1
                or path.suffix not in('.json','.lock')):
            raise ValueError('Only owned JSON/lock store files are supported')
        value[path.relative_to(store).as_posix()]=dict(bytes=info.st_size,sha256=sha(path))
    return value

def archive_inventory(archive,manifest):
    """Check the bytes inside the archive, not just its outer checksum."""
    expected={name[len(PREFIX):]:value for name,value in manifest['files'].items()if name.startswith(PREFIX)}
    if not expected:raise ValueError('Archive has no original COW store inventory')
    if any(Path(name).is_absolute()or'..'in Path(name).parts or Path(name).suffix not in('.json','.lock')for name in expected):
        raise ValueError('Unsupported archived COW store entry')
    found={};total=0
    archive=Path(archive);info=archive.lstat()
    if not stat.S_ISREG(info.st_mode)or info.st_uid!=os.getuid()or info.st_nlink!=1:
        raise ValueError('Owned non-aliased point archive required')
    fd=os.open(archive,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb')as stream:
        before=os.fstat(stream.fileno())
        if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['archive_sha256']:
            raise ValueError('Original point archive checksum changed')
        stream.seek(0)
        with tarfile.open(fileobj=stream,mode='r|gz')as reader:
            for member in reader:
                if not member.name.startswith(PREFIX):continue
                name=member.name[len(PREFIX):]
                if member.isdir():continue
                if not member.isfile()or name not in expected or name in found:
                    raise ValueError('Unexpected archived store member')
                total+=member.size
                if total>MAX_STORE_BYTES:raise ValueError('Closed store archive exceeds retention bound')
                raw=reader.extractfile(member);digest=hashlib.file_digest(raw,'sha256').hexdigest()
                if digest!=expected[name]:raise ValueError('Archived store bytes differ from preserved inventory')
                found[name]=dict(bytes=member.size,sha256=digest)
        after=os.fstat(stream.fileno())
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Point archive changed during inspection')
        os.fsync(stream.fileno())
    if set(found)!=set(expected):raise ValueError('Original archived store members are missing')
    sync(Path(archive).parent)
    return found

def retire_store(workspace,point,evidence):
    """One explicit, idempotent retirement; original results and bytes stay archived."""
    workspace=private_root(workspace);point=private_root(point);evidence=private_root(evidence)
    if not point.is_relative_to(workspace):raise ValueError('Evaluation point lies outside owned workspace')
    store=point/'development/cow-store';intent=evidence/'closed-store-retirement-intent.json'
    if store.exists()and store.resolve()!=store:raise ValueError('Linked COW store ancestor refused')
    complete=evidence/'closed-store-retirement-complete.json'
    if complete.exists():return read(complete)
    if not store.exists()and not intent.exists():return dict(status='SKIP',reason='no cached store')
    handle=point/'development-process.json'
    if not handle.exists():handle=point/'evaluation-process.json'
    records=[point/'absolute-point.json',point/'development/result.json',point/'development/settlement.json',handle]
    if not all(p.is_file()and not p.is_symlink()for p in records):return dict(status='SKIP',reason='original completion records unavailable')
    if owner_live(read(handle)):return dict(status='SKIP',reason='original evaluator is still live')
    result=read(records[1]);settled=read(records[2]);score=read(records[0])
    if (result.get('status')!='COMPLETE'or settled.get('status')!='COMPLETE'
            or type(result.get('cases'))is not int or result['cases']<=0
            or settled.get('completed_cases')!=result['cases']or result.get('weights_updated')is not False
            or any(score.get(k)!=result.get(k)for k in('passed','finished','runtime_id'))):
        return dict(status='SKIP',reason='complete original assessment not proven')
    manifest_path=evidence/'evidence-verification.json'
    if manifest_path.is_symlink():raise ValueError('Linked archive proof refused')
    if not manifest_path.is_file():return dict(status='SKIP',reason='completed archive unavailable')
    try:manifest=read(manifest_path)
    except json.JSONDecodeError:return dict(status='SKIP',reason='archive publication in progress')
    if manifest.get('status')!='PRESERVED':return dict(status='SKIP',reason='archive not preserved')
    for p in records:
        if manifest['files'].get(p.relative_to(point).as_posix())!=sha(p):
            raise ValueError('Original assessment records differ from archived proof')
    wanted=archive_inventory(evidence/'point-evidence.tar.gz',manifest)
    remaining=inventory(store)
    if (not set(remaining)<=set(wanted)or any(v!=wanted[k]for k,v in remaining.items())
            or not intent.exists()and remaining!=wanted):
        raise ValueError('Current COW store changed from its archived bytes')
    expected=dict(schema=1,store=str(store),point=str(point),archive_sha256=manifest['archive_sha256'],
        files=wanted,producer=read(handle),completion_records_sha256={p.relative_to(point).as_posix():sha(p)for p in records},
        models_deleted=0,optimizer_states_deleted=0,raw_case_response_json_retained=True)
    if intent.exists():
        if read(intent)!=expected:raise ValueError('Original retirement intent changed')
    else:durable(intent,expected)
    if owner_live(read(handle)):raise ValueError('Original evaluator became live during retention')
    if inventory(store)!=remaining:raise ValueError('COW store changed during retention')
    for name in remaining:
        path=store/name
        if path.is_symlink()or not path.resolve().is_relative_to(store):raise ValueError('COW store path changed')
        path.unlink()
    if store.exists():
        for d in sorted((p for p in store.rglob('*')if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
            if d.is_symlink():raise ValueError('Linked store directory refused')
            d.rmdir()
        store.rmdir();sync(store.parent)
    value=dict(schema=1,status='CLOSED_STORE_RETIRED',bytes_retired=sum(v['bytes']for v in wanted.values()),
        store=str(store),archive_sha256=manifest['archive_sha256'],models_deleted=0,optimizer_states_deleted=0,
        raw_case_response_json_retained=True,archive_member_bytes_verified=True)
    durable(complete,value);return value

def run_pass(policy):
    fields={'schema','workspace','groups','keep_latest','maximum_stores_per_pass','maintenance_lock'}
    if (type(policy)is not dict or not fields<=set(policy)or set(policy)-fields-{'completion_points'}
            or type(policy['schema'])is not int or policy['schema']!=1
            or type(policy['keep_latest'])is not int or not 1<=policy['keep_latest']<=8
            or type(policy['maximum_stores_per_pass'])is not int or not 1<=policy['maximum_stores_per_pass']<=128):
        raise ValueError('Closed bounded owner retention policy required')
    if (type(policy['groups'])is not list or not 1<=len(policy['groups'])<=32
            or any(type(policy[k])is not str or not Path(policy[k]).is_absolute()for k in('workspace','maintenance_lock'))):
        raise ValueError('Explicit owned retention groups and absolute paths required')
    workspace=private_root(policy['workspace']);lock=Path(policy['maintenance_lock']);private_root(lock.parent)
    if lock.is_symlink():raise ValueError('Linked maintenance lease refused')
    values=[];requested_complete=True
    if 'completion_points'in policy:
        targets=policy['completion_points']
        if type(targets)is not list or not 1<=len(targets)<=128:raise ValueError('Explicit bounded completion points required')
        for name in targets:
            if type(name)is not str or not Path(name).is_absolute()or not Path(name).resolve().is_relative_to(workspace):
                raise ValueError('Completion point outside owned workspace')
            point=Path(name)
            files=[point/'absolute-point.json',point/'development/result.json',point/'development/settlement.json']
            if not all(p.is_file()and not p.is_symlink()for p in files):requested_complete=False;continue
            score,result,settled=map(read,files)
            requested_complete=requested_complete and score.get('update')==68 and result.get('status')=='COMPLETE'and settled.get('status')=='COMPLETE'and result.get('cases')==settled.get('completed_cases')
    with lock.open('a+')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in policy['groups']:
            if (type(group)is not dict or set(group)!={'points','evidence','arms'}
                    or type(group['arms'])is not list or not 1<=len(group['arms'])<=64):raise ValueError('Closed retention group required')
            points=owned_group(group['points']);evidence=owned_group(group['evidence'])
            if not points.is_relative_to(workspace):raise ValueError('Retention group outside workspace')
            for arm in group['arms']:
                if type(arm)is not str or re.fullmatch('[a-z0-9][a-z0-9-]{0,127}',arm)is None:raise ValueError('Explicit bounded arm name required')
                candidates=sorted(p for p in(points/arm).glob('update-*')if p.is_dir())
                last=points/arm/'update-068/absolute-point.json'
                if 'completion_points'not in policy:requested_complete=requested_complete and last.exists()
                for point in candidates[:-policy['keep_latest']]:
                    e=evidence/arm/point.name
                    if not e.is_dir()or(e/'closed-store-retirement-complete.json').exists():continue
                    if not(point/'development/cow-store').exists()and not(e/'closed-store-retirement-intent.json').exists():continue
                    if any(e.glob('cow-store-retirement-*-intent.json')):continue
                    value=retire_store(workspace,point,e)
                    if value['status']=='CLOSED_STORE_RETIRED':values.append(value)
                    if len(values)>=policy['maximum_stores_per_pass']:
                        return dict(status='PASS',stores=len(values),bytes_retired=sum(v['bytes_retired']for v in values),all_requested_update68_records_present=False)
    return dict(status='PASS',stores=len(values),bytes_retired=sum(v['bytes_retired']for v in values),all_requested_update68_records_present=requested_complete)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-passes',type=int,default=1);parser.add_argument('--interval-seconds',type=int,default=60)
    args=parser.parse_args()
    if not 1<=args.maximum_passes<=1440 or not 1<=args.interval_seconds<=3600:parser.error('Finite positive pass and interval bounds required')
    policy=read(args.policy);args.output.mkdir(mode=0o700,parents=False,exist_ok=False)
    try:
        for number in range(args.maximum_passes):
            value=run_pass(policy);durable(args.output/('pass-%04d.json'%number),value);print(json.dumps(value),flush=True)
            if value['all_requested_update68_records_present']:break
            if number+1<args.maximum_passes:time.sleep(args.interval_seconds)
        durable(args.output/'exit.json',dict(status='COMPLETE',passes=number+1,scope='Finite owner storage maintenance; no training or evaluation scheduling'))
    except BaseException as exc:
        durable(args.output/'exit.json',dict(status='NEEDS_REVIEW',error=type(exc).__name__+': '+str(exc),automatic_retry=False));raise
if __name__=='__main__':main()
