"""Owner-side bounded working storage for native learning experiments.

No model math, client-selected paths, workflow ledger or acceptance policy.
The caller supplies an owned private workspace and measured artifact identities.
"""
from pathlib import Path
import contextlib,fcntl,hashlib,json,math,os,shutil,tarfile,time


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def private_root(path):
    path=Path(path)
    if path.is_symlink() or not path.is_dir() or path.stat().st_uid!=os.getuid() or path.stat().st_mode&0o077:
        raise ValueError('Private owned workspace required')
    return path.resolve()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def atomic(path,value):
    path=Path(path);temporary=path.with_suffix('.pending')
    with temporary.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


class ResourceUnavailable(RuntimeError):pass


class ResourceBudget:
    """Conservative campaign allocation and prospective RAM admission."""
    def __init__(self,root):self.root=private_root(root)
    @contextlib.contextmanager
    def locked(self):
        with (self.root/'resource.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield
    def reserve(self,operation,*,maximum_gpu_seconds,ram_bytes,free_ram_bytes,reserve_ram_bytes,owner):
        if (type(operation) is not str or not operation or type(maximum_gpu_seconds) not in (int,float)
                or not math.isfinite(maximum_gpu_seconds) or maximum_gpu_seconds<=0
                or any(type(x) is not int or x<0 for x in [ram_bytes,free_ram_bytes,reserve_ram_bytes])):
            raise ValueError('Explicit bounded phase resource request required')
        with self.locked():
            path=self.root/'resource-budget.json';data=json.loads(path.read_text())
            if operation in data['active'] or operation in data['settled']:raise ValueError('Existing original phase requires reconciliation')
            committed=data['prior_upper_bound_gpu_seconds']+data['charged_gpu_seconds']+sum(r['maximum_gpu_seconds'] for r in data['active'].values())
            if committed+maximum_gpu_seconds>data['ceiling_gpu_seconds']:raise ResourceUnavailable('Campaign allocation ceiling has no unreserved capacity')
            if free_ram_bytes-sum(r['ram_bytes'] for r in data['active'].values())<ram_bytes+reserve_ram_bytes:
                raise ResourceUnavailable('Working storage lacks its unreserved headroom')
            data['active'][operation]=dict(maximum_gpu_seconds=maximum_gpu_seconds,ram_bytes=ram_bytes,owner=owner,started_epoch=time.time())
            atomic(path,data)
    def settle(self,operation,allocated_gpu_seconds=None):
        with self.locked():
            path=self.root/'resource-budget.json';data=json.loads(path.read_text())
            if operation in data['settled']:return data['settled'][operation]
            record=data['active'][operation]
            # Unknown work retains the entire reservation as a conservative
            # charge. Reconciliation must inspect its original child handles.
            charge=record['maximum_gpu_seconds'] if allocated_gpu_seconds is None else allocated_gpu_seconds
            if type(charge) not in (int,float) or not math.isfinite(charge) or not 0<=charge<=record['maximum_gpu_seconds']:
                raise ValueError('Settlement exceeds the declared bounded phase')
            record=dict(record,charged_gpu_seconds=charge,settled_epoch=time.time(),accounting='Upper-bound reservation charged' if allocated_gpu_seconds is None else 'Measured allocated GPU wall seconds')
            del data['active'][operation];data['settled'][operation]=record;data['charged_gpu_seconds']+=charge;atomic(path,data)
            return record

    def reconcile(self,operation,*,allocated_gpu_seconds,evidence,evidence_sha256):
        """Audit a formerly unknown settlement; never erase its original charge.

        The owner must establish the allocation from original process/phase
        evidence. This accounting API verifies the private evidence bytes, not
        their scientific interpretation. Active work cannot be reconciled.
        """
        evidence=Path(evidence)
        if (evidence.is_symlink() or not evidence.is_file()
                or evidence.stat().st_uid!=os.getuid() or evidence.stat().st_mode&0o077
                or sha(evidence)!=evidence_sha256
                or type(allocated_gpu_seconds) not in (int,float)
                or not math.isfinite(allocated_gpu_seconds) or allocated_gpu_seconds<0):
            raise ValueError('Pinned private allocation audit required')
        with self.locked():
            path=self.root/'resource-budget.json';data=json.loads(path.read_text())
            if operation in data['active']:raise ValueError('Active work cannot be reconciled')
            record=data['settled'][operation]
            if record['accounting']!='Upper-bound reservation charged':
                raise ValueError('Only an unknown upper-bound settlement can be audited')
            audit=dict(operation=operation,allocated_gpu_seconds=allocated_gpu_seconds,
                       evidence_path=str(evidence.resolve()),evidence_sha256=evidence_sha256)
            previous=next((x for x in data.get('reconciliations',[]) if x['operation']==operation),None)
            if previous is not None:
                if {k:previous[k] for k in audit}!=audit:raise ValueError('Original reconciliation differs')
                return previous
            if allocated_gpu_seconds>record['maximum_gpu_seconds']:
                raise ValueError('Reconciliation exceeds the original reservation')
            audit.update(original_charged_gpu_seconds=record['charged_gpu_seconds'],
                         adjustment_gpu_seconds=allocated_gpu_seconds-record['charged_gpu_seconds'],
                         reconciled_epoch=time.time())
            data.setdefault('reconciliations',[]).append(audit)
            data['charged_gpu_seconds']+=audit['adjustment_gpu_seconds']
            atomic(path,data);return audit


def copy_checkpoint(destination,candidate):
    """Persist exactly the completed model and moments, with byte verification."""
    destination=Path(destination);destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    private_root(destination);files=[]
    for role,path_key,sha_key in [('model','candidate_path','candidate_sha256'),('optimizer','optimizer_state_path','optimizer_state_sha256')]:
        source=Path(candidate[path_key]);expected=candidate[sha_key]
        if source.is_symlink() or sha(source)!=expected:raise ValueError('Completed source artifact differs')
        target=destination/('candidate.gguf' if role=='model' else 'optimizer-state.bin')
        with source.open('rb') as incoming,target.open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing,8<<20);outgoing.flush();os.fsync(outgoing.fileno())
        if sha(target)!=expected:raise ValueError('Durable checkpoint copy differs')
        target.chmod(0o400);files.append(dict(role=role,path=str(target),sha256=expected,bytes=target.stat().st_size))
    result=dict(schema=1,status='VERIFIED',files=files,optimizer_completed_steps=candidate['optimizer_completed_steps'],model_bytes_transferred_between_hosts=False)
    write(destination/'checkpoint-verification.json',result);return result


def preserve_evidence(point,destination):
    """Retain all original nonbinary evidence as one private tar archive."""
    point=private_root(point);destination=Path(destination);destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    target=destination/'point-evidence.tar.gz';manifest={}
    with tarfile.open(target,'x:gz') as archive:
        for path in sorted(point.rglob('*')):
            if path.is_symlink():raise ValueError('Evidence links are not accepted')
            if path.is_file() and path.suffix not in ('.gguf','.bin'):
                name=str(path.relative_to(point));manifest[name]=sha(path);archive.add(path,arcname=name,recursive=False)
    result=dict(schema=1,status='PRESERVED',archive_sha256=sha(target),files=manifest,binaries_included=False)
    write(destination/'evidence-verification.json',result);return result


def retire_cached_candidate(workspace,candidate,*,successor,evidence_directory,durable_checkpoint=None):
    """Retire only exact completed cache roles after a qualified successor."""
    workspace=private_root(workspace);evidence_directory=private_root(evidence_directory)
    if not 1<=candidate['optimizer_completed_steps']<successor['optimizer_completed_steps']:
        raise ValueError('A qualified later checkpoint is required')
    evidence=json.loads((evidence_directory/'evidence-verification.json').read_text())
    if evidence['status']!='PRESERVED' or sha(evidence_directory/'point-evidence.tar.gz')!=evidence['archive_sha256']:
        raise ValueError('Preserved original evidence required')
    if durable_checkpoint is not None:
        saved=json.loads((Path(durable_checkpoint)/'checkpoint-verification.json').read_text())
        if saved['status']!='VERIFIED' or {f['sha256'] for f in saved['files']}!={candidate['candidate_sha256'],candidate['optimizer_state_sha256']}:
            raise ValueError('Completed durable milestone copy required')
    intent=evidence_directory/'cache-retirement-intent.json'
    completed=evidence_directory/'cache-retirement-complete.json'
    if intent.is_symlink() or completed.is_symlink():raise ValueError('Retirement record links refused')
    reconciling=intent.exists()
    paths=[]
    for key,hash_key in [('candidate_path','candidate_sha256'),('optimizer_state_path','optimizer_state_sha256')]:
        path=Path(candidate[key])
        if path.is_symlink() or not path.resolve().is_relative_to(workspace):
            raise ValueError('Exact owned cached artifact required')
        if path.exists():
            if not path.is_file() or sha(path)!=candidate[hash_key]:raise ValueError('Cached artifact bytes changed')
        elif not reconciling:raise ValueError('Cached artifact missing before original retirement')
        paths.append((path,hash_key))
    expected=dict(schema=1,candidate_sha256=candidate['candidate_sha256'],successor_sha256=successor['candidate_sha256'],files=[dict(path=str(p),sha256=candidate[k]) for p,k in paths])
    if reconciling:
        if json.loads(intent.read_text())!=expected:raise ValueError('Original retirement intent differs')
    else:write(intent,expected)
    if completed.exists():return json.loads(completed.read_text())
    for path,_ in paths:
        if path.exists():path.unlink()
    result=dict(schema=1,status='CACHE_BINARIES_RETIRED',candidate_sha256=candidate['candidate_sha256'],source_scores_response_evidence_preserved=True)
    write(evidence_directory/'cache-retirement-complete.json',result);return result
