"""Owner-only retirement of verified experimental curve binaries.

Scores, response evidence and source records remain. Clients cannot submit
arbitrary paths through the training API. This helper is for an owned archive.
"""
from pathlib import Path
import hashlib,json,os,re

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write_once(path,value):
    path=Path(path)
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text())!=value:raise ValueError('Existing retention record binds different bytes')
        return
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as out:json.dump(value,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def retire_point(archive,arm,update,policy,*,protected_updates=(),successor_update=None):
    archive=Path(archive)
    if archive.is_symlink() or not archive.is_dir() or archive.stat().st_uid!=os.getuid() or archive.stat().st_mode&0o077:raise ValueError('Private owned archive root required')
    archive=archive.resolve()
    if type(arm) is not str or re.fullmatch('[a-z][a-z0-9-]{0,95}',arm) is None or type(update) is not int or not 1<=update<=68:raise ValueError('Bounded curve arm/update required')
    point=archive/arm/('update-'+str(update).zfill(3))
    if point.is_symlink() or not point.resolve().is_relative_to(archive):raise ValueError('Point path escapes the owned archive')
    final=policy.get('report_complete') is True
    if not final and (update in policy['milestone_updates'] or update in protected_updates):
        return dict(status='PROTECTED',arm=arm,update=update)
    def checked_record(directory):
        if directory.is_symlink() or not directory.resolve().is_relative_to(archive):raise ValueError('Archive receipt directory escapes this task')
        path=directory/'archive-verification.json'
        if path.is_symlink():raise ValueError('Archive receipt link rejected')
        value=json.loads(path.read_text())
        if value.get('status')!='VERIFIED' or value.get('directory')!=str(directory):raise ValueError('Verified original archive receipt required')
        small=next((f for f in value['files'] if f['path']=='curve-point.json'),None)
        if small is None or sha(directory/'curve-point.json')!=small['sha256']:raise ValueError('Immutable measured point evidence required')
        measured=json.loads((directory/'curve-point.json').read_text())
        if measured.get('arm')!=arm or measured.get('update')!=int(directory.name.split('-')[-1]):raise ValueError('Curve point identity differs')
        return value,measured
    record,measured=checked_record(point)
    if final:
        if re.fullmatch('[a-f0-9]{64}',str(policy.get('report_sha256',''))) is None or re.fullmatch('[a-f0-9]{40}',str(policy.get('report_commit',''))) is None:
            raise ValueError('After-report retirement requires the completed report identity and publication commit')
        successor=None
    else:
        if type(successor_update) is not int or not update<successor_update<=68:raise ValueError('Qualified later checkpoint required before retirement')
        directory=archive/arm/('update-'+str(successor_update).zfill(3))
        _,successor=checked_record(directory)
        if successor['protocol_id']!=measured['protocol_id']:raise ValueError('Successor belongs to another experiment')
    protected=set(policy.get('selected_current_model_sha256s',[])) if final else set()
    roles={'training/candidate.gguf','training/optimizer-state.bin','merged-F32.gguf'}
    selected=[f for f in record['files'] if f['path'] in roles and not (f['path'].endswith('.gguf') and f['sha256'] in protected)]
    primary=next((f for f in record['files'] if f['sha256']==measured['model_sha256'] and f['path'].endswith('.gguf')),None)
    named=Path(record['named_model'])
    if named.parent!=archive/'named' or named.is_symlink():raise ValueError('Named model link escapes this archive')
    if primary is not None and primary in selected:
        selected.append(dict(path='@named/'+named.name,bytes=primary['bytes'],sha256=primary['sha256']))
    phase='after-report' if final else 'during-study'
    prior=point/'retention-during-study-complete.json'
    if final and prior.exists():
        if prior.is_symlink():raise ValueError('Retirement receipt link rejected')
        historical=json.loads(prior.read_text())
        if historical.get('original_archive_id')!=digest(record):raise ValueError('Prior retirement belongs to another archive')
        removed={(f['path'],f['bytes'],f['sha256']) for f in historical['removed']}
        selected=[f for f in selected if (f['path'],f['bytes'],f['sha256']) not in removed]
    intent=dict(schema=1,kind='pcc.curve-retention.v1',phase=phase,arm=arm,update=update,
        policy_id=digest(policy),original_archive_id=digest(record),removed=selected,
        successor_model_sha256=None if successor is None else successor['model_sha256'],
        evidence_preserved=True,weights_published=False)
    target=point/('retention-'+phase+'-intent.json');completed=point/('retention-'+phase+'-complete.json')
    if target.is_symlink() or completed.is_symlink():raise ValueError('Retention record link rejected')
    if completed.exists():
        saved=json.loads(completed.read_text())
        if saved.get('original_archive_id')!=digest(record):raise ValueError('Retirement receipt belongs to another archive')
        return saved
    def locate(item):
        path=archive/'named'/item['path'][7:] if item['path'].startswith('@named/') else point/item['path']
        if path.is_symlink() or not path.resolve().is_relative_to(archive):raise ValueError('Artifact link/path rejected')
        return path
    if not target.exists():
        checked={};observations={}
        for item in selected:
            path=locate(item)
            if not path.is_file():raise ValueError('Archive binary missing; no retirement authorized')
            info=path.stat();key=(info.st_dev,info.st_ino)
            actual=checked.get(key)
            if actual is None:actual=sha(path);checked[key]=actual
            if info.st_size!=item['bytes'] or actual!=item['sha256']:raise ValueError('Archive binary bytes differ; no retirement authorized')
            observations[item['path']]=dict(device=info.st_dev,inode=info.st_ino,bytes=info.st_size,mtime_ns=info.st_mtime_ns)
        intent['verified_file_observations']=observations
        # Durable intent precedes unlink, so a crash can reconcile the same files.
        write_once(target,intent)
    else:
        saved=json.loads(target.read_text());observations=saved.get('verified_file_observations')
        if {k:v for k,v in saved.items() if k!='verified_file_observations'}!=intent or type(observations) is not dict:raise ValueError('Original retirement intent differs')
        intent=saved
    for item in selected:
        path=locate(item)
        if path.exists():
            info=path.stat();observed=dict(device=info.st_dev,inode=info.st_ino,bytes=info.st_size,mtime_ns=info.st_mtime_ns)
            if observed!=intent['verified_file_observations'][item['path']] and (info.st_size!=item['bytes'] or sha(path)!=item['sha256']):raise ValueError('Binary changed after retention intent; preserve it')
            path.unlink()
    for directory in {locate(item).parent for item in selected}:
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    result=dict(intent,status='BINARIES_RETIRED_EVIDENCE_RETAINED')
    write_once(completed,result)
    return result
