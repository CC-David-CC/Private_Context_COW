"""Private named GGUF preservation from completed registered native jobs."""
from pathlib import Path
import hashlib,json,os,re
from tools.gguf_reader import GGUFFile
from tools.pcc_native_training import sha256

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def preserve(store,model_contract_id,candidate,*,job_id,name,expected_sha256,notes,report_sha256,write):
    if type(name) is not str or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',name) or name.endswith('.gguf'):raise ValueError('Use a bounded model label without a path or .gguf suffix')
    if type(notes) is not str or len(notes.encode())>16384 or (report_sha256 is not None and not re.fullmatch('[a-f0-9]{64}',str(report_sha256))):raise ValueError('Bounded owner notes and optional report SHA-256 required')
    if candidate.get('mode')!='full' or expected_sha256!=candidate.get('candidate_sha256'):raise ValueError('Expected complete standalone model identity required')
    store=Path(store).resolve();source=Path(candidate['candidate_path'])
    if source.is_symlink() or (store/'runs') not in source.resolve(strict=True).parents or sha256(source)!=expected_sha256:raise ValueError('Registered candidate bytes/path differ')
    root=store/'exports';root.mkdir(mode=0o700,exist_ok=True)
    if root.is_symlink():raise ValueError('Export store link rejected')
    info=GGUFFile(source)
    value=dict(schema=1,kind='pcc.private-model-export.v1',name=name,file=name+'.gguf',job_id=job_id,model_contract_id=model_contract_id,model_sha256=expected_sha256,bytes=source.stat().st_size,tensor_types=info.by_type(),owner_notes=notes,owner_report_sha256=report_sha256,report_authority='Owner-provided report reference; not independent qualification',weights_published=False,promoted=False)
    value['export_id']=identity(value);target=root/value['file'];manifest=root/(name+'.json')
    if manifest.exists():
        if manifest.is_symlink() or json.loads(manifest.read_text())!=value:raise ValueError('Export name already binds different provenance or notes')
    if target.exists():
        if target.is_symlink() or sha256(target)!=expected_sha256:raise ValueError('Existing export bytes differ')
    else:
        try:os.link(source,target)
        except OSError as error:raise ValueError('Named preservation requires a same-filesystem private store') from error
    os.chmod(target,0o400)
    if not manifest.exists():write(manifest,value)
    return value


def read_export(path,model_contract_id):
    path=Path(path)
    if path.is_symlink():raise ValueError('Export manifest link rejected')
    value=json.loads(path.read_text())
    if type(value) is not dict or value.get('export_id')!=identity({k:v for k,v in value.items() if k!='export_id'}) or value.get('model_contract_id')!=model_contract_id or value.get('name')!=path.stem or value.get('file')!=path.stem+'.gguf':raise ValueError('Saved export identity/provenance differs')
    target=path.with_suffix('.gguf')
    if target.is_symlink() or not target.is_file() or target.stat().st_size!=value.get('bytes'):raise ValueError('Saved export is missing or has different storage size')
    return value
