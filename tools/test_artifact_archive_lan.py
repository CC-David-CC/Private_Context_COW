"""Explicit owner LAN fixture: actual SSH copy/readback and refusal behavior."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess
from tools.pcc_artifact_archive import copy_artifact


def run(root,host,mount,mount_uuid,directory):
    root=Path(root);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    data=bytes(range(256))*4096;source=root/'owned-fixture.bin';source.write_bytes(data)
    expected=hashlib.sha256(data).hexdigest();target=directory+'/owned-fixture.bin'
    common=dict(host=host,mount=mount,mount_uuid=mount_uuid,destination=target)
    value=copy_artifact(source,expected,receipt=root/'first.json',**common)
    assert value['status']=='VERIFIED_ARCHIVE' and not value['reused'] and source.read_bytes()==data
    received=subprocess.check_output(['ssh','-o','BatchMode=yes',host,'cat -- '+shlex.quote(target)])
    assert received==data
    repeated=copy_artifact(source,expected,receipt=root/'repeated.json',**common)
    assert repeated['reused'] and source.read_bytes()==data
    refused=0
    for label,changes in [('wrong-source',dict(expected_sha256='0'*64)),
                          ('wrong-drive',dict(mount_uuid='wrong-drive',destination=directory+'/wrong-drive.bin')),
                          ('existing-local-receipt',dict(receipt=root/'first.json'))]:
        kwargs=dict(common,receipt=root/(label+'.json'));kwargs.update(changes)
        digest=kwargs.pop('expected_sha256',expected)
        try:copy_artifact(source,digest,**kwargs)
        except (ValueError,RuntimeError,FileExistsError):refused+=1
        else:raise AssertionError(label+' admitted')
    link=root/'source-link';link.symlink_to(source)
    try:copy_artifact(link,expected,receipt=root/'linked.json',**common)
    except ValueError:refused+=1
    else:raise AssertionError('Source symlink admitted')
    pending=directory+'/interrupted.bin.partial'
    code='from pathlib import Path;p=Path('+repr(pending)+');f=p.open("xb");f.write(b"preserved interrupted bytes");f.close()'
    subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(code)],check=True)
    try:copy_artifact(source,expected,receipt=root/'interrupted.json',**dict(common,destination=directory+'/interrupted.bin'))
    except RuntimeError:refused+=1
    else:raise AssertionError('Incomplete original copy overwritten')
    preserved=subprocess.check_output(['ssh','-o','BatchMode=yes',host,'cat -- '+shlex.quote(pending)])
    assert preserved==b'preserved interrupted bytes'
    assert source.read_bytes()==data and refused==5
    result=dict(status='PASS',scope='Actual owner SSH archival fixture; no model updates or source deletion',
                bytes=len(data),copy_readback_exact=True,idempotent_copy_reused=True,
                refusal_cases=refused,source_retained=True)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','host','mount','mount-uuid','directory'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.host,a.mount,a.mount_uuid,a.directory)))
