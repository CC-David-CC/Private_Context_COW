from pathlib import Path
import subprocess,time,json
u=Path('/home/dflanag3/spatial-artifacts/branchwrite-ubuntu-baseline-20261018-v1');b=Path('/home/dflanag3/spatial-artifacts/branchwrite-ubuntu-baseline-20261018-v1/standalone-sc01-q8-dynamic-v1/client-run');key=Path('/home/dflanag3/spatial-artifacts/branchwrite-ubuntu-baseline-20261018-v1/standalone-sc01-q8-dynamic-v1/ssh')
while not (b/'command.exit').exists():time.sleep(1)
argv=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=bw-llm60','-o','UserKnownHostsFile='+str(key/'known_hosts'),'-i',str(key/'id_ed25519'),'dflanag3@llm-60','stop-owned-campaign']
p=subprocess.run(argv,capture_output=True,text=True,timeout=30);(b/'automatic-stop.json').write_text(json.dumps(dict(exit_code=p.returncode,stderr=p.stderr)))
if p.returncode:raise SystemExit(p.returncode)
for name in ['credentials.json','id_ed25519','id_ed25519.pub']:(key/name).unlink()
(b/'secret-retirement.json').write_text(json.dumps(dict(credentials_removed=True,private_key_removed=True)))
