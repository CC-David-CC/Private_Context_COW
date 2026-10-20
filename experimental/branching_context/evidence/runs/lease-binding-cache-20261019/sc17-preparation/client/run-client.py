from pathlib import Path
import subprocess
b=Path(__file__).parent
python='/home/dflanag3/spatial-artifacts/branchwrite-dev16-package-20261019-v1/installation/venv/bin/python'
argv=[python,'-B',str(b/'client-driver.py'),'--checkout','/home/dflanag3/src/branchwrite-inline-task-20261019','--fixture',str(b/'workload.json'),'--state',str(b/'state'),'--credentials',str(b/'ssh/credentials.json'),'--ssh-key',str(b/'ssh/id_ed25519'),'--known-hosts',str(b/'ssh/known_hosts'),'--output',str(b/'output')]
with (b/'launch.stdout').open('x') as o,(b/'launch.stderr').open('x') as e:
 p=subprocess.run(argv,stdout=o,stderr=e)
(b/'command.exit').write_text(str(p.returncode))
