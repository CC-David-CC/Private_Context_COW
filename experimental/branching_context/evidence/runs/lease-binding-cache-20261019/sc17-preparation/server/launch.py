from pathlib import Path
import subprocess
d=Path(__file__).parent
with (d/'launch.stdout').open('xb') as o,(d/'launch.stderr').open('xb') as e:p=subprocess.run(['flock','-n','/home/dflanag3/branching-g3/native-client-speed.gpu0.lock','/home/dflanag3/src/Strata/.venv/bin/python','-B',str(d/'run-server.py')],stdout=o,stderr=e)
(d/'command.exit').write_text(str(p.returncode))
raise SystemExit(p.returncode)
