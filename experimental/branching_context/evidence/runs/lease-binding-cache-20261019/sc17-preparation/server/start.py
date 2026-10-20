import subprocess,pathlib,os
d=pathlib.Path(__file__).parent
with (d/'steering-launch.stdout').open('x') as o,(d/'steering-launch.stderr').open('x') as e:
 p=subprocess.Popen(['/home/dflanag3/src/Strata/.venv/bin/python','-B',str(d/'launch.py')],stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True,close_fds=True)
(d/'launch.pid').write_text(str(p.pid))
print(p.pid)
