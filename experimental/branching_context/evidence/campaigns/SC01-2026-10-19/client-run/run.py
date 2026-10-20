from pathlib import Path
import subprocess,json
b=Path(__file__).parent
with (b/'stdout').open('xb') as o,(b/'stderr').open('xb') as e:p=subprocess.run(json.loads((b/'argv.json').read_bytes()),stdout=o,stderr=e)
(b/'command.exit').write_text(str(p.returncode))
raise SystemExit(p.returncode)
