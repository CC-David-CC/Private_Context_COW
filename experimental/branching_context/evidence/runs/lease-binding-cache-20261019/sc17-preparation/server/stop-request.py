import os
from pathlib import Path
if os.environ.get('SSH_ORIGINAL_COMMAND')!='stop-owned-campaign':raise SystemExit(1)
import json,urllib.request
d=Path('/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/vc-rerun-20261018-v1/standalone-SC17-q8-binding-cache-dev16-v1')
try:
 credentials=json.loads((d/'credentials.json').read_text())
 request=urllib.request.Request('http://127.0.0.1:'+str(credentials['port'])+'/cow/status',headers={'Authorization':'Bearer '+credentials['api_key']})
 with urllib.request.urlopen(request,timeout=10) as response:value=json.load(response)
 (d/'final-status.json').write_text(json.dumps(value,indent=2)+'\n')
except Exception as exc:
 (d/'final-status-error.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)))+'\n')
finally:
 (d/'STOP').touch(exist_ok=True)
