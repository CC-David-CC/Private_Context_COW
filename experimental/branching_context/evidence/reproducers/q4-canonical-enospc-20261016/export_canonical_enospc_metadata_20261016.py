"""Export byte-checked metadata from an already read-only private mount."""
import hashlib
import json
import os
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve(strict=True)
destination = Path(sys.argv[2]).absolute()
receipt = Path(sys.argv[3]).absolute()
assert os.geteuid() == 0 and source.is_mount()
assert os.statvfs(source).f_flag & os.ST_RDONLY
assert not destination.exists() and not receipt.exists()
destination.mkdir(mode=0o700)
files = {}
for path in sorted(source.rglob('*')):
    relative = path.relative_to(source)
    if relative.parts[0] in ('lost+found', 'checkpoints'):
        continue
    assert not path.is_symlink()
    target = destination / relative
    if path.is_dir():
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        continue
    assert path.is_file()
    raw = path.read_bytes()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    assert target.read_bytes() == raw == path.read_bytes()
    target.chmod(0o600)
    files[relative.as_posix()] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
for path in [destination, *destination.rglob('*')]:
    os.chown(path, 1000, 1000)
record = dict(schema='canonical-readonly-export-v1', source=str(source),
              destination=str(destination), readonly_source=True, files=files,
              file_count=len(files), total_bytes=sum(row['bytes'] for row in files.values()))
with receipt.open('x', encoding='utf-8') as stream:
    json.dump(record, stream, sort_keys=True, indent=2)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
receipt.chmod(0o600)
os.chown(receipt, 1000, 1000)
print(json.dumps(dict(exported_files=len(files), exported_bytes=record['total_bytes'])))
