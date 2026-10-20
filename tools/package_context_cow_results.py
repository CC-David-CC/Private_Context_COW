"""Export measurement reports, preserving failed runs and recording redactions."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--redact-root', action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, policy='Only supplied personal path roots are replaced. Metrics, failures, '
                    'tokens and fingerprints are retained. Redacted profiles are evidence, not executable profiles.',
                    files=[])
    for directory in args.input:
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.suffix not in ('.json', '.log', '.stdout', '.stderr', '.txt'):
                continue
            original = path.read_bytes()
            value = original.decode('utf-8')
            for root in args.redact_root:
                value = value.replace(root, '<measurement-home>/')
                value = value.replace(root.replace('\\', '\\\\'), '<measurement-home>/')
            output = value.encode('utf-8')
            relative = Path(directory.name) / path.relative_to(directory)
            if path.suffix == '.log':
                relative = relative.with_suffix('.log.txt')
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(output)
            manifest['files'].append(dict(path=relative.as_posix(), original_bytes=len(original),
                                         original_sha256=hashlib.sha256(original).hexdigest(),
                                         exported_sha256=hashlib.sha256(output).hexdigest(),
                                         path_redacted=original != output))
    (args.output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(files=len(manifest['files']), output=str(args.output))))


if __name__ == '__main__':
    main()
