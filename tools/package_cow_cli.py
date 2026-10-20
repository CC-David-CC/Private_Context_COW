"""Package the private terminal client without model, engine or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, unquote
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error('output already exists; choose a new artifact path')
    paths = ['tools/cow_cli.py', 'docs/context-cow/CLI.md',
             'docs/context-cow/CLI-CLIENT-README.md',
             'docs/context-cow/CONTEXT-COW-GUIDE.md']
    # Include the compact evidence linked directly by the supplied guides.
    # Never recursively collect an entire repository or an external run tree.
    linked = []
    for name in paths:
        if name.endswith('.md'):
            for link in re.findall(r'\]\(([^)]+)\)',(root/name).read_text(encoding='utf-8')):
                if link.startswith('https://'):
                    continue
                target = ((root/name).parent/link.split('#',1)[0]).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    raise ValueError('Missing or external guide dependency: '+link)
                linked.append(target.relative_to(root).as_posix())
    figure = 'experimental/branching_context/evidence/figures/2026-10-15-browser-durable-latency/'
    paths = list(dict.fromkeys(paths+linked+[figure+'latency.png',figure+'measurements.json']))
    payloads = {name:(root/name).read_bytes() for name in paths}
    readme = payloads['docs/context-cow/CLI-CLIENT-README.md'].decode('utf-8')
    def relocate(match):
        link = match.group(1)
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        target = ((root/'docs/context-cow')/unquote(parsed.path)).resolve()
        rebased = target.relative_to(root).as_posix()
        if parsed.fragment:
            rebased += '#'+parsed.fragment
        return ']('+rebased+')'
    readme = re.sub(r'\]\(([^)]+)\)',relocate,readme)
    payloads['README.md'] = readme.encode('utf-8')
    # Evidence indexes belong in this small client package, but their complete
    # historical source tree and raw runs do not. Keep those references explicit
    # as source-checkout text instead of shipping broken clickable links.
    transforms = {'README.md':'Rebased local links after relocating the client guide, preserving fragments'}
    for name,data in tuple(payloads.items()):
        if not name.endswith('.md'):
            continue
        omitted = []
        def package_link(match):
            label,link = match.groups()
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                return match.group(0)
            target = ((root/name).parent/unquote(parsed.path)).resolve()
            if target.is_relative_to(root) and target.relative_to(root).as_posix() in payloads:
                return match.group(0)
            omitted.append(link)
            return label+' (full source checkout: `'+link+'`)'
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',package_link,data.decode('utf-8'))
        if omitted:
            note = ('> Client package note: references marked "full source checkout" '
                    'are outside this client-only ZIP.\n\n')
            payloads[name] = (note+text).encode('utf-8')
            transforms[name] = dict(source_only_references=omitted)
    manifest = dict(schema='cow-cli-client-package-v1',
        scope='Terminal client only; existing COW service required; no GPU engine or model included',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),
        packager_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        documentation_transforms=transforms,
        files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
               for name,data in payloads.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation avoids replacing a completed or still-live package.
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in payloads.items():
            archive.writestr('context-cow-cli/'+name,data)
        archive.writestr('context-cow-cli/manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name,entry in manifest['files'].items():
            raw = archive.read('context-cow-cli/'+name)
            if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise RuntimeError('Archived source differs: '+name)
    print(json.dumps(dict(output=str(args.output.resolve()),
        bytes=args.output.stat().st_size,sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        source_commit=manifest['source_commit'],files=len(payloads)),indent=2))


if __name__ == '__main__':
    main()
