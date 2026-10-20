#!/usr/bin/env python3
"""Create a same-origin browser fixture page with real fetch and synthetic history."""
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
api_key = sys.argv[3] if len(sys.argv) > 3 else ""
html = source.read_text(encoding="utf-8")
html = html.replace('src="web/', 'src="/web/').replace('href="web/', 'href="/web/')
html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8"><base href="/">', 1)
first_script = '<script src="/web/math.js"></script>'
if html.count(first_script) != 1 or html.count('<script src="/web/app.js"></script>') != 1:
    raise SystemExit("expected the standard same-origin Strata script tags")
legacy = json.dumps([{"role": "user", "text": "must not import this legacy message"}])
startup = ('<script>localStorage.setItem("strata.apikey",'
           + json.dumps(json.dumps(api_key))
           + ');localStorage.setItem("strata.chat",'
           + json.dumps(legacy) + ');</script>')
html = html.replace(first_script, startup + first_script)
html = html.replace('<script src="/web/app.js"></script>',
                    '<script src="/web/app.js"></script><script src="/web/test_cow_native_socket.js"></script>')
destination.write_text(html, encoding="utf-8")
