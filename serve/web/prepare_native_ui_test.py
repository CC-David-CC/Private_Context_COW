#!/usr/bin/env python3
"""Create a browser-test copy that injects the mock API before app startup."""
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
html = source.read_text(encoding="utf-8")
tag = '<script src="web/app.js"></script>'
injected = '<script src="web/test_cow_native_ui.js"></script>' + tag
if html.count(tag) != 1:
    raise SystemExit("expected exactly one app.js script tag")
destination.write_text(html.replace(tag, injected), encoding="utf-8")
