#!/usr/bin/env python3
"""Qualify the bundled KaTeX and MathJax viewers in the live COW browser UI.

This attaches to a caller-started loopback service and never submits a model
request. It adds one transient equation row to the page only, checks both
typesetter DOMs, toggles math off/on, reloads persisted preferences, and
verifies the namespace's server-owned projection remains byte-for-byte stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from serve.web.cow_native_smoke import DevTools, get_json


class BrowserTestError(RuntimeError):
    pass


class DiagnosticDevTools(DevTools):
    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {
            "expression": expression, "awaitPromise": True, "returnByValue": True, "userGesture": True,
        })
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            exception = details.get("exception", {})
            message = exception.get("description") or details.get("text") or "page expression failed"
            raise BrowserTestError("Browser expression exception: " + str(message)[:500])
        return result.get("result", {}).get("value")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_loopback_http(url: str) -> bool:
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    return parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")


def page_projection(devtools: DevTools, namespace: str) -> dict:
    return devtools.evaluate("(async()=>{const id=" + json.dumps(namespace) + ";"
        "const [b,c]=await Promise.all([ContextWorkspace.nativeRequest('workspace/branches?chat='+encodeURIComponent(id)),"
        "ContextWorkspace.nativeRequest('workspace/chats')]);return {branches:b.branches||[],"
        "chat:(c.chats||[]).find(x=>x.id===id)||null}})()")


def safe_client_errors(devtools: DevTools, key: str) -> list[dict]:
    entries = []
    for event in devtools.events:
        method = event.get("method")
        params = event.get("params", {})
        if method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            value = details.get("exception", {}).get("description", details.get("text", "JavaScript exception"))
            entries.append({"kind": "runtime_exception", "text": str(value)[:1000]})
        elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
            args = params.get("args", [])
            value = " ".join(str(a.get("value", a.get("description", ""))) for a in args)
            entries.append({"kind": "console_error", "text": value[:1000]})
        elif method == "Log.entryAdded" and params.get("entry", {}).get("level") == "error":
            entry = params["entry"]
            message = str(entry.get("text", ""))[:1000]
            status_match = re.search(r"\b(4\d\d|5\d\d)\b", message)
            url = urlsplit(str(entry.get("url", "")))
            status = int(status_match.group(1)) if status_match else None
            entries.append({"kind": "log_error", "text": message,
                            "url_path": url.path if url.scheme in ("http", "https") else None,
                            "http_status": status,
                            "expected_unauthenticated_bootstrap": status == 401,
                            "fixture_optional_route_absent": status == 404 and url.path in ("/metrics", "/mcp")})
    if key:
        for entry in entries:
            entry["text"] = entry["text"].replace(key, "[redacted]")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="loopback URL for an already-running COW service")
    parser.add_argument("--output-dir", required=True, type=Path, help="new private report and screenshot directory")
    parser.add_argument("--browser", default="/snap/bin/chromium")
    parser.add_argument("--tmp-root", default=str(Path.home() / "snap/chromium/common"),
                        help="private temporary browser-profile parent; profile is removed at exit")
    args = parser.parse_args()
    if not is_loopback_http(args.base_url):
        parser.error("only an already-running loopback HTTP COW service is supported")
    output = args.output_dir.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    tmp_root = Path(args.tmp_root).resolve()
    if not tmp_root.is_dir():
        parser.error("Chromium temporary profile root does not exist")
    key = os.environ.get("COW_API_KEY", "")
    profile = tempfile.mkdtemp(prefix="cow-math-engines-", dir=tmp_root)
    process = None
    devtools = None
    checks = []
    stage = "Chromium startup"
    report_path = output / "report.json"
    try:
        process = subprocess.Popen([
            args.browser, "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
            "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
            "--remote-allow-origins=*", "--remote-debugging-port=0", f"--user-data-dir={profile}",
            args.base_url.rstrip("/") + "/",
        ], stdout=(output / "chromium.stdout.log").open("wb"),
           stderr=(output / "chromium.stderr.log").open("wb"))
        port_file = Path(profile) / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None and not port_file.exists():
            time.sleep(0.1)
        if process.poll() is not None or not port_file.exists():
            raise BrowserTestError("Chromium did not start its DevTools endpoint")
        port = int(port_file.read_text(encoding="ascii").splitlines()[0])
        version = get_json(f"http://127.0.0.1:{port}/json/version")
        pages = get_json(f"http://127.0.0.1:{port}/json/list")
        page = next((item for item in pages if item.get("type") == "page"), None)
        if not page or not page.get("webSocketDebuggerUrl"):
            raise BrowserTestError("Chromium page target was not created")
        devtools = DiagnosticDevTools(page["webSocketDebuggerUrl"])
        devtools.call("Page.enable")
        devtools.call("Runtime.enable")
        devtools.call("Log.enable")
        devtools.call("Emulation.setDeviceMetricsOverride", {
            "width": 1365, "height": 900, "deviceScaleFactor": 1, "mobile": False,
        })
        stage = "health and native mode detection"
        service = devtools.evaluate("fetch('/health').then(r=>r.json())")
        if service.get("service") != "context-cow":
            raise BrowserTestError("Service health did not identify Context COW")
        devtools.wait_for("typeof ContextWorkspace!=='undefined'&&ContextWorkspace.nativeMode&&ContextWorkspace.loading===false",
                          "native workspace detection")
        if key:
            stage = "authenticated startup"
            devtools.wait_for("ContextWorkspace.nativeReady===false&&document.getElementById('input').disabled",
                              "unauthenticated startup")
            retained = devtools.evaluate("(()=>{localStorage.setItem('strata.apikey',JSON.stringify(" + json.dumps(key) + "));"
                "document.getElementById('api-key').value=" + json.dumps(key) + ";return store.get('apikey','')!==''})()")
            if not retained:
                raise BrowserTestError("Temporary browser profile did not retain the API key")
            devtools.evaluate("ContextWorkspace.init()")
        devtools.wait_for("ContextWorkspace.nativeReady===true", "authenticated COW workspace")
        stage = "empty namespace creation"
        devtools.evaluate("ContextWorkspace.makeNativeChat('Math Viewer Qualification').then(()=>true)")
        devtools.wait_for("ContextWorkspace.currentChat&&ContextWorkspace.current===null&&document.getElementById('workspace-save-state').textContent.includes('Empty namespace')",
                          "fresh empty namespace")
        namespace = devtools.evaluate("ContextWorkspace.currentChat")
        canonical_before = page_projection(devtools, namespace)
        if canonical_before.get("branches") or not canonical_before.get("chat"):
            raise BrowserTestError("New namespace is not empty in the canonical COW projection")
        before_hash = hashlib.sha256(json.dumps(canonical_before, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        stage = "display-only equation row"
        # A display-only row never enters the composer and never calls an inference endpoint.
        equation = "Equation viewer probe: $x^2$."
        devtools.evaluate("messages.push({role:'assistant',text:" + json.dumps(equation) + ",reasoning:'',time:Date.now()});renderChat()")
        devtools.wait_for("document.querySelector('.st-msg--assistant .st-bubble')", "display-only equation row")

        def select_viewer(enabled: bool, engine: str) -> None:
            devtools.evaluate("document.getElementById('sampling-btn').click();"
                "document.getElementById('s-math').setAttribute('aria-checked'," + json.dumps("true" if enabled else "false") + ");"
                "document.getElementById('s-math-engine').value=" + json.dumps(engine) + ";"
                "document.getElementById('s-apply').click()")
            devtools.wait_for("settings.math===" + json.dumps(enabled) + "&&settings.mathEngine===" + json.dumps(engine),
                              f"{engine} display preference")

        stage = "MathJax render"
        select_viewer(True, "mathjax")
        devtools.wait_for("!!document.querySelector('.chat-math mjx-container')", "actual MathJax container", timeout=30)
        mathjax_state = devtools.evaluate("({row:messages.at(-1)?.text,rendered:!!document.querySelector('.chat-math mjx-container'),"
            "error:!!document.querySelector('.chat-math mjx-merror'),tag:document.querySelector('.chat-math mjx-container')?.tagName})")
        if mathjax_state.get("row") != equation or not mathjax_state.get("rendered") or mathjax_state.get("error"):
            raise BrowserTestError("MathJax did not render the unchanged equation row")
        devtools.screenshot(output / "01-mathjax.png")
        checks.append({"id": "MATHJAX-RENDER", "status": "PASS", "actual_element": mathjax_state["tag"]})

        stage = "KaTeX render"
        select_viewer(True, "katex")
        devtools.wait_for("!!document.querySelector('.chat-math .katex')", "actual KaTeX element", timeout=30)
        katex_state = devtools.evaluate("({row:messages.at(-1)?.text,rendered:!!document.querySelector('.chat-math .katex'),"
            "error:!!document.querySelector('.chat-math .katex .katex-error')})")
        if katex_state.get("row") != equation or not katex_state.get("rendered") or katex_state.get("error"):
            raise BrowserTestError("KaTeX did not render the unchanged equation row")
        devtools.screenshot(output / "02-katex.png")
        checks.append({"id": "KATEX-RENDER", "status": "PASS", "actual_element": "katex"})

        stage = "disable math and preserve source"
        select_viewer(False, "katex")
        devtools.wait_for("!document.querySelector('.chat-math')&&document.querySelector('.st-msg--assistant .st-bubble')?.textContent.includes('$x^2$')",
                          "plain source text after disabling math")
        plain_state = devtools.evaluate("({row:messages.at(-1)?.text,plain:document.querySelector('.st-msg--assistant .st-bubble')?.textContent.includes('$x^2$'),"
            "mathNodes:document.querySelectorAll('.chat-math').length})")
        if plain_state.get("row") != equation or plain_state.get("plain") is not True or plain_state.get("mathNodes") != 0:
            raise BrowserTestError("Disabling math removed or changed the source equation")
        devtools.screenshot(output / "03-plain-source.png")
        checks.append({"id": "MATH-DISABLED-SOURCE", "status": "PASS", "literal_preserved": True})

        stage = "re-enable math"
        select_viewer(True, "katex")
        devtools.wait_for("!!document.querySelector('.chat-math .katex')", "re-enabled KaTeX", timeout=30)
        devtools.screenshot(output / "04-katex-reenabled.png")
        stage = "reload and persisted selection"
        devtools.call("Page.reload", {"ignoreCache": True})
        devtools.wait_for("typeof ContextWorkspace!=='undefined'&&ContextWorkspace.nativeReady===true",
                          "COW workspace after reload")
        devtools.evaluate("ContextWorkspace.openChatNative(" + json.dumps(namespace) + ").then(()=>true)")
        devtools.wait_for("ContextWorkspace.currentChat===" + json.dumps(namespace) + "&&ContextWorkspace.current===null",
                          "same empty namespace after reload")
        persisted = devtools.evaluate("({math:settings.math,engine:settings.mathEngine,saved:JSON.parse(localStorage.getItem('strata.sampling')||'{}'),"
            "namespace:ContextWorkspace.currentChat})")
        if persisted.get("math") is not True or persisted.get("engine") != "katex" or persisted.get("namespace") != namespace:
            raise BrowserTestError("Viewer selection did not persist through reload")
        devtools.evaluate("messages.push({role:'assistant',text:" + json.dumps(equation) + ",reasoning:'',time:Date.now()});renderChat()")
        devtools.wait_for("!!document.querySelector('.chat-math .katex')", "persisted KaTeX render after reload", timeout=30)
        stage = "canonical projection after reload"
        canonical_after = page_projection(devtools, namespace)
        after_hash = hashlib.sha256(json.dumps(canonical_after, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if canonical_after != canonical_before or after_hash != before_hash:
            raise BrowserTestError("Display-only equation changed the canonical COW namespace projection")
        devtools.screenshot(output / "05-persisted-katex-after-reload.png")
        checks.append({"id": "MATH-PERSIST-RELOAD", "status": "PASS", "same_namespace": True,
                       "canonical_projection_unchanged": True})

        stage = "report and capture client errors"
        errors = safe_client_errors(devtools, key)
        source_root = Path(__file__).resolve().parents[2]
        source_files = [
            "serve/web/cow_math_engines_browser_test.py",
            "serve/web/app.js", "serve/web/math.js", "serve/web/math-adapter.js",
            "serve/web/math-katex.js", "serve/web/math-mathjax.js",
            "serve/web/vendor/katex/dist/katex.min.js", "serve/web/vendor/mathjax/tex-chtml.js",
        ]
        source_hashes = {name: sha256(source_root / name) for name in source_files}
        result = {
            "schema": 1, "status": "PASS", "inference_kind": "none",
            "scope": "live same-origin COW UI; display-only equation row; no generation requests",
            "browser": version.get("Browser"), "browser_gpu_disabled": True,
            "service": service.get("service"), "namespace_id": namespace,
            "canonical_projection_sha256_before": before_hash,
            "canonical_projection_sha256_after": after_hash,
            "display_equation_source_sha256": hashlib.sha256(equation.encode()).hexdigest(),
            "source_sha256": source_hashes,
            "checks": checks,
            "client_errors": errors,
            "client_error_count": len(errors),
            "client_errors_unexpected_for_fixture_scope": sum(
                not entry.get("expected_unauthenticated_bootstrap") and not entry.get("fixture_optional_route_absent")
                for entry in errors),
            "api_key_in_report": False,
            "screenshots": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                            for p in sorted(output.glob("*.png"))},
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "complete", "status": "PASS", "report": str(report_path)}))
        return 0
    except Exception as error:
        errors = safe_client_errors(devtools, key) if devtools else []
        error_text = str(error).replace(key, "[redacted]") if key else str(error)
        report_path.write_text(json.dumps({"schema": 1, "status": "FAIL", "stage": stage, "error": error_text,
            "inference_kind": "none", "browser_gpu_disabled": True, "client_errors": errors,
            "client_error_count": len(errors), "api_key_in_report": False}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "complete", "status": "FAIL", "report": str(report_path), "error": str(error)}))
        return 1
    finally:
        if devtools:
            try:
                devtools.call("Browser.close")
            except Exception:
                pass
            devtools.close()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
