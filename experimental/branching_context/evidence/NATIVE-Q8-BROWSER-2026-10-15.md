# Assembled real-Q8 browser smoke: bounded PASS

Frozen source: `ac6b8e6d465d866a0ae341d9d37a28400aea582f`.
Host: llm-60, GPU0, RTX PRO 6000 Blackwell. Native capacity98,304,
prefill chunk8,192, Q8 weights and FP16 KV use the already pinned execution
profile `b62b4e1842ae4c9b5a0b9c6ea372cb9dcaa2333206adf4b2dd9b7a747f7323d1`.
Executed profile SHA-256:
`14b3d16b9ba65e84e42c16352ffcd88304e4ff3ea2302a235d75ef1ecb164e59`.

## Executed checks

Chromium155, with browser GPU acceleration disabled, drove the actual COW
HTTP server and real native model. No fetch mock or synthetic inference.
Eight grouped checks passed: SMOKE-HEALTH, SMOKE-AUTH,
SMOKE-EMPTY-NAMESPACE, SMOKE-PREPARE-ANSWER, SMOKE-FORK-ISOLATION,
SMOKE-GRAPH, SMOKE-MATH and SMOKE-TRACE. No failures or skips were reported.
The corrected v2 launcher recorded command.exit0 and wrapper.exit0.

The attached real-source text was44,027 UTF-8 bytes,8,806 source tokens,
SHA `e744d8bff6bebbdc3d0fc0d0ad445083f4c0eb74c6c2e93424e69dcdf1cfd0b4`.
Both answers naturally stopped after two output tokens, with no resume rounds.
The UI message model matched canonical history and the rendered Markdown
projection matched its normalized text. Fork answering left the source tip
unchanged. Graph selection retained drafts; unavailable snapshot boundaries
had no actions; rewind was published and old tips remained visible without
counting as current history. Root inspected the shared-branch and previous-tip
screenshots and checked their hashes against the report.

Math used an explicitly transient display fixture and actual KaTeX typesetting;
disabling rendering preserved literal source. This is not evidence of generated
LaTeX correctness. MathJax has separate display-only evidence.

## Resource and provenance observations

The pre-shutdown coordinator reported max_active_per_gpu0=1, completed6,
failed0, queued0 and active0 across the two browser runs. Accounted RAM peak
was9,186,322,784 bytes, with96GiB RAM and128GiB disk budgets. Disk files used
1,050,826,714 bytes; no pressure eviction occurred. These are pool accounting,
not whole-process memory or deduplicated GPU memory. After rewind, both
selected tips had8,842 canonical token IDs and stopped controllers with two
accepted output tokens. Do not substitute the earlier namespace HTTP fixture's
prefix length for this UI request's tokenization.

Preflight initially compared same-named runtime files against historical build
inputs. Complete-path verification resolved the apparent mismatch: qualified
runtime16/16, historical API inputs13/13 at their original paths, native source
and executable pins16/16. Actual imports came from the frozen ac6 candidate;
all4,764 manifest files matched. No profile mutation or old module overlay.

V1's report passed8 checks, but its shell wrapper wrote a literal `$result`.
Its child exit remains unknown. The defect and attempt are retained. V2 used
the verified run-command helper via a transferred LF script and a new namespace
against the existing model process. No duplicate live inference was launched.

After final collection, server1855813, native1855826 and driver1857899/1860162
were verified absent. The ordinary chat server remained off.

## Evidence and limitations

Raw root:
`/home/dflanag3/spatial-artifacts/namespace-q8-browser-ac6-20261014/`.
Final report: `browser-smoke-v2/output/report.json`.
Numeric exit records: `browser-smoke-v2/run/command.exit` and `wrapper.exit`.
Profile/source/runtime receipt: `browser-smoke-v2/evidence.json`.

The adjacent JSON report and PROVENANCE JSON are byte-identical copies; their
SHA-256 values are respectively
`bb4e026c64dbc7e8084473b63fcebedbca9e5b8a76e2c9effbfc33696f243165` and
`bfd2a743422c740a0209d3332d808cb7c464781ada8d6dcc9cf49cc4a85b5973`.

Raw evidence and frozen source were streamed directly from llm-60 to private
box-git; no local archive spool was created. Final archive:
`/srv/data/git/private_context_cow.evidence/20261015-namespace-q8-browser-ac6/browser-v2-terminal-raw.tar`,
1,551,093,760 bytes, SHA
`926808ae1d114fcb1bb5c7212501023e4abd8a88c734715ea4b4345662433bce`.
Stream and destination hashes match; tar listing passed5,442 entries. Root
independently reread the archive hash and checked the manifest, numeric v2
receipts, malformed v1 receipt and packager entry. Directory0700/archive0600.
The BACKUP JSON records scope, excluded external model/pack/binaries and the
separately labelled incomplete first archive. Frozen source includes its tracked
optional control-vector asset; that asset was not enabled in this run.

No independent logits were compared in this browser run: numerical max error
is unmeasured, not zero. It is not a speed benchmark, long-output qualification,
GPU cancellation/restart test, iterator-resume test or original G1-G5 pass.
Those capabilities have separately scoped evidence. The active goal remains
native-speed Context COW for one active request, including useful multi-context
pool operation and remaining qualification/packaging work.
