# Opt-in native COW token-history serving

Implemented `serve/cow_adapter.py` and standalone `serve/cow_http.py`. The
ordinary chat server/default engine path is unchanged and remains off. This
transport consumes caller-rendered token IDs, not OpenAI chat messages. Real
chat template rendering, text decoding, UI/SQLite projection and ordinary chat
endpoint integration remain outstanding; this is not a completed chat release.

The COW Store is the sole canonical history and branch authority. Before
admission, the adapter checks exact token-prefix equality and the requested
revision. A retry resolves its original source through a checksum-validated
existing coordinator intent; submission still validates the complete request
signature and existing CAS/recovery path. No second history representation or
mutable native access is introduced on HTTP handler threads.

Only text token IDs, greedy temperature 0 and an explicit positive output bound
are supported. Unknown fields/options are rejected. Authentication applies to
reads and writes; foreign browser Origins are rejected. Public bind requires a
key. Streamed NDJSON token/snapshot events are provisional: only `complete`
contains the acknowledged durable receipt. Failed publication emits `error`,
never `complete`. Disconnect signals transient consumed-boundary cancellation.

Executed test-first on Linux:

- HTTP RED: absent `serve.cow_http` module.
- Final combined supplemental suite: 26 tests (6 prefix, 11 adapter, 9 HTTP),
  no failures/errors/skips. Real HTTP sockets; synthetic computation and a
  real canonical Store in adapter fixtures. Not GPU/model qualification.
- Existing coordinator/pool control regressions: 106 passed, no skips.
- G0: 51 passed, result `20261014T222154207438Z-G0.json` on serving-v1.
- Reference mutation smoke: all four deliberately incorrect outcomes rejected.

Machine-readable IDs and source/dependency hashes:
`SERVING-CONTROL-2026-10-14.json`, `SERVING-POOL-CONTROL-2026-10-14.json`.
Tests cover altered inputs on retry, corrupted intent checksum, wrong source
branch/revision, uncommitted intents, tokenization boundary changes, auth,
unknown fields, revision checks, disconnect and failed durable publication.

## Linux developer launch

Use a separately pinned serving profile and a compatible store. The initial
launcher explicitly requires GPU=[0], with full-size defaults (96 GiB shared
RAM, 128 GiB disk). Do not silently relabel older checkpoints to a new profile.

```sh
export PYTHONPATH="$PWD/experimental/branching_context/src:$PWD"
python3 -m serve.cow_http --profile /absolute/path/serving-profile.json \
  --store /absolute/path/context-store --host 127.0.0.1 --port 8099
```

API operations:

- POST `/cow/create`: `branch`, initial `token_ids`, `operation_id`.
- GET `/cow/head?branch=A`: canonical tip/root/revision.
- POST `/cow/fork`: new `name`, durable `snapshot`.
- POST `/cow/rewind`: `branch`, `revision`, `snapshot`, `operation_id`.
- POST `/cow/generate`: `branch`, `revision`, `operation_id`,
  `rendered_token_ids` (the entire current prompt), `max_tokens`, optional
  `settings:{"temperature":0}` and `stream` (default true).

All operations return canonical COW data, never independent SQLite branch
moves. A nonstream generation response also waits for durable publication.
The next qualification is a real-Q8 HTTP run under a profile bound to the new
control code, then prompt rendering/ordinary chat integration. Existing GPU
speed/cancellation evidence is separate and does not qualify this transport.

## Live qualification launch

`scripts/gpu_http_probe.py` launched on llm-60 in tmux `cow-http-q8-v1`.
Frozen candidate: `request-stop-20261014/serving-v1`; output: `http-q8-v1`.
Its new profile fingerprints the candidate's actual Python control/transport
files along with the pinned native build. It computes a fresh 65,536-token
prefix rather than rewriting any old checkpoint identities. Through actual
loopback HTTP it exercises two A/B suffixes (~8K), up to 128 generated outputs,
durable retries, stream timing and forward-work counters. After closing the
server/native worker, it runs two independent fresh native references under the
same new profile. Final tokens and full frontiers must match exactly.

Campaign is RUNNING; no real-HTTP GPU PASS is claimed. One GPU lock serializes
the campaign. 96K capacity, 96 GiB RAM, 128 GiB disk; no chat server restart.
Result and 30-second heartbeats are incremental. The HTTP listener is ephemeral
loopback with a test-only key and is closed by the campaign, not a replacement
production chat server. This campaign does not qualify ordinary chat templates.

## Terminal HTTP result: bounded real Q8 PASS

`http-q8-v1` finished PASS, native exit 0; both independent fresh references
matched all 128 tokens and the full final frontier for each branch, zero error.
Saved-prefix work remained zero, parent payload stayed unchanged and durable
retries returned the existing answer/receipt. Actual received generation rates
were 58.42 TPS (A) and 55.61 TPS (B); TTFT 5.92 s and 3.31 s; median TPOT
16.96 ms and 17.49 ms. Whole durable requests were 13.90 s and 11.54 s. These
are single A/B measurements, not repeated-run medians. Setup/model loading and
the paid initial prefix computation are separate. Full manifest/profile/build/
model fingerprints and timing/work counters: `SERVING-HTTP-Q8-2026-10-14.json`.
Raw result SHA binds the compact report; unabridged result remains on llm-60 and
locally under evidence/runs. Controller/support lists are summarized with a
full-record digest; they were not independently controller-qualified here.

Actual HTTP disconnect campaign `http-drop-q8-v1` also passed two checks:
disconnect after three received token events drained partial native generation,
without canonical snapshot/branch publication; the following HTTP request
produced 32 matching tokens and an identical final frontier, no prefix work or
parent mutation. Native exit 0 and GPU idle. Counters/actual consumed count are
in `SERVING-HTTP-CANCEL-Q8-2026-10-14.json`. Received and consumed counts need not
coincide at network disconnect. Not general cancellation-race certification.

## Text boundary candidate: CPU-qualified only

`CowTextAdapter` renders reusable roots without an open assistant-generation
marker and full requests with one. It requires frontend-compatible encode
callbacks; literal-looking message tags are never independently interpreted by
this helper. Generated tokens remain canonical; linear token-by-token UTF-8
decoding is a presentation view. Pending bytes come from the original canonical
controller for empty-suffix continuation. Durability does not flush incomplete
characters; only a pinned EOS boundary completes a turn.

The full regression run initially failed (1 failure, 10 errors) because new
token events unconditionally required backend.is_eog. Corrected the coordinator
to derive optional turn_complete metadata from the pinned execution eos_tokens;
generic profiles without that policy preserve their existing event contract.
No existing test/assertion changed. Failed JSON retained as
`serving-pool-control-v2-before-fix.json`. After correction: supplemental suite
33/33, existing pool/control regressions 106/106, G0 51/51 (timestamp
20261014T223337895604Z). Candidate `serving-v2` is separate from frozen v1, so
the real GPU/HTTP results above do not qualify its new text-view/control code.
Next: actual frontend encoder binding, text HTTP integration and real-Q8 tests.

Frontend binding added in `CowPromptEncoder`: it delegates to the existing
Service rendering/encoding methods without constructing an engine or Service.
Root and generation modes match ordinary encoding, including literal-looking
tool/control/thinking tags. The encoder callback now receives generation mode
explicitly; earlier 33-case evidence predates that signature update. A fresh
combined run against the current source passed 35/35 on llm-60 (including both
ordinary-Service parity cases), no skips/errors/failures. Executed IDs and source,
dependency and template hashes: `SERVING-FRONTEND-CONTROL-2026-10-14.json`.
Parity uses ByteTokenizer/MockEngine for CPU fixtures, not the Q8 tokenizer/model.
Text endpoint integration and real-Q8 template validation remain next.

Private raw backup verified directly llm-60 to box-git, without routing an
archive through Windows:
`/srv/data/git/private_context_cow.evidence/20261014-http/http-q8-evidence-v1.tgz`.
244,539,456 bytes, mode 0600, parent directory 0700; matching SHA-256 on both hosts:
`3727ddd75ff1aad2a64b5da22fb0ec3e0baa96e02518deaf6199f576c1348749`.
Contains unabridged HTTP/reference results, frontier files, profile/fixture,
logs and control-suite evidence including the failed pre-fix regression.
Excludes store/checkpoint payload directories and model weights. It is an
evidence backup, not a complete resumable context-store backup.
