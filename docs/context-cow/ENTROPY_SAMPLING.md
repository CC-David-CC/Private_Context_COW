# Typical and DynaTemp generation settings

These settings apply to one generation. Forks inherit the saved settings and
draw cursor. A later continuation may change settings without resetting its
accepted tokens, grammar/parser progress, output decoder or random stream.
Settings cannot change between selecting a token and consuming it.

| Setting | Meaning | Disabled value |
|---|---|---|
| `typical_p` | Retain candidates near the distribution's expected information content | `1` |
| `dynatemp_range` | Adapt temperature between `max(0, temperature-range)` and `temperature+range` | `0` |
| `dynatemp_exponent` | Shape the normalized-entropy response | `1` is the default exponent |

Typical-P must be in `(0, 1]`. DynaTemp range must be nonnegative, its exponent
positive, and its upper temperature finite. Boolean and non-finite settings
are rejected. A zero base temperature keeps greedy selection and uses no RNG
draws. A positive base temperature uses one draw even if DynaTemp computes an
effective temperature of zero.

The pinned order is legal support, optional DRY, Min-P, Top-K, Typical,
static/DynaTemp temperature, Top-P, categorical selection. Typical ranks
`abs(-log(p)-entropy)` with token-ID tie breaking. It retains candidates until
their mass is strictly greater than `typical_p`. DynaTemp uses normalized
entropy of the remaining support, raised to its configured exponent.
These definitions specify the prototype's arithmetic; they do not promise
bit parity with other engines' samplers or improved answer quality.

## CLI examples

These commands require an already configured COW service with a v4-capable
execution profile and an existing branch named `code-review`. The service URL
below is the default local URL. Remote services require the existing API-key
setup through `COW_API_KEY`. This guide does not start a server; the normal chat
server remains off during the private campaign.

Typical alone:

```text
python tools/cow_cli.py --url http://127.0.0.1:8099 ask --branch code-review --text "Review this code" --temperature 0.80 --min-p 0 --top-k 0 --top-p 1 --typical-p 0.95 --max-tokens 128
```

DynaTemp alone:

```text
python tools/cow_cli.py --url http://127.0.0.1:8099 ask --branch code-review --text "Review this code" --temperature 0.55 --min-p 0.06 --top-k 0 --top-p 1 --dynatemp-range 0.35 --dynatemp-exponent 1 --max-tokens 128
```

Change only the dynamic range when continuing an unfinished answer:

```text
python tools/cow_cli.py --url http://127.0.0.1:8099 continue --branch code-review --dynatemp-range 0.25 --max-tokens 128
```

Omitted continuation fields inherit their saved values. A new `ask` starts a
new generation scope. To disable inherited Typical/DynaTemp explicitly on a
continuation, set `--typical-p 1 --dynatemp-range 0`.

## Executed validation and limits

On llm-60 GPU0, combined Typical+DynaTemp+DRY passed Q4 API128/64/8 and Q4/Q8
HTTP/CLI1024/64/16 process-separated restart campaigns. There were 106 exact
full-logit-vector comparisons, zero maximum error, identical tokens and
controllers, and three explicit strict-restore receipts with zero forward
tokens. Both HTTP/CLI runs changed settings on continuation. Parent state and
retry receipts passed their existing audits. All native/worker processes
exited cleanly. CPU564/564 and G051/51 passed without skips.

Resident Q4 timing at1024-prefix/64-suffix/128 outputs then measured median
generation103.04/103.45TPS on branches A/B. Complete requests took1.638/1.632s,
with first consumed outputs at245.2/241.5ms and median token intervals9.68/9.64ms.
Each branch had one warmup and three measured repetitions. Integrity checks
remained enabled; startup, per-token audit reads and whole-parent byte audits
were excluded from the timed request. This is direct Python COW API timing,
without HTTP transport or a durable full disk checkpoint write per answer.
All tokens/controllers and final timed frontiers matched their references;
the separate preceding audit checked390 complete vectors. See
`ENTROPY-Q4-TIMING-1K-2026-10-16.json` for individual trials and work counters.
A controlled overhead percentage and longer-context results are outstanding.

This result qualifies that combined configuration at the tested sizes. The
standalone examples above have policy, CLI and CPU selection coverage; their
separate real-GPU selection audit is still outstanding. Matched sampler overhead,
long-context behavior, full GBNF GPU coverage and HIP/SYCL remain unmeasured.
XTC, Mirostat2 and Adaptive-P are not serving-integrated at this checkpoint.
Original cumulative G1-G5 remain unqualified; cross-arrangement equivalence
remains an accepted prototype limitation.

Evidence: `experimental/branching_context/evidence/ENTROPY-QUALIFICATION-2026-10-16.json`.
The exact 19 executed overlay files match private Git source commit
`9737850c4eaa958cd693d8233622dbff1981ba06`, as recorded separately in
`ENTROPY-SOURCE-VERIFICATION-2026-10-16.json`. Raw evidence was fully streamed
back from box-git and every archived member verified; see `ENTROPY-BACKUP-2026-10-16.json`.

## XTC development checkpoint

XTC is wired in the private integration branch with `--xtc-probability` and
`--xtc-threshold` on ask, answer, append-user and continue. Values must be finite
numbers in [0,1]. New-generation defaults are probability0 (disabled) and
threshold0.1; continuation inherits omitted values. Active XTC requires the
pinned native V5 profile and a schema4 controller. Greedy uses no RNG draws.
Eligible stochastic XTC uses one activation draw plus one categorical draw;
unsupported profiles reject the request. The combined CPU suite629 and G051
passed. The runtime results below supersede this CPU-only checkpoint.
The normal chat service remains off.

## XTC executed validation

Committed private source `0d7efac9` passed Q4 API128/64/8 and Q4/Q8
HTTP/CLI1024/64/16 restart campaigns: 106 full-vector comparisons, zero
maximum logit error, identical tokens/controllers, and three strict restores
with zero saved-prefix forward work. The independent collector reread98 raw
candidate/reference vector pairs. Native counters recorded88 selections,
52 activation draws, five activations and seven removed candidates. Both
activation and nonactivation were exercised. CPU629 and G051 passed with
zero failures/skips. All campaign processes exited cleanly.

On Q4 with8192 prefix tokens,512 suffix tokens and128 outputs, the resident
Python COW API measured median95.71/95.15 generationTPS on branches A/B.
Complete requests took1.909/1.913s; first consumed outputs arrived at406.9/408.4ms.
Each branch had one warmup and three measured repetitions. Integrity checks,
COW and consumed-state publication remained enabled. Startup, HTTP transport,
per-token oracle reads and a full durable checkpoint write per answer were
outside the timer. All390 preceding full-vector comparisons and eight timed
final-frontier checks had zero error; tokens/controllers matched. Timed calls
performed only the512 suffix tokens plus128 generated tokens, with no prefix
recomputation or disk loads.

This is the XTC preset, not the earlier combined Typical+DynaTemp+DRY policy.
Different policies and generated answers prevent a controlled overhead claim.
These bounded results do not qualify arbitrary contexts, grammar workflows,
fault scenarios, HIP/SYCL or original cumulative G1-G5.

Evidence: `XTC-QUALIFICATION-2026-10-16.json`,
`XTC-SOURCE-VERIFICATION-2026-10-16.json` and
`XTC-Q4-TIMING-8K-2026-10-16.json` under the branching evidence directory.
Mirostat2 controller/native/CLI integration remains in progress; Adaptive-P
integration, general GBNF CLI controls and evaluate/adopt CLI commands remain
outstanding.

## Mirostat 2 integration checkpoint

The private CLI now accepts `--mirostat 2`, `--mirostat-tau` and
`--mirostat-eta` on generation actions. The combined source passed688 CPU
checks and G051; real-GPU restart qualification is still UNRUN at this
checkpoint. These flags require the new pinned V6 execution profile.

For a new request, `--mirostat 2 --temperature .8 --mirostat-tau 4
--mirostat-eta .05` selects standalone feedback sampling. Min-P defaults to0
for an explicit new mode2 request; Top-K is0 and Top-P is1. Explicit conflicting
probability filters are rejected. DRY can be enabled before Mirostat. This
prototype does not combine Mirostat with active XTC, Typical or DynaTemp.

Continuation sends only fields you supply and inherits the rest. Changing
temperature, tau or eta while Mirostat remains active preserves its current
feedback value, mu. Disabling with `--mirostat 0` clears feedback; enabling
it again initializes mu to twice tau. To switch an active Mirostat continuation
to greedy, also disable Mirostat explicitly. Forks preserve independent copies
of the committed feedback and RNG cursor. Sampling does not commit either
until the selected token has been consumed.

See `MIROSTAT2-INTEGRATION-2026-10-16.json` and
`MIROSTAT2-INTEGRATED-CPU-2026-10-16.json`. No Mirostat throughput, answer-quality
or GPU correctness claim follows from these CPU results.

The later bounded GPU campaign on committed source `781004f1` passed Q4
API128/64/8 and Q4/Q8 HTTP/CLI1024/64/16: 106 complete-vector comparisons
with zero logit error, identical tokens/controllers and three strict restores
with zero saved-prefix forward computation. The native counters recorded88
Mirostat selections. HTTP continuation changed temperature, tau and eta while
preserving feedback. All campaign processes exited successfully. This supersedes
the CPU-only runtime status above; it does not qualify broad grammar/fault
combinations or original cumulative G1-G5. Resident throughput is a separate
measurement. See `MIROSTAT2-QUALIFICATION-2026-10-16.json`.

## Linear Mirostat timing and restart validation

The later exact linear-retention implementation on source de26ad5a passed
Q4 and Q8 restart audits:458 full-vector comparisons, zero error, and six
strict restores with zero saved-prefix computation. CPU690/G051 passed.
On the matched Q4 8192/512/128 resident workload, generation rose from
32.14/32.16 to78.84/78.69 TPS for A/B. Host selection fell from21.48 to2.91
ms/output while model compute stayed about7.76ms. Integrity checks remained
enabled. Each timed request includes512 suffix-prefill tokens; startup and
full checkpoint writes are outside the timer. Q8 throughput is unmeasured.

See [the figure, raw measurements and scope](measurements/mirostat2-linear-20261016/README.md).
The current combined Adaptive-P extension is a separate qualification effort;
its CPU tests do not imply a GPU pass or supersede this measured source.
