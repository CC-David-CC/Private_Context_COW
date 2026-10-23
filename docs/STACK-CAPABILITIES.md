# BW, COW and native capability integration

The integration was developed on `feat/stack-unification-20261023`. Its optional
features have bounded qualification below; code presence alone is not native
qualification or a claim that every backend supports the same operations.

Strata and the dense native adapters own execution, native state and GPU work.
COW owns immutable histories/controllers, branches, checkpoints, placement and
resource limits. BW owns tasks, tools, verification and its one workflow ledger.
Both dense Wen and Qwen Flash MoE remain supported targets. Shared semantics
have common tests; model-specific features have explicit requirements.

## Delivery gates

1. Repair applicable tests; establish a checked backend descriptor, authenticated
   COW discovery and BW admission checks. Qualify both native model families.
2. Qualify recoverable inventory and local placement; add bounded notifications
   and paged recovery without changing canonical state.
3. Add explicitly configured peers, pinned/checksummed bounded checkpoint transfer
   and one COW publication authority. Begin with compatible Wen A100 workers.
4. Add optional prefill/decode roles only after complete-state handoff passes.
5. Evaluate optional fastokens against actual tokenizer IDs and complete workflows.

Every new optimization stays independently selectable. Existing FIFO, native BW
and standard CLI serving remain defaults. Source changes travel from the local
worktree through box-git/GitHub before deployment. Runs release their owned GPU
allocations; ephemeral numerical caches and abandoned staging are cleaned up.
Canonical histories and BW workflow records are separate from numerical caches.

## Capability contract

`GET /cow/info` remains schema 1 and unchanged for older clients.
Authenticated `GET /cow/capabilities` adds schema 1, model/execution identities,
worker descriptors, feature support/enabled/qualification states, scheduler and
RAM/disk limits. No weights paths, build paths or credentials are published.
Descriptors are detached observations collected after native startup validation;
reading them does not acquire a GPU lease. Startup-checked is not G1-G5 qualified.
Worker failure/replacement does not authorize a different model or profile.

An execution-profile-bound codec is not a portable format. Consumers still need
the complete original identity, native restore checks and canonical controller.
No descriptor permits importing Wen state into Qwen Flash, mixing adapters, or
relabelling an execution profile. Unsupported grouped native capture is distinct
from supported independent workers.

## Test campaigns and the earlier 36 errors

Run `python tools/cow_campaign.py --campaign cpu --report <private-output.json>`
on Linux. The checked manifest classifies every implementation test module;
unclassified additions fail before execution. Native tests are selected before
import, and a selected campaign with missing prerequisites is BLOCKED. Tests in
other campaigns are NOT_APPLICABLE to that invocation, never counted as passes.
The original cumulative `scripts/verify.py --gate G0` and G1-G5 requirements remain.

The earlier collection errors comprised 20 missing native/diagnostic/calibration
prerequisites, six import errors, six incomplete snapshot fixtures, one incomplete
coordinator fixture, one incomplete CMake fixture and two source-patching tests.
The source timing patch is already integrated; its helper now validates and
accepts the complete existing block without instrumenting it twice. The older
prefill patch generator requires an explicit pre-integration source tree and
belongs to `legacy-build`. It is not a current native-runtime qualification.
The retained Q8 evidence check previously skipped by default is a separate
`retained-evidence` campaign with explicit prerequisites.

Each execution report records its source revision and dirty state, manifest hash,
failures, errors, skips and excluded modules. Unknown campaign names or malformed
requirements fail validation; they cannot quietly exclude a required test.
Imported fixture classes are deduplicated by test identity. Remaining native or
historical prerequisites remain visible even when the CPU campaign passes.

The first repaired Linux CPU campaign at `816ab0d` passed 2,033 tests with no
failures, errors, skips or expected failures. G0 passed its 51 reference tests.
BW at `8b821b0` passed 63 focused connection, receipt, retry, lost-acknowledgment
and stream tests against the actual COW CPU HTTP/CLI stack. The isolated test
environment includes the pinned `llguidance==1.9.1` dependency and explicitly
selects this checkout's `tools/cow_cli.py`. These results establish CPU boundary
behavior only.

The inventory/notification increment at `e46921f` passed 2,039 Linux CPU tests
with no failures, errors, skips or expected failures, plus all 51 G0 tests.
Child-process import paths are part of the campaign environment. A full CPU
campaign without its required C++ compiler reports BLOCKED before imports;
explicit pure-Python module selections remain usable on such a host.

Wen 1.1 on A100 and Qwen Flash Q4 on RTX PRO 6000 each passed a complete BW
code-repair/test/finish workflow with the optional features off and on. The
Wen BW client ran on a separate LAN host over SSH forwarding. Reopening completed
workflows left COW branch heads unchanged. Both configurations shut down with
zero ephemeral session directories. Wen additionally recovered the complete
ten-location inventory over ten HTTP pages with duplicate delivery. These are
bounded full-task checks, not broad model-quality, throughput or G1-G5 claims.
Those first receipts remain unchanged; the subsequent increments are recorded in
[STACK-UNIFICATION-FINAL-20261023.json](../experimental/branching_context/evidence/STACK-UNIFICATION-FINAL-20261023.json).

The final CPU campaign at `9b90735` passed 2,077 tests with no failures, errors,
skips or expected failures. G0 passed 51, and tokenizer/HTTP/setup regressions
passed 27. A transfer fault selection at the same revision passed 25 overlapping
tests, including deadline expiry and killing a receiver during actual staging.
These counts must not be added together.

BW at `211717d` passed 158 focused integration tests. The broader selection
exposed an offline-replay regression: completed workflows unnecessarily queried
server capabilities. Discovery now applies to unfinished work; completed replay
uses the original ledger. The real HTTP/subprocess test closes the listener
before reopening the finished result. Outdated synthetic model IDs/discovery
fixtures were corrected without relaxing production identity checks.
The related COW HTTP/recovery selection at `1c3c08b` passed 53 tests. Its older
restart fixture now checks the existing durable operation-lookup contract with
production checkpoint annotations. Six saved native BW workflows reopened with
their servers stopped, returning identical results and unchanged ledger/journal
bytes. No GPU was allocated for those final replay checks.

Native configured-peer recovery on A100 passed for an existing source and an
identical new root. GPU 3 supplied the completed checkpoint; GPU 0 produced the
same complete response and controller without repeating prefix work. Full BW
workflows also passed with combined workers and with explicit prefill/decode
roles. Roles divide root creation/input preparation from continuation requests;
suffix prefill remains on the generation worker.

LAN cache storage round-trips from llm-60 through llm-49, llm-79 and r730 passed
checksums, canonical-state preservation, duplicate handling and six complete
native response/controller comparisons on return to llm-60. Two concurrent
receivers passed. No A100 WAN cache bandwidth test was run. Storage-peer CPU
validation does not establish native execution on another GPU architecture.

Optional fastokens 0.3.4 passed 5,180 token-ID checks across Wen 1.1 and Qwen Flash
Q4. Both models passed complete BW workflows with it off and on, including
pause/resume and replay of finished work. The selected adapter retains COW's
ordinary short-piece path and accelerates long BPE pieces. Default configuration
still uses the Python tokenizer. Dependencies were installed only on llm-60.

See [CACHE-FABRIC.md](CACHE-FABRIC.md) for accepted flags, bounds, configuration,
measured timings and scope. Remote inference dispatch, a production storage-only
cache service, cross-profile checkpoint portability, grouped native restore and
running-request migration remain outside this increment. The Strata C++ kernels
were not changed. No broad throughput or cumulative G1-G5 claim is made.

## Inspiration

Thank you to DeepInfra, NVIDIA Dynamo, KVCR and fastokens contributors for the
published designs. See [CACHE-FABRIC.md](CACHE-FABRIC.md) for the reference links,
attribution and bounded results of the preceding experiment.
