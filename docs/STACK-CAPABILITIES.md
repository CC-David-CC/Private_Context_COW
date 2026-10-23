# BW, COW and native capability integration

The implementation is on `feat/stack-unification-20261023`. Work is incremental;
code presence is not a native qualification or a main-release claim.

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
behavior only. Native Wen/Qwen workflows and later delivery gates remain open.

## Inspiration

Thank you to DeepInfra, NVIDIA Dynamo, KVCR and fastokens contributors for the
published designs. See [CACHE-FABRIC.md](CACHE-FABRIC.md) for the reference links,
attribution and bounded results of the preceding experiment.
