# Host-selected agent grammars

This optional prototype constrains tokens before greedy selection. The host
chooses READ, EDIT, TEST or FINISH and supplies that state's grammar. The model
cannot switch the host state by writing a state name or a tool call.

The four static files are in
`experimental/branching_context/examples/agent_grammars/`. READ permits
`read_file` for two named paths. EDIT permits a `propose_patch` object for
`src/main.py` with a JSON string. TEST permits `run_tests` for the unit suite.
FINISH permits a fixed finish object. These are example tool names; this change
does not install or execute those tools.

This first version accepts static GBNF only. JSON Schema input is rejected;
there is no partial schema compiler that silently drops unsupported keywords.

Executed qualification on llm-60: 297 combined CPU tests (41 focused), G0 51/51,
and a real Q4 CUDA restart demonstration over 7,231 prefix tokens. All four
actions completed (13/29/11/10 tokens including EOS). Saving READ after four
consumed tokens and resuming in new Python/native processes preserved tokens,
controller and final logits exactly. Both strict restores computed zero prefix
tokens; five final-frontier comparisons had zero error. This is a bounded
matching-operator restart test, not an independent fresh-history grammar oracle
or cumulative G1-G5 qualification. Failures encountered are retained in the
qualification record.

The request requires a native profile/readiness capability named
`consumed-selected-receipt-v1`. The isolated builder
`experimental/branching_context/scripts/build_grammar_probe.py` adds the actual
selected-token receipt at the native consumed boundary; an older worker is
rejected before constrained generation. Numerical kernels are reused. This
ticket qualifies CUDA only; ordinary serving endpoints, HIP and SYCL are unrun.

Initial generation speed for these short constrained actions was 11.1–12.7 TPS.
Full frontier reads and Python mask/controller processing remain costs. This
path has no native-speed performance qualification. The proposed Min-P-only
profile is separate and is not implemented here.

The private request-owned-controller follow-up measured23.1–30.4 generation TPS,
with CPU304/G051 and bounded exact restart passing. It preserves validation at
restore/publication and avoids copying the vocabulary policy between tokens.
See [the measured comparison](GRAMMAR_SPEED_20261016.md) for timing boundaries,
fingerprints and limitations. Host logit selection remains a major cost.

A further private native-selector measurement reached102.5-105.2 generation TPS
for these same short actions, CPU320/G051 and exact bounded restart passing.
The worker receives the legal mask and checks/selects from its existing full
frontier. See [native selection measurements](GRAMMAR_NATIVE_SELECTION_20261016.md).
Long-output/context and Q8 performance remain unqualified.

Install the optional dependency in the inference host's environment:

```bash
python -m pip install -r experimental/branching_context/requirements-grammar.txt
```

Use that host's verified interpreter. Windows uses `python`; the current Linux
campaign uses `/home/dflanag3/src/Strata/.venv/bin/python`. The existing serving
path does not import the optional parser for ordinary schema-1 controllers.

```python
from context_time_travel.grammar_first import GrammarEngine
from grammar_first_agent import load_grammars, generate_action, next_state

engine = GrammarEngine(tokenizer, load_grammars(), eos_tokens=profile_eos_tokens)
state = "READ"
pending_inputs = ()

while state is not None:
    result, action = generate_action(
        model, snapshot, engine=engine, state=state, stream=operation_id,
        input_ids=pending_inputs, max_tokens=256,
    )
    snapshot = result["snapshot"]
    if action is None:
        # Keep the partial snapshot. Do not execute an incomplete action.
        break
    outcome = supervisor.execute(action, state)  # Schema, semantics, permission, execution.
    new_state = next_state(state, outcome)
    # Use the real model template. Include the actual result, even for denial.
    # The action's tokens are already in snapshot; do not append them twice.
    pending_inputs = encode_supervisor_result(outcome, new_state)
    state = new_state
```

The supervisor and result formatter above are supplied by the host application.
`accepted`, `applied` and `passed` must come from the supervisor's actual checks
and execution, never from a claim in the model's output. A denial leaves the
state unchanged. A failed test returns to EDIT. A grammar-valid finish object
does not prove a patch is correct or that any tests ran.

If output reaches its token limit, the snapshot is a valid consumed **partial**
scope. Continue with `model.run_state_request(snapshot, (), n, engine=engine,
state=state, stream=operation_id, resume=True)`. That operation preserves the
original grammar, stream and progress. Reassemble the complete action from the
scope's canonical tokens before calling a tool. Do not treat its new suffix
alone as a complete JSON action. Starting a new request without `resume=True`
explicitly starts a new host-selected grammar scope.

At each position the selected token is the highest raw-logit legal token, with
the lowest token ID breaking ties. No random draw or forced-token shortcut is
used. EOS is legal only when the grammar can finish. Parser failure and empty
legal support raise an error; they never turn constraints off.

The immutable controller records schema version 2, the agent state, grammar and
tokenizer identities, parser version, generation start offset and the existing
consumption/decoder fields. A transient matcher is reconstructed by replaying
only that scope's tokens from canonical history. No opaque grammar state is
claimed to reside in the native KV serializer. Forked scopes get independent
matchers. Restoring a disk checkpoint restores numerical state; the supplied
compatible grammar engine independently restores its controller progress.

The opt-in request holds one native lease for all its output tokens and captures
once afterward. It reads full logits through the current backend protocol for
each selection, so its speed must be measured separately from unconstrained
native greedy generation. Stream callbacks are provisional. Use the existing
`save_branch` coordinator for a durable checkpoint and revision-checked branch
acknowledgment; a returned in-memory request is not that acknowledgment.

This example is a host integration starting point. It does not implement durable
tool side effects, exactly-once tool execution, a complete chat supervisor, or
GBNF enforcement in the ordinary OpenAI-compatible serving endpoints. The host
must persist its own workflow progress and operation IDs alongside its actual
tool results. Existing original G1-G5 qualification remains separate.
