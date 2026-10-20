# Min-P sampling with Context COW

For configurable temperature, Min-P, Top-K and Top-P on each generation call,
including partial continuation and sweeps from a shared prefill, see
[Generation settings](GENERATION_SETTINGS.md). The v1 measurements below remain
historical evidence for the original Min-P preset.

Min-P is an opt-in sampling path for the branching API. Greedy remains the
default. The first supported preset is:

```python
sampling = {
    "temperature": 0.70,
    "min_p": 0.05,
    "top_k": 0,
    "top_p": 1.0,
}
```

Repetition penalty is fixed at 1.0. Top-K, Top-P filtering, adaptive samplers and
custom stop sequences are outside this version's supported sampling policy.

## Use the branching API

With a prepared `model` and an immutable prefix snapshot:

```python
answer = model.run_sampled_request(
    prefix_snapshot,
    question_token_ids,
    128,
    sampling=sampling,
    stream="code-review-A",
)
```

To combine Min-P with a registered GBNF grammar:

```python
answer = model.run_state_request(
    prefix_snapshot,
    question_token_ids,
    128,
    engine=grammar_engine,
    state="EDIT",
    stream="code-review-A",
    sampling=sampling,
)
```

The grammar sets legal tokens. Min-P then removes unlikely legal tokens and
samples the rest. The host still checks meaning and permissions before
executing a proposed tool action.

To continue a partial answer, use its returned snapshot with no new input:

```python
continued = model.run_sampled_request(
    answer["snapshot"], [], 128,
    sampling=None, stream="ignored-on-resume", resume=True,
)
```

For a grammar answer, use `run_state_request` with the same engine/state and
`resume=True`. The saved controller supplies the policy, random stream, draw
cursor, parser progress, token counts and pending UTF-8 decoder bytes. A new
request can select a new stream; resuming an existing request preserves its
stream. Forking a partial answer copies its consumed controller boundary.
Starting a new request starts that named stream at draw zero. Give independent
new requests distinct stream names if you want independent random draws. Using
the same name deliberately makes the draw sequence reproducible; it does not
turn Min-P into greedy selection.

The returned snapshot is an immutable consumed state. It is not a durable save
acknowledgment. Use the existing `save_branch` operation to publish a disk save
and a revision-checked branch move. `restore_checkpoint` explicitly requires
that disk image and reports native work; it cannot silently recompute a prefix.

## What is selected

The declared order is legal support, Min-P, temperature, then categorical
sampling. The cutoff is inclusive:

`raw_logit >= maximum_legal_logit + log(min_p)`.

The sampler checks the full finite logit vector, including masked positions.
It does not substitute Strata's older capped shortlist for the full vocabulary.
Weights and the cumulative sum use binary64, in increasing token-ID order.

The branch owns its HMAC-SHA256 random stream and draw cursor. The host passes
one exact 53-bit draw to a read-only native selector. A sampled token cannot
be published until it has been consumed. Failed or cancelled requests do not
publish their provisional RNG progress.

Schema 3 is the versioned canonical controller record for this path. Old
schema-1 and schema-2 records remain readable. Checkpoint payload format is
unchanged. A pinned native profile must advertise
`full-vocab-min-p-u53-v1`; unsupported profiles fail before materialization.

## Validation runner

On llm-60, the frozen audit runner is:

```sh
root=/home/dflanag3/spatial-artifacts/minp-integration-20261016-v1
PYTHONPATH=/home/dflanag3/spatial-artifacts/grammar-first-20261016-v1/deps \
/home/dflanag3/src/Strata/.venv/bin/python -B \
  "$root/observer-v3/run_min_p_probe.py" \
  --stage campaign --runtime-root "$root/green-v2/source" \
  --profile "$root/q4-profile-v2/profile.json" \
  --fixture /home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1/fixtures-v1/0-native_cow_engine/fixture.json \
  --run-dir "$root/my-new-run" --prefix 1024 --suffix 64 --outputs 16 \
  --timed-repetitions 3
```

Use a new run directory. Hold the established GPU lock and check that no other
inference job is running. Add `--grammar` with `review.gbnf` to test the combined
path. The two fresh references never import checkpoints or invoke the native
sampler; they read full logits and use the independent Python selector. All
audit output frontiers are checked. Later resident timing repetitions exclude
those per-token audit reads, but keep backend integrity checks enabled.

Sampling byte counters describe logical input sizes: one full logit vector and
one packed legal-token mask per choice. They are not measured memory traffic.
The selector makes multiple passes over those inputs. Its native timer measures
selection arithmetic, excluding mask-file I/O and protocol overhead.

The runner terminates the producer's Python/native processes, strictly restores
the prefix in a new process, tests both suffixes, then restores a save after four
consumed output tokens. It compares complete controllers and tokens, not just
sample seeds. Actual output counts and early stopping are retained.

This is a private API experiment, not an ordinary-server sampler integration or
an installer promise. Exactness is tied to the same model, execution profile and
native operator/batch boundaries. It does not require different random streams
to give identical answers, establish cross-arrangement equivalence, or certify
the original G1–G5 gates. See [executed results](MIN_P_RESULTS_20261016.md)
and the machine-readable Min-P qualification evidence for the tested scope.
