# Choose sampler settings for each generation

The defaults `temperature=0.70`, `min_p=0.05`, `top_k=0`, and `top_p=1.0`
are a preset. They are not the only supported settings. A generation call fixes
its settings before selecting its first token. A later call can choose different
settings. There is no per-token settings callback.

Keep a prepared immutable snapshot containing the shared material. Run several
generations from that same snapshot. Each returns a child snapshot; none changes
the prepared parent. A resident checkpoint can supply the prefill each time. A
strict disk restore can supply it after restart, without recomputing its tokens.
An unavailable checkpoint can require reconstruction on the ordinary recovery
path; strict restore fails instead of silently replaying.

## Python API

With an already configured `BranchingModel` and a fully consumed `question`
snapshot:

```python
settings = dict(min_p=0.05, top_k=40, top_p=0.95, repetition_penalty=1.0)
answers = []
for temperature in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    answer = model.run_sampled_request(
        question, (), 512,
        sampling=dict(settings, temperature=temperature),
        stream="question-temperature-sweep",
    )
    answers.append(answer)
```

The empty suffix means the whole question is already consumed. The output limit
is 512 tokens; a natural stopping token can end a call earlier. The shared stream
name makes each independent sampled answer start from the same draw sequence.
Use different stream names when independent streams are wanted. Different
temperatures need not produce different answers.

Continue one partial answer with new settings:

```python
continued = model.run_sampled_request(
    answers[2]["snapshot"], (), 512,
    sampling=dict(settings, temperature=0.9),
    stream="ignored-on-resume", resume=True,
)
```

Resume preserves the saved RNG stream and cursor, repetition counts, stop state,
and pending UTF-8 bytes. Grammar-first `run_state_request` also preserves its
parser progress and host-selected agent state. It changes the sampler policy
once on a detached request-owned controller. The old snapshot stays unchanged.
A stopped answer cannot resume; start a new generation scope instead.

## Ranges and fixed arithmetic

| Setting | Accepted values | Meaning |
|---|---|---|
| Temperature | Finite number at least 0 | 0 chooses the highest legal logit; positive values sample. |
| Min-P | Finite number from 0 through 1 | 0 disables this relative-probability filter. |
| Top-K | Integer from 0 through 2^64−1 | 0 disables it; larger than the support retains that support. |
| Top-P | Finite number greater than 0 through 1 | 1 disables nucleus filtering. |
| Repetition penalty | 1.0 | Other penalties remain outside this extension. |

For positive temperature the recorded order is legal/grammar support, Min-P on
raw logits, Top-K, temperature, Top-P, then categorical selection. Top-K and
Top-P rank by descending raw logit, breaking ties by ascending token ID. Top-P
keeps the smallest ranked prefix reaching its threshold and always retains at
least one token. Categorical accumulation uses ordinary binary64 additions in
ascending token-ID order. Grammar constraints always apply first.

Temperature 0 chooses the maximum legal raw logit, breaking ties by ascending
token ID. It consumes no RNG draw. The other settings remain recorded so a later
call can return to sampling. These filters all retain a highest-scoring token.

The native extension advertises `full-vocab-min-p-u53-v2`. Historical v1 profiles
continue to support positive temperature with Top-K 0 and Top-P 1. Other settings
require v2 and are rejected before materializing incompatible work. The default
greedy request path remains opt-in unchanged: omitting sampler settings still
uses its existing controller and request identity.

Controller schema 3 is retained. Its recorded sampler order identifies the
enabled filters, and validates against its policy. Existing v1 controller
records remain readable. Execution profiles include the native capability and
build identity; checkpoints are not silently exchanged between profiles.

## Retries and exactness

A new operation ID means new work. Reusing an operation ID requires the original
normalized settings, input, resume choice, revision, and output limit. A retry
returns the original receipt rather than sampling again. A continuation has a
new operation ID but retains the earlier RNG stream.

Exactness means matching tokens, settings, initial controller, model, operator,
and native batch boundaries. A settings switch must occur at the same generation
boundary in the reference. Cross-arrangement equivalence and original G1–G5
qualification remain separate limitations. See the executed evidence for tested
models, sizes, failures, and the distinction between CPU fixtures and real GPU
measurements.
