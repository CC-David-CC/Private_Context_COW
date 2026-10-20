# Use Min-P with the COW CLI

This opt-in path uses the existing COW service, branch references, shared RAM and
disk pools, and durable request receipts. It does not change Strata's separate
ordinary chat server. Start with an already configured COW service and branch.
The server's pinned backend profile must advertise `full-vocab-min-p-u53-v2`
for all four settings and the temperature-zero sampled continuation. Historical
v1 profiles support positive temperature with Top-K 0 and Top-P 1.

Ask a new question using Min-P:

```sh
python tools/cow_cli.py --url http://YOUR-SERVER:8099 ask \
  --branch YOUR-BRANCH --text "Explain the error-handling path." \
  --max-tokens 512 --temperature 0.70 --min-p 0.05 --top-k 40 --top-p 0.95
```

On PowerShell, put the command on one line instead of using the shell's trailing
backslashes. Use the server's existing API-key setup through `COW_API_KEY`.
The CLI connects to that service; it does not load a model on your PC.

You can also use the four sampler flags with `answer` for a prepared initial
prompt, or with `append-user` after the previous answer has finished. Omitting
all flags on a new request keeps the existing greedy behavior. Any explicit
sampler knob opts into the preset: temperature 0.70, Min-P 0.05, Top-K 0,
Top-P 1.0, with the provided fields replaced. Temperature 0 alone retains the
legacy greedy path. Repetition penalty remains fixed at 1.0.

Continue an answer that reached its token limit:

```sh
python tools/cow_cli.py --url http://YOUR-SERVER:8099 continue \
  --branch YOUR-BRANCH --max-tokens 512 --temperature 0.9
```

Continue changes only the explicitly supplied settings. The command above
changes temperature while preserving saved Min-P, Top-K and Top-P. With no
sampler flags it keeps all saved settings. It preserves the RNG stream, draw
cursor and output-decoder boundary instead of starting at draw zero. A call
uses fixed settings throughout its whole chunk; the next call can change them.
Temperature 0 chooses greedily without consuming RNG draws. Returning to a
positive temperature resumes the saved RNG cursor. After a natural end-of-turn,
submit new input rather than trying to continue the stopped answer.

The output limit is a maximum. Natural stopping can produce fewer tokens.
To sweep multiple settings from the same prepared question, fork its saved
snapshot for each answer, or use the Python loop in
[Generation settings](GENERATION_SETTINGS.md). Reusing a question means choosing
its snapshot again, rather than appending one sweep's answer to the next.

Each new generation operation starts its own named random stream using its
operation ID. Different operation IDs can therefore produce different Min-P
answers while still reusing the same available prefix computation. A resumed
answer preserves its original stream even though the continuation has a new
operation ID. Forking preserves the snapshot's consumed controller state.

## Errors and retries

The coordinator signs the normalized sampling policy and resume decision into
its existing operation identity. It detaches the policy at admission so changing
a caller's dictionary cannot alter queued work. Greedy request signatures retain
their previous shape.

Do not reuse an operation ID with changed sampler settings, input or output
limits. After a lost acknowledgment, use the original request journal:

```sh
python tools/cow_cli.py --url http://YOUR-SERVER:8099 retry-request \
  --operation-id ORIGINAL-ID
```

That command replays the exact stored request. A committed operation returns its
original receipt, including after a service restart and after the branch has
advanced. It must not run the sampler again or move the branch backward.

If inference finished but the disk save failed, `retry-publication` retries the
existing consumed state while it remains eligible for recovery. It does not
sample a new answer. Capacity errors remain errors; optional checkpoint copies
may be evicted according to the configured pool policy, while retained canonical
history remains available for explicit reconstruction with a recomputation notice.

An older backend profile rejects unsupported settings before admitting an
inference job. Invalid ranges, booleans, nonfinite numbers, unknown fields and
non-unit repetition penalties are rejected. Grammar controllers are rejected by
plain text continuation; use the existing grammar-first Python API for those.
GBNF state selection through the service remains a separate ticket.

This is a supported-policy integration step, not original G1–G5 certification.
See the private serving qualification and current handoff for executed GPU sizes,
known failures and remaining long-context work. The [API guide](MIN_P_SAMPLING.md)
describes the sampler and matching-profile exactness contract.
