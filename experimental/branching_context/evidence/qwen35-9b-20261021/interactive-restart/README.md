# Executed interactive CLI restart check

Qwen3.5-9B Q8, FP16 KV, pinned llama.cpp adapter, separate Linux VM
BranchWrite 0.1.0.dev72 client, normal authenticated COW launcher on llm-49.

The CLI created a mobile-friendly dog website. After the server process was
terminated and a new process launched against the same durable store (empty RAM
pool), the still-running client requested a light-green background and italic
dog descriptions. Independent sandboxed Chromium desktop/mobile checks passed,
including a hidden-to-visible greeting button transition and no external requests.

The first post-restart request imported 851,842,570 native state bytes from the
saved 24,340-token prefix. Its 2,333 physical forward tokens equal the independently
counted newly consumed canonical tokens. Saved-prefix forward work was zero.
The checkpoint file checksum and canonical object identities were checked.
The file also contains 993,280 bytes of next logits, separate from native state.

This CLI run does not add a fresh 24K numerical-oracle comparison. Earlier bounded
matching-schedule numerical campaigns are separate. It does not qualify arbitrary
batch arrangements, broad controllers, physical GPU sharing or cumulative gates.

After successful completion, a test-driver command mistakenly included a newline
and triggered its single-line assertion before submitting `/history`. The terminal
closed; model work was already durable. This is retained as a harness failure,
not a model or backend failure. Interactive fork/retry checks remain outstanding.

PDF and CAD model baselines have not produced accepted artifacts. PDF attempts
included syntax, flowable and wrapping failures; CAD repeatedly chose unavailable
STEP export APIs. Native restart success does not establish model task competence.
