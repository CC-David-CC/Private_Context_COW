# Q4 namespace-bound real-GPU pilot at full capacity

Bounded pilot PASS, not original G1-G5 or browser qualification.

Root inspected terminal raw result and used the existing HTTP summarizer to
retain profile/build/model identities, executed checks, counters and timings.
NAMESPACE-Q4-FULL96-PILOT-2026-10-14.json is that summary; its raw-result digest
binds the full remote evidence. NAMESPACE-Q4-FULL96-PREFLIGHT-2026-10-14.json
records both actual command capacities, model/build identities, CPU affinity and
pool configuration. Root independently verified the derived profile file SHA256
`b30688ad932e5fd99e0e1e9e691573f3080546e97351d1d2ca93b43b35bc4db6`.

On llm-60/RTX PRO 6000 Blackwell, the frozen `source-full96` candidate used
capacity 98,304, actual SUT/reference `--max-context 98304`, GPU=[0], FP16 KV,
8,192-token prefill chunks and shared pools of 96 GiB RAM/128 GiB disk with a
16 GiB host reserve. All 32 CPUs were available. The established 40 GiB resident-budget setting
is a host expert-cache budget, not a GPU-memory ceiling.

Actual prepared prefix: 1,612 tokens (1,577 source tokens), two distinct suffixes
of 58/44 tokens, eight actual generated tokens each. Namespace/context preparation,
source-bound forks, first-question appends, canonical workspace projection and
durable retries completed through HTTP. Saved-prefix work was not repeated:
request forward deltas were 58+8 and 44+8. Parent payload checksum stayed unchanged.
After SUT shutdown, independent native processes with the matching request
schedule matched all output tokens, complete greedy controllers, root frontiers
and final full frontiers exactly. Maximum error: 0. Native PID 1795772 exited 0;
the outer job was terminal when inspected. Total campaign: 68.097 seconds.

These short eight-token intervals measured about 61.6/63.3 received TPS, with
TTFT of 0.958/0.611 seconds and durable requests of 2.185/1.841 seconds. They are single short pilot
observations, not long-output performance estimates or medians. Earlier 12K
pilot data is not promoted into this result. Expansion to actual 8K source plus
roughly 512 new tokens and 128 outputs remains pending.

Source is frozen at the 953e4817-era namespace/provenance integration; newer
publication-recovery endpoints and browser code are outside this GPU result.
No every-token frontier, stochastic/grammar, browser, crash or cumulative-gate
claim. Original preflight failure and 12K-capacity pilot remain separate evidence.
Raw files and checkpoints stay under
`/home/dflanag3/spatial-artifacts/gpu-http-q4-namespace-luna-20261014/gpu-tiny-8-full96`.
Ordinary chat remains off; private Git only.
