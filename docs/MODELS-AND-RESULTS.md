# Choose a model and understand its results

PCC is the model server; the downloaded weights determine much of what it can
do. The default large Flash download and the smaller Wen research models use
different engine routes inside the same COW system.

## Downloads

These are exact weight-download sizes, not total installed space or VRAM needs.
GB is decimal; GiB is binary. Packing, build products and context caches add
space. Do not use a weight-file size as a hardware qualification.

| Model | Download size | Download / setup |
| --- | --- | --- |
| **Qwen3.8 Flash UD-Q4_K_XL** | **111,334,654,784 bytes — 111.33 GB / 103.69 GiB** across the pinned shards | `./cow-setup`; [large-model installation](TWO-COMPUTER-SETUP.md). |
| **Wen 1.1 BF16** | **16,244,817,856 bytes — 16.24 GB / 15.13 GiB** | [Download the pinned GGUF](https://huggingface.co/davidflanagancomm/Wen/resolve/7328e9fab25b207523b5236aa29c81be24eed924/Wen-1.1-BF16.gguf?download=true); use the separate dense adapter below. |
| **Wen 1 BF16**, original comparison model | **15,942,827,584 bytes — 15.94 GB / 14.85 GiB** | [Download the historical GGUF](https://huggingface.co/davidflanagancomm/Wen/resolve/674907f2b65baa576e678b9c9a06bf07c3db9076/Wen-BF16.gguf?download=true). It is no longer in the current HF tree. |

[The Wen model card](https://huggingface.co/davidflanagancomm/Wen) describes the
research models. Current Wen is 1.1. It is mostly BF16, with three changed
final-MLP matrices stored in F32 after a tiny native LoRA merge; that explains
its larger download. The earlier pilot scored both versions 15/19 with the same
four failures. The version number does not establish a quality improvement.

### Download Wen with the Hugging Face CLI

Install the `hf` CLI in a separate download environment, then fetch the pinned
file. This downloads weights only; it does not configure or launch a server.

```sh
python3 -m venv .venv-download
.venv-download/bin/python -m pip install huggingface_hub==2.3.0
.venv-download/bin/hf download davidflanagancomm/Wen Wen-1.1-BF16.gguf \
  --revision 7328e9fab25b207523b5236aa29c81be24eed924 --local-dir models/wen-1.1
sha256sum models/wen-1.1/Wen-1.1-BF16.gguf
```

Expected SHA-256:
`49425fd802b04b400a1725e8afe527c405bb15e828d0275ff295e38396839f34`.
The public model does not require logging in. Add `--dry-run` to inspect the
download without fetching it. Both pinned commands were checked in dry-run mode;
the cached files used for evaluation also match the published size and hash.

For the original comparison model:

```sh
.venv-download/bin/hf download davidflanagancomm/Wen Wen-BF16.gguf \
  --revision 674907f2b65baa576e678b9c9a06bf07c3db9076 --local-dir models/wen-1
sha256sum models/wen-1/Wen-BF16.gguf
```

Expected SHA-256:
`40f048a00bc2a2a8c2db163999f0af90613d136376e9b3c9372457b1769b44ca`.

### Run the dense model

Follow [the pinned dense-adapter build and server guide](QWEN35-9B-BACKEND.md),
including its **Wen naming patch** before building the dependency. Select the
downloaded file with `cow_server_setup.py --model`, omit the Flash expert pack,
and use a new server directory for each model/profile. The guide's architecture
89 example is for a 4090; other GPUs need their actual build configuration.

`./cow-setup` remains the Flash installer. Supplying Wen weights does not turn
that installer into a dense-model setup. Wen is not qualified on every listed
GPU, and an AMD preview or two P4s do not inherit the A100 results.

## What has actually been measured

The retained llm-60 results use an RTX PRO 6000 Blackwell 96 GB, FP16 KV and
bounded workloads. They required no new use of that released machine.

| Model / workload | Result | Interpretation |
| --- | --- | --- |
| Flash Q4, six short questions | Median 74.63 generated tokens/s; 33.20 output tokens/s over the whole durable request | Generation and whole-request timing have different boundaries. |
| Flash Q8, same campaign | Median 62.29 generated tokens/s; 28.35 over the whole durable request | A bounded serving comparison, not an agent-quality score. |
| Flash Q4 and Q8, native pressure tests | Each matched 768 generated IDs and 12 full-logit vectors to independent references, maximum observed error zero | Checked context mechanics and numerical preservation. |
| IQ3_XXS / IQ2_XS on llm-60 | Inventoried, not qualified | Files present do not establish speed, correctness or task ability. |
| Lower quants on the 4090 setup | Short-chat passes plus coding timeout, memory and recovery limitations | Different hardware/workloads; [full campaign](../validation/lower-quants-llm49-20261021/). |

**[S³ owns the full comparison, provenance and timing definitions](https://github.com/CC-David-CC/sss/blob/main/docs/MODEL-COMPARISON.md).**
It keeps historical Flash evidence separate from the current paired Wen 1 /
Wen 1.1 tool-use trial. A same-suite leaderboard across all Flash quants and Wen
has not yet been established. No day-long real-world inference campaign is
claimed by these small tests.

## How testing and learning fit together

[Skills³ (S³): Skills, Scoring, Stability](https://github.com/CC-David-CC/sss)
owns the independent skill tree, tests, scoring, mixed practice and regression
reports. PCC owns native computation, training resources and immutable candidate
identities. BW owns task/tool execution and the checked lesson records.

An earlier 1,000-example narrow-skill Wen experiment improved its target wording
but lost five of 16 retention checks and barely improved blind generalization.
No hat qualified. The bounded five-skill mixed-replay pilot now completes four
rounds/16 updates/128 example slots with unchanged held-out score (14/40) and
no lost protected successes; it also produces no qualified hat. The machinery
works in that scope, while useful learning remains to be demonstrated. The
full [S³ learning evidence](https://github.com/CC-David-CC/sss/blob/main/validation/README.md)
includes failed engineering attempts and distinguishes them from model outcomes.
