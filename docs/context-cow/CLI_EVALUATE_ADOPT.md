# Evaluate alternatives from the CLI

These commands are integrated into the private COW branch. The CPU checkpoint
has 20 passing new tests and G0 51/51. A broader run has two unchanged-base
prepared-presentation assertion failures. A subsequent bounded Q4 CLI workflow
and independent numerical audit passed: 47 exact full-frontier comparisons,
maximum error zero, matching tokens and controllers. That audit covers the
earlier frozen G4 source and a 181-token prefix with eight-token answers and
continuation; it is not qualification of the later retry/grammar merge or
cumulative G4 certification. See the machine-readable
[G4 CLI evidence](../../experimental/branching_context/evidence/G4-CLI-CPU-2026-10-16.json).
The [numerical audit](../../experimental/branching_context/evidence/CLI-WORKFLOW-NUMERICS-2026-10-16.json)
records the independent oracle and exact execution boundaries.

Connect to the COW service URL supplied by its operator. Set `COW_API_KEY` in your
client environment. The CLI uses Python's standard library and does not load a
model. The examples below use `http://SERVER:PORT` as a placeholder.

First load or prepare a context. Fork candidate branches and generate their
answers using the existing `fork`, `ask` or `answer` commands. Keep the returned
snapshot IDs. One saved prefill can support several candidates; requests remain
serialized on the current single-GPU configuration.

Write an evaluation specification in a local JSON file. Use the shared task
snapshot ID from your preparation receipt as `task_root`. For a mechanical
example, put the following in `evaluation.json` and replace the placeholder:

```json
{
  "task_root": "PASTE_SHARED_TASK_SNAPSHOT_ID",
  "evaluator": {
    "name": "fixture-check",
    "version": "1",
    "source": "result = {'mandatory': {'fixture_ok': 'PASS' if fixtures['expected'] == b'PASS' else 'FAIL'}, 'score': len(candidate['tokens']), 'unresolved': []}"
  },
  "metric": {"name": "consumed_tokens", "order": "maximize"},
  "seed": 17,
  "mandatory": ["fixture_ok"],
  "timeout_seconds": 5
}
```

Create a local `expected.bin` containing exactly the four ASCII bytes `PASS`.
The example scorer checks the fixture and ranks by token count. It demonstrates
the workflow; it does not measure answer quality. Supply your own deterministic
scorer and meaningful checks for a real task. The scorer receives detached
candidate token/controller data and fixture bytes in the existing isolated
Linux evaluator. It cannot operate the live model or execute your host tools.
The host remains responsible for semantic validation and permissions.

Evaluate each candidate with its own operation ID:

```text
python tools/cow_cli.py --url http://SERVER:PORT evaluate --snapshot-id SNAPSHOT_A --spec evaluation.json --fixture expected=expected.bin --operation-id eval-a
python tools/cow_cli.py --url http://SERVER:PORT evaluate --snapshot-id SNAPSHOT_B --spec evaluation.json --fixture expected=expected.bin --operation-id eval-b
```

The client reads fixture files on your computer and sends their bytes. A fixture
argument is never interpreted as a path on the inference server. Reports contain
content hashes, immutable candidate identities and evaluator provenance.
Mandatory check values are status strings such as `PASS` and `FAIL`, not JSON
booleans. Keep each report's returned `evaluation_id`.

Inspect and select:

```text
python tools/cow_cli.py --url http://SERVER:PORT inspect-evaluation --evaluation-id EVALUATION_A
python tools/cow_cli.py --url http://SERVER:PORT select-evaluations --evaluation-ids EVALUATION_A EVALUATION_B --mandatory fixture_ok
```

The server validates reports and chooses the winner. A client-supplied score or
winner is not accepted. Mandatory checks must pass; incomparable or conflicting
reports are rejected. Selection alone moves no branch.

Read `branches` to obtain the destination branch's current revision, then adopt:

```text
python tools/cow_cli.py --url http://SERVER:PORT branches
python tools/cow_cli.py --url http://SERVER:PORT adopt-evaluation --branch main --revision CURRENT_REVISION --evaluation-ids EVALUATION_A EVALUATION_B --mandatory fixture_ok --operation-id adopt-a
python tools/cow_cli.py --url http://SERVER:PORT history --branch main
```

Adoption moves `main` to the selected existing snapshot. Candidate branches and
old results remain available. A stale revision or pending branch publication
rejects new adoption. If the answer is still active, use `continue`; if its turn
has ended, use `append-user` for the next question.

For a lost acknowledgment, retain the operation ID and replay its exact saved
request with `retry-request`. Reusing that ID with a different candidate set,
revision, specification or fixture is rejected. The server also retains durable
evaluation and adoption receipts across restart, including recovery between the
successful branch move and its final adoption acknowledgment. An evaluator that
has not committed its report may run again after interruption; isolated scoring
must not perform host side effects.

The HTTP routes are authenticated `POST /cow/evaluate`,
`GET /cow/evaluation/<id>`, `POST /cow/evaluation/select` and
`POST /cow/evaluation/adopt`. Inspection and selection do not move branches.
These routes require the existing fail-closed Linux evaluator isolation; they do
not silently substitute an unsandboxed scorer on unsupported systems.
