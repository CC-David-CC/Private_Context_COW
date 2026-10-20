# Distinguish a rejected request from a lost reply

CLI exit code 2 means an error. It cannot by itself prove the server rejected a
mutation. A request may have committed before its completion reply was lost.
Streamed text does not prove a durable result either.

With `--machine-result`, handled errors now also return this JSON envelope:

```json
{"schema":1,"error":{"outcome":"indeterminate","operation_id":"original-id","message":"stream ended without a complete durable receipt"}}
```

`rejected` means explicit HTTP 4xx rejection or a local error before a POST may
have been sent. After dispatch, missing/malformed replies and uncertain failures
remain `indeterminate`. The normal exit code and stderr message are preserved.
An operation ID can be null when local validation failed before admission.

BranchWrite requests machine results for mutations. It accepts rejection only
from the exact error-envelope shape and a matching operation identity (or an
explicit null before admission). Unknown/legacy exit-2 failures remain uncertain.
It retains the original intent and never automatically sends a replacement.

Resolve an uncertain request using the existing operation lookup and original
client journal. A successful reconciliation returns the original saved receipt.
A new ID means new work. Current operation lookup is coordinator-process scoped;
server restart after an unacknowledged commit needs separate qualification and
may return unknown. Do not claim universal exactly-once recovery.

The qualification proxy deliberately withholds only the actual completion after
upstream durable publication. It forwards real request bytes/authentication and
does not mock the model. Authentication is never included in proxy evidence.
Client-process recovery is distinct from server/native-process restart.

The paired Q4 fixtures keep the simple and expanded position questions. Record
image-answer quality separately from loss/recovery mechanics. The short question
has an observed `NOT_VISIBLE` miss for a known left-side square; the owner
accepts this documented model edge while continuing the prototype. It is still
a quality FAIL. No expected label, legal-choice mask, numerical tolerance or
protected H02 assertion is changed. Q8 must exercise both in its later end-to-end
campaign; it is deferred, not assumed equivalent or passing.
