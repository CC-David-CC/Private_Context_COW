# Atomic presentation storage integrated

Luna's Store patch51e9dba is integrated as95208bdc. Namespace-bound branches
use explicit schema2; legacy unbound schema1 branch bytes and request digests
are unchanged. Typed annotations are immutable content-addressed JSON blobs
published before the existing atomic branch reference rename. That same rename
binds head, event, operation receipt and annotation digest. Missing/corrupt
referenced annotations fail reads; changed text under a retry ID conflicts.
An uncertain blob-directory or branch-directory flush is completed on retry.

Fork operations have explicit namespace and operation identities. Equal model
roots cannot identify a chat. An active-answer fork resolves its exact source
snapshot within the same namespace and preserves the published assistant ID.
The public lookup can read the original source after the branch advances.
Root annotations remain valid when numerical publication moves revision0to1.

Integrated execution on llm-60/Python3.12.3 passed155 CPU/filesystem controls,
56 serving controls and G051; no failures, skips or xfails. This includes the
14 Store publication tests and one added actual-Store/numerical-publisher seam
test using synthetic native bytes. It verifies that a real checkpoint and
presentation receipt remain readable after Store restart. It does not qualify
GPU inference or a model-tokenizer/display correspondence. Adjacent machine
profiles record executed IDs and source hashes.

Luna's manually written compact publication summary had its UTF-8 BOM removed
and CRLF normalized for machine reading. No factual fields were changed; raw
remote logs and executed verifier profiles were left intact. Its recorded G0
was before the last lookup-shape edit; the new integrated G0 result covers the
actual integrated source instead. Coordinator presentation binding and HTTP/UI
projection remain pending. No GPU job or chat server was started.
