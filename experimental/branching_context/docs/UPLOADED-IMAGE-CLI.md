# Reuse an uploaded image through the CLI

The ordered `prepare --parts-file` input accepts text, local PNG/JPEG files, and
already-uploaded image IDs. The latter avoids another upload; the server still
requires the same namespace's upload grant and retained original. An image digest
alone does not grant access. Expired derived embeddings may be re-encoded under
the existing cache policy; absent originals are an explicit error.

Example parts file, replacing the digest with the actual upload receipt:

```json
[
  {"type":"text","text":"Inspect this page: "},
  {"type":"image","image_id":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
  {"type":"text","text":" Where is the green square?"}
]
```

Use the existing `upload-image --namespace-id ID --file LOCAL_IMAGE` first, then
`prepare --namespace-id ID --branch NAME --parts-file PARTS --operation-id OP`.
Configure the authenticated URL/API key as usual. Local paths are read by the
client and never sent as server file requests. Uploaded IDs are exact lowercase
SHA256 values. Input order is preserved and at most eight images are accepted.
URLs and malformed/mixed reference objects reject before preparation.

2026-10-17 executed evidence: three new reference tests plus affected CLI/image
checks, 29/29 PASS with a fresh client journal; G0 51/51. Real Q4 H02 used this
CLI binding for three actual screenshots, all correctly delivered/interpreted
in one attempt each. No full-logit oracle ran in H02; maximum numerical error
is unmeasured. BranchWrite's final suite passed 323/323. See its separate
`validation/H02_IMAGES_Q4_2026-10-17.md` for the complete result and limitations.

Actual frozen COW source on llm-60:
`/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/cow-image-reference-green-source-v2`.
Runtime profile a2a59a68a40016352310436cc418b4e8839d521d224837e4a69cad00bcc79633;
native SHA ae62045f75a8e11ba9d01b4da301d3bb87fe4ff342bab0ea7edc57276377824e.
Initial launcher/journal/harness failures are retained separately. Private raw
archive `/srv/data/git/private_context_cow.evidence/20261017-branchwrite-h02-v1/evidence.tar.gz`,
14,188,881 bytes, SHA256
118382c1f5006ab4a48c78a3d0e56f66a38754d3e9e581a9fab907665fd7f71e.

This binding changes no native computation, numerical tolerance or history
representation. Namespace-denial checks use real CLI/HTTP fixtures; their CPU
encoder is not GPU certification. H02 itself used the actual pinned GPU encoder.
Full VC00-VC09 and original G1-G5 remain unqualified.
