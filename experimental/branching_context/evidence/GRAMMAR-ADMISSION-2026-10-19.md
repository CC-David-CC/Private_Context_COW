# Grammar admission: bounded CPU implementation
COW grammar registration is opt-in: --grammar-store /absolute/private/directory,
with --text and --api-key. Parent directory must already exist. Root is0700,
records0600. POST /cow/grammars accepts only inline schema1/states definitions,
strict finite JSON without duplicate keys,maximum4MiB; no client file paths.
Response maps local state names to content-derived public names and records
compiler/tokenizer/engine identities. Identical registration returns the same
receipt. Capacity errors are explicit. No registered grammar grants tool rights.

Definitions are durably published before in-memory admission. Reload recompiles
and verifies the saved identities; corruption rejects before new admission.
The router retains startup grammars and resolves each coordinator request to a
single immutable engine. Adding another group cannot redefine queued work or a
saved parser. Definitions currently have bounded retention without automatic GC;
they must remain available while saved controllers refer to them.

Executed:65 CPU/HTTP checks,0failures/errors/skips,plus a strengthened duplicate-key
case with otherwise valid source. Actual files,subprocess reload,real llguidance,
and deterministic coordinator backend;NO native GPU model qualification.
Injected fsync failure is a supplemental fault test,not actual disk-full or
power-loss evidence. The earlier corruption-side-effect and caller-mutation
failures are retained in private evidence.
Client registration journaling and automatic per-project grammar selection are
NOT implemented yet. These changes are isolated from the live L06 Q8 campaign.
Raw evidence:/home/dflanag3/spatial-artifacts/cow-grammar-admission-20261019-v1.
