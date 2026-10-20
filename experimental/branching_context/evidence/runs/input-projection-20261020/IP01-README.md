# IP01: immutable derived input views

New history objects retain the same canonical JSON bytes, counts and SHA256 IDs.
Decoded spans/native IDs are immutable derived caches bound to the exact owned
encoded byte object. Public span reads remain detached; subclasses and changed
encoded identities use uncached reconstruction. A validated text extension appends
only exact decimal token bytes to the final canonical text span. New IDs still
validate. No prior history object chain is retained and image rows remain lazy.
Reference canonical_inputs, input binding equations, stores and snapshots remain
unchanged. The live API has not been switched to this candidate.

Eight new focused tests (red retained, then green),98 selected input/schedule/
binding CPU regressions,51 G0 reference tests,4 reference mutants killed. No
failures/skips in the final runs. Initial3 integration-test import errors were a
missing tools directory on the runner path; retained as environment failures,
then the same tests were rerun. No test assertions or tolerances were weakened.

CPU benchmark:one warm-up,three measured repetitions,20 append/proof steps at
1K/8K/32K/64K, text and mixed-image metadata. All final input digests match across
serving4b1f8fda, pause base780e824a and this candidate. At64K text,medians were
0.93022s,0.74716s and0.25524s respectively; mixed0.92619s,0.75373s,0.25717s.
This isolates metadata functions, not native GPU inference or whole-request TPS.
Raw samples and source fingerprints are in benchmark.json. No numerical max-error
measurement or real-GPU qualification has been performed for this candidate yet.
No G1-G5 qualification or physical-sharing certification is claimed.
