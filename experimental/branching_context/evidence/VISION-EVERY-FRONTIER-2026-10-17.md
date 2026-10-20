# VC09 tiny/medium every-generated-frontier restart: bounded PASS

Current renderer/profile0c0bc546, unchanged native binary/model. The new harness
retains and checks every generated frontier, instead of the earlier sampled
small-case audit. Native operators, batch boundaries and tolerances are unchanged.

- tiny: 128 text/control+49 image rows=177 cells; suffix32; outputs{'A': 8, 'B': 8}; 102 full vectors exact/error0;7 controllers.
  Private archive183834636B SHA834fbb10186d260c562ef9496f6fefcace1cf05a7e8cb8c3663e1f55db170802; destinationbox-git:/srv/data/git/private_context_cow.evidence/20261017-vision-full-frontiers-tiny-v1/evidence.tar.gz.
- medium: 1024 text/control+1073 image rows=2097 cells; suffix64; outputs{'A': 16, 'B': 16}; 190 full vectors exact/error0;7 controllers.
  Private archive287047453B SHA261036a0afc15501d5fd8c2fbc0de95a9b7d47858e04f402508249abd6267a76; destinationbox-git:/srv/data/git/private_context_cow.evidence/20261017-vision-full-frontiers-medium-v1/evidence.tar.gz.

Total292 full248320-float comparisons and14 complete controller comparisons.
Each size: four independently initialized uncached raw native references, three
cold prefix restores and a restart after four consumed generated tokens. Exact
work, prefix bytes/checkpoint hashes and all18 process exits audited separately.
Every restore evaluates0 saved-prefix tokens. CPU17 focused/G0 51 pass.

Private receiver verified every archive member before optional test state
retirement. Original image/embedding inputs, canonical histories, raw vectors,
commands, source/build/profile fingerprints and failures remain retained.

Next: current CLI timing paths and text control, then remaining supported
matrix. No inference job remains. Original G1-G5 unqualified; Q8 deferred.
The older8K target result remains separately scoped to profile2e5e0a02.
