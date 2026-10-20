# IP02: share already-derived pure bindings with full schedule audits

The backend expected-binding function now keys the existing bounded schedule
hash cache by independently hashed canonical bytes, consumed count and rotary
cursor. Values are immutable native/position hash pairs, not histories, logits,
KV or controller state. Every native check still queries the engine. Returned
public dictionaries are detached, and subclasses/changed encoded objects use the
unchanged uncached reference equations. Complete entry/publication audits remain.

Seven new focused tests,105 selected regressions,51G0 and4reference mutants pass.
Initial integration found one failure: the receipt cursor's existing test requires
one fresh reference binding per current append. That independent check was kept;
only backend expectations feed the shared pure hash cache. No existing tests or
assertions were changed. Initial failed output remains alongside the corrected run.

CPU benchmark models backend expected-binding, a fresh receipt cursor check and
complete publication audit. Hash cache starts empty in each repetition; warm-up1,
measured3. At64K/64steps, median total0.89059s IP01 versus0.88048s IP02, final audit
0.05006s versus0.02498s. At32K total0.45024s versus0.44928s. Total improvement is
small; the fresh reference computation remains. All history and schedule digests
match across arms/repetitions. This is not a GPU TPS or whole-request result.
The IP01 baseline package was extracted verbatim from commit cf055d8d.

The candidate remains undeployed and real-GPU qualification is NOT RUN. No
numerical max-error value, broader controller qualification or G1-G5 PASS claimed.
