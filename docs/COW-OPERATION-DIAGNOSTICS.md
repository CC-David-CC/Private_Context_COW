# Inspect a live COW operation failure

An authenticated operator can query `GET /cow/operation-diagnostic?operation=ID`
against the coordinator that accepted the original operation. This opt-in query
returns the ordinary process-local operation status and, when its failed future
has settled, a bounded exception type and message. It never dispatches model work
or changes the branch. Error messages may contain private paths or task details;
keep them in the private incident record.

`diagnostic_scope: coordinator-process-only` is deliberate: this information is
lost when the coordinator stops. A live failure can occur after publication has
already committed. It is not a durable failure receipt or proof of abort. Use
the existing `/cow/operation` and BW reconciliation path to check for a durable
original result. Preserve an unresolved operation and its identity if no durable
result can be established. Publication recovery, when offered, is separate from
restarting model computation.

The existing operation-status endpoint and its response remain unchanged. The
diagnostic endpoint cannot reconstruct the error for operations whose coordinator
has already stopped. Durable failure and crash-window recovery remain future work.
