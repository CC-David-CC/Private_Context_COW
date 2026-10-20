"""Join a browser revisit to actual same-process eviction, load and durable use.

This reads recorded evidence only. Numerical agreement and process-restart
qualification are separate checks; a PASS here makes neither claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def audit_revisit(report, trace):
    from serve.cow_http import validate_pool_spill_reload_trace
    if report.get('status') != 'PASS' or report.get('inference_kind') != 'real':
        raise ValueError('A passing real-inference browser report is required')
    revisit = report.get('reusable_context_report', {}).get('revisited_oldest_branch')
    if not isinstance(revisit, dict) or revisit.get('path') != '/cow/text/resume':
        raise ValueError('Expected the browser oldest-branch zero-input resume')
    request, ack = revisit.get('request'), revisit.get('client_ack')
    if not isinstance(request, dict) or not isinstance(ack, dict):
        raise ValueError('Captured request and durable client acknowledgment are required')
    for field in ('branch', 'operation_id'):
        if not isinstance(request.get(field), str) or not request[field] or ack.get(field) != request[field]:
            raise ValueError('Client acknowledgment does not match request ' + field)
    revision = request.get('revision')
    receipt = ack.get('receipt')
    if (type(revision) is not int or revision < 0 or not isinstance(receipt, dict)
            or receipt.get('branch') != request['branch']
            or type(receipt.get('revision')) is not int or receipt['revision'] != revision+1
            or not isinstance(revisit.get('new_tip'), str) or not revisit['new_tip']
            or receipt.get('snapshot') != revisit['new_tip']
            or ack.get('response_status') != 200 or ack.get('eof_consumed') is not True
            or not isinstance(revisit.get('request_id'), str) or not revisit['request_id']
            or ack.get('request_id') != revisit['request_id']):
        raise ValueError('Revisit does not bind a complete durable client acknowledgment')
    generated, counters = revisit.get('generated_tokens'), revisit.get('counter_deltas')
    if (type(generated) is not int or generated <= 0 or not isinstance(counters, dict)
            or type(counters.get('forward_tokens')) is not int
            or type(counters.get('prefill_tokens')) is not int
            or counters['forward_tokens'] != generated or counters['prefill_tokens'] != 0):
        raise ValueError('Revisit must evaluate exactly its output tokens and zero prefix tokens')
    identities = [report.get('server_model_contract_id_from_durable_receipts'),
                  report.get('server_execution_profile_id_from_durable_receipts'),
                  revisit.get('source_tip')]
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError('Missing model, profile or source identity')
    chain = validate_pool_spill_reload_trace(trace, model_contract_id=identities[0],
        execution_profile_id=identities[1], source_snapshot=identities[2],
        branch=request['branch'], operation_id=request['operation_id'])
    if chain['result_snapshot'] != revisit['new_tip']:
        raise ValueError('Trace result differs from the acknowledged durable snapshot')
    return dict(schema='cow-client-pressure-revisit-v1',
        status='PASS_REQUEST_BOUND_SPILL_RELOAD', causal_chain=chain,
        generated_tokens=generated, native_forward_tokens=counters['forward_tokens'],
        native_prefill_tokens=0, prefix_forward_tokens=0,
        numerical_audit='SEPARATE_REQUIRED', restart_audit='SEPARATE_REQUIRED',
        scope='One browser revisit loaded its previously evicted source from disk, '
              'used it and acknowledged durable completion without prefix evaluation. '
              'Freed bytes are pool accounting, not measured process RSS.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--pool-trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw_report, raw_trace = args.browser_report.read_bytes(), args.pool_trace.read_bytes()
    try:
        result = audit_revisit(json.loads(raw_report), json.loads(raw_trace))
    except (ValueError, KeyError, TypeError) as error:
        result = dict(status='FAIL_REQUEST_BOUND_SPILL_RELOAD', error=str(error))
    result.update(browser_report_sha256=hashlib.sha256(raw_report).hexdigest(),
                  pool_trace_sha256=hashlib.sha256(raw_trace).hexdigest(),
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'PASS_REQUEST_BOUND_SPILL_RELOAD' else 1


if __name__ == '__main__': raise SystemExit(main())
