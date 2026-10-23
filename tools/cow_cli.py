"""Small standard-library terminal client for the Context COW HTTP API.

This client connects to an already-running service and never starts a model.
It does not store canonical conversation history or automatically retry
mutations. Its private request journal retains each mutation's exact POST body
for explicit replay; that body can include submitted prompts or code.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import stat
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


_SNAPSHOT_ID = re.compile(r'[0-9a-f]{64}\Z')
_RECOMPUTATION_NOTICE = ('Notice: saved prefix is unavailable; recomputing it from '
                         'conversation history. This operation is still running.')


class ClientError(RuntimeError):
    pass


class ExplicitRejection(ClientError):
    """The server returned an explicit HTTP client-error response."""


# CLI invocation state only, never a server commit flag. Once a POST may have
# reached the wire, lack of a receipt cannot prove rejection. Existing exit 2
# is retained; --machine-result supplies the actual outcome category.
_post_may_have_been_sent = False


def default_state_dir() -> Path:
    """Return a per-user directory for pending client request intents."""
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA')
        if not base:
            base = str(Path.home() / 'AppData' / 'Local')
        return Path(base) / 'Strata' / 'ContextCOW' / 'requests'
    base = os.environ.get('XDG_STATE_HOME')
    return (Path(base) if base else Path.home() / '.local' / 'state') / 'strata-context-cow' / 'requests'


class RequestJournal:
    """Private client-side intent records; these are not conversation history."""
    SCHEMA = 1

    def __init__(self, directory):
        self.directory = Path(directory).expanduser()
        self.operations = self.directory / 'operations'

    def _prepare_dirs(self):
        for path in (self.directory, self.operations):
            if path.exists() and path.is_symlink():
                raise ClientError('request journal directory may not be a symlink')
            try:
                path.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not path.is_dir():
                    raise ClientError('request journal path is not a directory')
                if os.name != 'nt':
                    os.chmod(path, 0o700)
            except OSError as exc:
                raise ClientError(f'cannot prepare request journal: {exc}') from None

    def _path(self, op):
        return self.operations / f'{op}.json'

    @staticmethod
    def _intent_digest(op, url, path, body_json):
        binding = json.dumps([op,url,path,body_json],ensure_ascii=False,
                             separators=(',',':')).encode('utf-8')
        return hashlib.sha256(binding).hexdigest()

    @staticmethod
    def _record_digest(record):
        bound = {key:value for key,value in record.items() if key != 'record_sha256'}
        payload = json.dumps(bound,ensure_ascii=False,sort_keys=True,
                             separators=(',',':')).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _read(self, op):
        path = self._path(op)
        if path.is_symlink():
            raise ClientError('request journal record may not be a symlink')
        try:
            raw = path.read_bytes()
            if os.name != 'nt':
                os.chmod(path, 0o600)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ClientError(f'cannot read request journal: {exc}') from None
        try:
            record = json.loads(raw.decode('utf-8'))
        except (UnicodeError, ValueError):
            raise ClientError('request journal record is corrupt') from None
        required = {'schema','operation_id','url','path','body_json','intent_sha256','record_sha256','state'}
        allowed_paths = {'/cow/namespace','/cow/reseed','/cow/text/context','/cow/text/prepare','/cow/text/ask',
                         '/cow/text/answer','/cow/text/resume','/cow/text/append-user','/cow/text/retry',
                         '/cow/fork','/cow/rewind','/cow/evaluate','/cow/evaluation/adopt',
                         '/cow/training/lessons','/cow/training/jobs'}
        if (not isinstance(record, dict) or set(record) not in (
                required, required | {'receipt'}, required | {'receipt', 'answer'})
                or record.get('schema') != self.SCHEMA or record.get('operation_id') != op
                or not isinstance(record.get('url'), str) or not isinstance(record.get('path'), str)
                or record['path'] not in allowed_paths
                or not isinstance(record.get('body_json'), str)
                or record.get('state') not in ('pending','complete')):
            raise ClientError('request journal record is corrupt or has an unsupported schema')
        try:
            body = json.loads(record['body_json'])
        except ValueError:
            raise ClientError('request journal request body is corrupt') from None
        if not isinstance(body, dict) or body.get('operation_id') != op:
            raise ClientError('request journal operation binding is corrupt')
        if (not isinstance(record.get('intent_sha256'),str)
                or record['intent_sha256'] != self._intent_digest(op,record['url'],record['path'],record['body_json'])):
            raise ClientError('request journal intent checksum does not match')
        if (not isinstance(record.get('record_sha256'),str)
                or record['record_sha256'] != self._record_digest(record)):
            raise ClientError('request journal record checksum does not match')
        if record['state'] == 'complete':
            if 'receipt' not in record or not isinstance(record['receipt'], dict):
                raise ClientError('completed request journal lacks its receipt')
            if 'answer' in record and not _valid_canonical_answer(record['answer']):
                raise ClientError('completed request journal has an invalid canonical answer')
        elif 'receipt' in record:
            raise ClientError('pending request journal unexpectedly contains a receipt')
        elif 'answer' in record:
            raise ClientError('pending request journal unexpectedly contains an answer')
        return record

    def get(self, op):
        self._prepare_dirs()
        return self._read(op)

    def _atomic_write(self, path, record, *, create_only=False):
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
        fd = None
        temp_path = None
        try:
            fd, temp_name = tempfile.mkstemp(prefix='.intent-', suffix='.tmp', dir=self.operations)
            temp_path = Path(temp_name)
            if os.name != 'nt':
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                fd = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if create_only:
                # link() gives atomic no-replace semantics for a new operation ID.
                os.link(temp_path, path)
                temp_path.unlink()
            else:
                os.replace(temp_path, path)
            if os.name != 'nt':
                dfd = os.open(self.operations, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
        except FileExistsError:
            raise ClientError('operation ID already has a journal entry; use retry-request to replay it') from None
        except OSError as exc:
            raise ClientError(f'cannot persist request journal before mutation: {exc}') from None
        finally:
            if fd is not None:
                os.close(fd)
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def begin(self, op, url, path, raw_body):
        self._prepare_dirs()
        if self._read(op) is not None:
            raise ClientError('operation ID already has a journal entry; use retry-request to replay it')
        try:
            body = json.loads(raw_body.decode('utf-8'))
        except (UnicodeError, ValueError):
            raise ClientError('cannot journal an invalid request body') from None
        if not isinstance(body, dict) or body.get('operation_id') != op:
            raise ClientError('request body does not bind the selected operation ID')
        record = {'schema':self.SCHEMA,'operation_id':op,'url':url,'path':path,
                  'body_json':raw_body.decode('utf-8'),'state':'pending'}
        record['intent_sha256'] = self._intent_digest(op,url,path,record['body_json'])
        record['record_sha256'] = self._record_digest(record)
        self._atomic_write(self._path(op), record, create_only=True)
        return record

    def complete(self, record, receipt, *, answer=None):
        if not isinstance(receipt, dict):
            raise ClientError('server acknowledgement lacks a validated receipt')
        if answer is not None and not _valid_canonical_answer(answer):
            raise ClientError('server acknowledgement lacks a canonical answer object')
        path = self._path(record['operation_id'])
        current = self._read(record['operation_id'])
        if current != record:
            raise ClientError('request journal changed while the operation was in flight')
        completed = dict(record, state='complete', receipt=receipt)
        if answer is not None:
            completed['answer'] = answer
        completed['record_sha256'] = self._record_digest(completed)
        self._atomic_write(path, completed)
        return completed


_last_operation_id = None


def operation_id(value: str | None) -> str:
    result = ('op-' + uuid.uuid4().hex) if value is None else value
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', result):
        raise ClientError('operation ID must start with a letter and contain at most 64 letters, digits, _ or -')
    return result


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a nonnegative integer') from None
    if result < 0 or result > 0xFFFFFFFFFFFFFFFF:
        raise argparse.ArgumentTypeError('must be an unsigned 64-bit integer')
    return result


def probability(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a finite number in (0, 1]') from None
    if not math.isfinite(result) or not 0 < result <= 1:
        raise argparse.ArgumentTypeError('must be a finite number in (0, 1]')
    return result


def positive_timeout(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a finite positive number') from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return result


def _validate_receipt(receipt, *, expected_branch=None, expected_operation=None,
                      expected_revision=None, expected_snapshot=None, answer=None):
    if not isinstance(receipt, dict):
        raise ClientError('complete event lacks a durable receipt object')
    branch = receipt.get('branch')
    if not isinstance(branch, dict):
        raise ClientError('receipt branch must be a structured branch head')
    name, revision, tip, root = (branch.get('name'), branch.get('revision'),
                                 branch.get('tip'), branch.get('root'))
    snapshot = receipt.get('snapshot')
    if (not isinstance(name, str) or type(revision) is not int or revision < 0
            or not isinstance(tip, str) or not isinstance(root, str) or not isinstance(snapshot, str)):
        raise ClientError('receipt has an invalid branch, revision, root, tip, or snapshot')
    if tip != snapshot:
        raise ClientError('receipt snapshot does not match the committed branch tip')
    if expected_branch is not None and name != expected_branch:
        raise ClientError('receipt is for a different branch than the request')
    if expected_revision is not None and revision != expected_revision:
        raise ClientError('receipt branch revision does not match the request transition')
    if expected_snapshot is not None and snapshot != expected_snapshot:
        raise ClientError('receipt snapshot does not match the requested checkpoint')
    if (expected_operation is not None and 'operation_id' in receipt
            and receipt['operation_id'] != expected_operation):
        raise ClientError('receipt is bound to a different operation ID')
    if isinstance(answer, dict) and 'snapshot' in answer and answer['snapshot'] != snapshot:
        raise ClientError('answer snapshot does not match the durable receipt')
    return receipt


def safe_text(text: str) -> str:
    """Keep Unicode while preventing terminal escape/control-sequence injection."""
    output = []
    for char in text:
        code = ord(char)
        if char in '\n\r\t' or code >= 0x20 and not (0x7f <= code <= 0x9f):
            output.append(char)
        else:
            output.append(f'\\u{code:04x}' if code <= 0xffff else f'\\U{code:08x}')
    return ''.join(output)


class CowClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30):
        try:
            parts = urlsplit(base_url)
            host = parts.hostname
            _ = parts.port
        except ValueError:
            host = None
            parts = None
        if (parts is None or parts.scheme not in ('http','https') or not host
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or parts.path not in ('','/')):
            raise ClientError('URL must be a service origin without credentials, path, query or fragment')
        if parts.scheme == 'http' and host.lower() not in ('127.0.0.1','localhost','::1'):
            raise ClientError('remote service URLs must use HTTPS; use an SSH tunnel for HTTP')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ClientError('timeout must be a finite positive number of seconds')
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method: str, path: str, body=None, *, extra_headers=None):
        global _post_may_have_been_sent
        raw = None if body is None else (body if isinstance(body, bytes)
                                          else json.dumps(body, ensure_ascii=False).encode('utf-8'))
        headers = {'Accept': 'application/json'}
        if raw is not None:
            headers['Content-Type'] = 'application/json; charset=utf-8'
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        if extra_headers is not None:headers.update(extra_headers)
        req = Request(self.base_url + path, data=raw, headers=headers, method=method)
        if method == 'POST':
            _post_may_have_been_sent = True
        try:
            with urlopen(req, timeout=self.timeout) as response:
                data = response.read()
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')
            try:
                value = json.loads(detail)
                detail = value.get('error', detail) if isinstance(value, dict) else detail
            except ValueError:
                pass
            error = ExplicitRejection if 400 <= exc.code < 500 else ClientError
            raise error(f'HTTP {exc.code}: {detail}') from None
        except (URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            raise ClientError(f'transport failed; inspect operation status before retrying: {exc}') from None
        try:
            return json.loads(data.decode('utf-8'))
        except (UnicodeError, ValueError):
            raise ClientError('server returned invalid JSON') from None

    def upload_image(self,namespace_id,original):
        if not self.api_key:raise ClientError('image uploads require COW_API_KEY')
        if type(namespace_id) is not str or re.fullmatch('[0-9a-f]{32}',namespace_id) is None:
            raise ClientError('invalid image namespace')
        if type(original) is not bytes or not 0<len(original)<=16*1024**2:
            raise ClientError('image must contain 1 byte to 16 MiB')
        digest=hashlib.sha256(original).hexdigest()
        receipt=self.request('POST','/cow/images?'+urlencode({'namespace':namespace_id}),original,
            extra_headers={'Content-Type':'application/octet-stream','X-COW-Content-SHA256':digest})
        if (not isinstance(receipt,dict) or receipt.get('namespace_id')!=namespace_id
                or receipt.get('image_id')!=digest or type(receipt.get('bytes')) is not int
                or receipt['bytes']!=len(original) or type(receipt.get('expires_ns')) is not int
                or receipt['expires_ns']<=0):
            raise ClientError('image upload acknowledgement differs from the namespace or uploaded bytes')
        return receipt

    def stream(self, path: str, body: dict, on_provisional):
        global _post_may_have_been_sent
        if isinstance(body, bytes):
            raw = body
            try:
                bound_body = json.loads(raw.decode('utf-8'))
            except (UnicodeError, ValueError):
                raise ClientError('client request body is not valid UTF-8 JSON') from None
        else:
            bound_body = body
            raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
        headers = {'Accept': 'application/x-ndjson', 'Content-Type': 'application/json; charset=utf-8'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        req = Request(self.base_url + path, data=raw, headers=headers, method='POST')
        _post_may_have_been_sent = True
        try:
            response = urlopen(req, timeout=self.timeout)
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')
            try:
                value = json.loads(detail)
                detail = value.get('error', detail) if isinstance(value, dict) else detail
            except ValueError:
                pass
            error = ExplicitRejection if 400 <= exc.code < 500 else ClientError
            raise error(f'HTTP {exc.code}: {detail}') from None
        except (URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            raise ClientError(f'transport failed; inspect operation status before retrying: {exc}') from None
        complete = None
        recomputation = None
        recomputation_notice_printed = False
        try:
            with response:
                try:
                    for raw_line in response:
                        try:
                            event = json.loads(raw_line.decode('utf-8'))
                        except (UnicodeError, ValueError):
                            raise ClientError('server sent malformed NDJSON') from None
                        if not isinstance(event, dict) or not isinstance(event.get('event'), str):
                            raise ClientError('server sent an invalid event')
                        if complete is not None:
                            raise ClientError('server sent data after the complete receipt')
                        if event['event'] == 'error':
                            raise ClientError('request did not complete durably: ' + str(event.get('error', 'server error')))
                        if event['event'] == 'failure':
                            raise ClientError('request failed before durable publication: ' + str(event.get('error', 'server failure')))
                        if event['event'] == 'complete':
                            if not isinstance(event.get('answer'), dict):
                                raise ClientError('complete event lacks answer or durable receipt')
                            _validate_receipt(event.get('receipt'),
                                expected_branch=bound_body.get('branch'),
                                expected_operation=bound_body.get('operation_id'), answer=event['answer'])
                            complete = event
                        elif event['event'] == 'text':
                            text = event.get('text')
                            if not isinstance(text, str):
                                raise ClientError('text event has no text string')
                            on_provisional(safe_text(text))
                        elif event['event'] == 'source_recomputation_started':
                            source = event.get('source_snapshot')
                            count = event.get('history_tokens')
                            if (not isinstance(source, str) or not _SNAPSHOT_ID.fullmatch(source)
                                    or type(count) is not int or count < 0):
                                raise ClientError('server sent invalid source recomputation progress')
                            current = (source, count)
                            if recomputation is not None and recomputation != current:
                                raise ClientError('server sent inconsistent source recomputation progress')
                            recomputation = current
                            if not recomputation_notice_printed:
                                print(_RECOMPUTATION_NOTICE, file=sys.stderr, flush=True)
                                recomputation_notice_printed = True
                        elif event['event'] == 'source_recomputed':
                            source = event.get('source_snapshot')
                            history = event.get('prefix_history_tokens')
                            forward = event.get('prefix_forward_tokens')
                            consumed = event.get('prefix_consumed_tokens')
                            if (recomputation is None or source != recomputation[0]
                                    or type(history) is not int or history != recomputation[1]
                                    or type(forward) is not int or forward != history
                                    or type(consumed) is not int or consumed != history):
                                raise ClientError('server sent invalid source recomputation completion')
                        elif event['event'] == 'source_recomputation_failed':
                            source = event.get('source_snapshot')
                            if recomputation is None or source != recomputation[0]:
                                raise ClientError('server sent invalid source recomputation failure')
                            raise ClientError('server reported source recomputation failed; no durable completion is claimed')
                        elif event['event'] in ('token', 'turn_complete'):
                            continue
                        else:
                            # Keep forward compatibility for nonterminal progress events.
                            continue
                except (http.client.HTTPException, TimeoutError, OSError) as exc:
                    raise ClientError(f'stream transport failed before a clean end: {exc}') from None
        finally:
            response.close()
        if complete is None:
            raise ClientError('stream ended without a complete durable receipt')
        return complete


def _print_json(value, stream=None):
    if stream is None:
        stream = sys.stdout
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), file=stream)


def _print_receipt(value):
    print('durable receipt:', file=sys.stderr)
    _print_json(value, stream=sys.stderr)


def _valid_canonical_answer(answer):
    """Accept exact completion answers, validating text only when it is present."""
    return (isinstance(answer, dict)
            and ('text' not in answer or isinstance(answer['text'], str)))


def _print_machine_result(answer, receipt):
    """Emit the exact complete-event answer and receipt as one parseable record."""
    if not _valid_canonical_answer(answer) or not isinstance(receipt, dict):
        raise ClientError('machine result lacks an exact answer or durable receipt')
    value = {'schema': 1, 'answer': answer, 'receipt': receipt}
    sys.stdout.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(',', ':'), allow_nan=False) + '\n')
    sys.stdout.flush()


def _emit_stream_completion(result, *, machine_result=False):
    if machine_result:
        _print_machine_result(result.get('answer'), result.get('receipt'))
    else:
        print(file=sys.stdout, flush=True)
        _print_receipt(result['receipt'])


def _mutating_id(args):
    global _last_operation_id
    op = operation_id(getattr(args, 'operation_id', None))
    _last_operation_id = op
    return op


def _mutation_binding(path, body):
    """Derive receipt constraints from the saved original request, never a fresh head."""
    if path == '/cow/evaluate':
        return None, None, body.get('snapshot_id')
    if path == '/cow/evaluation/adopt':
        revision = body.get('expected_revision')
        return body.get('branch'), revision + 1 if type(revision) is int else None, None
    branch = body.get('name') if path == '/cow/fork' else body.get('branch')
    snapshot = body.get('snapshot') if path in ('/cow/fork','/cow/rewind') else None
    revision = None
    if path == '/cow/fork':
        revision = 0
    elif path in ('/cow/rewind','/cow/reseed') and type(body.get('revision')) is int:
        revision = body['revision'] + 1
    elif path in ('/cow/text/ask','/cow/text/answer','/cow/text/resume','/cow/text/append-user','/cow/text/retry'):
        if type(body.get('revision')) is int:
            revision = body['revision'] + 1
    return branch, revision, snapshot


def _validate_mutation_ack(path, value, *, branch=None, operation=None, revision=None,
                           snapshot=None, streamed=False,training_body=None):
    if path in ('/cow/training/lessons','/cow/training/jobs'):
        if training_body is None:raise ClientError('Training validation requires original request bytes')
        return _training_ack(path,training_body,value)
    if streamed:
        receipt = value.get('receipt') if isinstance(value,dict) and isinstance(value.get('receipt'),dict) else value
        if not isinstance(receipt, dict):
            raise ClientError('server acknowledgement lacks a durable receipt')
        _validate_receipt(receipt, expected_branch=branch,
                          expected_operation=operation, expected_revision=revision,
                          expected_snapshot=snapshot,
                          answer=value.get('answer') if isinstance(value,dict) else None)
        return receipt
    if path == '/cow/namespace':
        if not isinstance(value, dict) or not isinstance(value.get('namespace_id'), str) or not value['namespace_id']:
            raise ClientError('namespace acknowledgement is missing its namespace ID')
        return value
    if path == '/cow/evaluate':
        record = value.get('record') if isinstance(value,dict) else None
        identity = value.get('evaluation_id') if isinstance(value,dict) else None
        try:
            record_id = hashlib.sha256((json.dumps(record,sort_keys=True,separators=(',',':'),
                ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')).hexdigest()
        except (TypeError,ValueError,UnicodeError):
            record_id = None
        if (not isinstance(value,dict) or value.get('operation_id') != operation
                or not isinstance(identity,str) or not _SNAPSHOT_ID.fullmatch(identity)
                or not isinstance(record,dict) or record.get('snapshot_id') != snapshot
                or record.get('kind') != 'evaluation' or record.get('schema') != 1
                or record_id != identity):
            raise ClientError('evaluation acknowledgement is not bound to its operation and snapshot')
        return value
    if path == '/cow/evaluation/adopt':
        head = value.get('branch') if isinstance(value,dict) else None
        target = value.get('snapshot_id') if isinstance(value,dict) else None
        identity = value.get('evaluation_id') if isinstance(value,dict) else None
        if (not isinstance(value,dict) or value.get('operation_id') != operation
                or not isinstance(identity,str) or not _SNAPSHOT_ID.fullmatch(identity)
                or not isinstance(target,str) or not _SNAPSHOT_ID.fullmatch(target)
                or not isinstance(head,dict) or head.get('name') != branch
                or type(head.get('revision')) is not int or head.get('revision') != revision
                or head.get('tip') != target or not isinstance(head.get('root'),str)):
            raise ClientError('adoption acknowledgement is not bound to its CAS transition')
        return value
    if path in ('/cow/fork','/cow/rewind','/cow/reseed'):
        head = value.get('branch') if isinstance(value, dict) and isinstance(value.get('branch'), dict) else value
        if (not isinstance(head, dict) or not isinstance(head.get('name'), str)
                or type(head.get('revision')) is not int or head['revision'] < 0
                or not isinstance(head.get('tip'), str) or not isinstance(head.get('root'), str)):
            raise ClientError('branch mutation acknowledgement is not a valid branch head')
        if branch is not None and head['name'] != branch:
            raise ClientError('branch mutation acknowledgement names a different branch')
        if revision is not None and head['revision'] != revision:
            raise ClientError('branch mutation acknowledgement has the wrong revision')
        if snapshot is not None and head['tip'] != snapshot:
            raise ClientError('branch mutation acknowledgement has the wrong snapshot')
        return value
    raise ClientError('no acknowledgement validator exists for this mutation')


def _canonical_journal_answer(result, *, streamed):
    if not streamed or not isinstance(result, dict):
        return None
    answer = result.get('answer')
    if _valid_canonical_answer(answer):
        return answer
    if answer is not None:
        raise ClientError('server acknowledgement has an invalid canonical answer object')
    return None


def _post_mutation(client, journal, op, path, body, *, streamed, expected_branch=None, on_provisional=None):
    raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
    record = journal.begin(op, client.base_url, path, raw)
    print(f'operation_id={op}', file=sys.stderr, flush=True)
    if streamed:
        result = client.stream(path, raw, on_provisional or (lambda _text: None))
    else:
        result = client.request('POST', path, raw)
    bound_branch,bound_revision,bound_snapshot = _mutation_binding(path,body)
    receipt = _validate_mutation_ack(path, result,
        branch=expected_branch if expected_branch is not None else bound_branch,
        operation=op, revision=bound_revision, snapshot=bound_snapshot, streamed=streamed,training_body=body)
    journal.complete(record, receipt, answer=_canonical_journal_answer(result, streamed=streamed))
    return result


def _retry_request(client, journal, op, on_provisional, *, machine_result=False):
    record = journal.get(op)
    if record is None:
        raise ClientError('no request journal entry exists for this operation ID')
    if record['url'] != client.base_url:
        raise ClientError('request journal URL differs; specify the original service URL')
    if record['state'] == 'complete':
        body = json.loads(record['body_json'])
        branch,revision,snapshot = _mutation_binding(record['path'],body)
        _validate_mutation_ack(record['path'],record['receipt'],branch=branch,
            operation=op,revision=revision,snapshot=snapshot,streamed=body.get('stream') is True,training_body=body)
        if machine_result and body.get('stream') is True:
            answer = record.get('answer')
            if answer is None:
                raise ClientError('completed request has no stored canonical answer for machine output')
            _print_machine_result(answer, record['receipt'])
        else:
            _print_receipt(record['receipt'])
        return
    raw = record['body_json'].encode('utf-8')
    body = json.loads(record['body_json'])
    print(f'operation_id={op}', file=sys.stderr, flush=True)
    streamed = body.get('stream') is True
    if streamed:
        result = client.stream(record['path'], raw, on_provisional)
    else:
        result = client.request('POST', record['path'], raw)
    branch,revision,snapshot = _mutation_binding(record['path'],body)
    receipt = _validate_mutation_ack(record['path'], result,
        branch=branch, operation=op, revision=revision,snapshot=snapshot,streamed=streamed,training_body=body)
    journal.complete(record, receipt, answer=_canonical_journal_answer(result, streamed=streamed))
    if streamed:
        _emit_stream_completion(result, machine_result=machine_result)
    else:
        _print_receipt(receipt)


def _head(client, branch):
    return client.request('GET', '/cow/head?' + urlencode({'branch': branch}))


def _messages_file(path):
    try:
        content = Path(path).read_text(encoding='utf-8-sig')
    except OSError as exc:
        raise ClientError(f'cannot read context file: {exc}') from None
    except UnicodeError:
        raise ClientError('context file is not valid UTF-8') from None
    if not content.strip():
        raise ClientError('context file is empty')
    return [{'role': 'system', 'content': content}]


def _structured_messages_file(path):
    """Explicit text-message input; never guess structure from a plain file."""
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate message field')
            result[key] = value
        return result
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError('messages file exceeds 1 MiB')
        messages = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique_fields)
        if not isinstance(messages, list) or not 0 < len(messages) <= 1024:
            raise ValueError('messages must be a nonempty list of at most 1024 entries')
        for item in messages:
            if (not isinstance(item, dict) or set(item) != {'role', 'content'}
                    or item['role'] not in ('system', 'user', 'assistant')
                    or type(item['content']) is not str):
                raise ValueError('messages require only a supported role and plain text content')
        return messages
    except (OSError, UnicodeError, ValueError) as error:
        raise ClientError('cannot read structured messages: ' + str(error)) from None


def _image_bytes(path):
    try:
        target=Path(path).expanduser()
        info=target.stat()
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=16*1024**2:
            raise ClientError('image must be a regular file containing 1 byte to 16 MiB')
        with target.open('rb') as stream:raw=stream.read(16*1024**2+1)
    except OSError as error:raise ClientError('cannot read local image: '+str(error)) from None
    if not 0<len(raw)<=16*1024**2:raise ClientError('image byte length changed or exceeds 16 MiB')
    return raw


def _image_messages(client,args,*,namespace_id=None):
    """Resolve all local inputs before uploading; journal only immutable IDs."""
    parts_path=getattr(args,'parts_file',None);images=getattr(args,'image',None) or []
    if parts_path is not None:
        if images:raise ClientError('--parts-file cannot be combined with --image')
        try:
            target=Path(parts_path).expanduser()
            with target.open('rb') as stream:raw=stream.read(1024**2+1)
            if len(raw)>1024**2:raise ClientError('ordered parts file exceeds 1 MiB')
            parts=json.loads(raw.decode('utf-8-sig'))
        except (OSError,UnicodeError,ValueError) as error:
            raise ClientError('cannot read ordered parts: '+str(error)) from None
        parent=target.absolute().parent
    else:
        if not images:return [dict(role='user',content=args.text)]
        parts=[dict(type='image',file=path) for path in images]+[dict(type='text',text=args.text)]
        parent=Path.cwd()
    if not isinstance(parts,list) or not 0<len(parts)<=256:
        raise ClientError('ordered parts must be a nonempty list of at most 256 entries')
    count=0;resolved=[]
    # Validate the complete structure before reading or uploading any image.
    for part in parts:
        if not isinstance(part,dict):raise ClientError('invalid ordered part')
        if set(part)=={'type','text'} and part['type']=='text' and isinstance(part['text'],str):
            resolved.append(dict(part))
        elif set(part)=={'type','file'} and part['type']=='image' and isinstance(part['file'],str) and part['file']:
            count+=1;resolved.append(dict(part))
        elif (set(part)=={'type','image_id'} and part['type']=='image'
                and type(part['image_id']) is str and re.fullmatch('[0-9a-f]{64}',part['image_id'])):
            # Already-uploaded immutable bytes; possession of the digest grants
            # no authority. The server still requires this namespace's upload
            # grant, expiry and retained-original checks before model work.
            count+=1;resolved.append(dict(part))
        else:raise ClientError('parts use text, local image files or uploaded image IDs; URLs are not server inputs')
    if count>8:raise ClientError('at most eight images are supported per request')
    for part in resolved:
        if part['type']=='image' and 'file' in part:part['raw']=_image_bytes(parent/part.pop('file'))
    for part in resolved:
        if part['type']=='image' and 'raw' in part:
            namespace=namespace_id if namespace_id is not None else args.namespace_id
            part['image_id']=client.upload_image(namespace,part.pop('raw'))['image_id']
    return [dict(role='user',content=resolved)]


def sampling_settings(args):
    """Resolve optional per-generation controls before any mutation or request."""
    temperature=getattr(args,'temperature',None)
    min_p=getattr(args,'min_p',None)
    top_k=getattr(args,'top_k',None)
    top_p=getattr(args,'top_p',None)
    typical_p=getattr(args,'typical_p',None)
    dynatemp_range=getattr(args,'dynatemp_range',None)
    dynatemp_exponent=getattr(args,'dynatemp_exponent',None)
    xtc_probability=getattr(args,'xtc_probability',None)
    xtc_threshold=getattr(args,'xtc_threshold',None)
    xtc_supplied=xtc_probability is not None or xtc_threshold is not None
    mirostat=getattr(args,'mirostat',None)
    mirostat_tau=getattr(args,'mirostat_tau',None)
    mirostat_eta=getattr(args,'mirostat_eta',None)
    mirostat_supplied=any(value is not None for value in (mirostat,mirostat_tau,mirostat_eta))
    adaptive_target=getattr(args,'adaptive_target',None)
    adaptive_decay=getattr(args,'adaptive_decay',None)
    adaptive_supplied=adaptive_target is not None or adaptive_decay is not None
    dry_multiplier=getattr(args,'dry_multiplier',None)
    dry_base=getattr(args,'dry_base',None)
    dry_allowed_length=getattr(args,'dry_allowed_length',None)
    dry_penalty_last_n=getattr(args,'dry_penalty_last_n',None)
    dry_supplied=any(value is not None for value in
        (dry_multiplier,dry_base,dry_allowed_length,dry_penalty_last_n))
    dynatemp_supplied = dynatemp_range is not None or dynatemp_exponent is not None
    if (temperature is None and min_p is None and top_k is None and top_p is None
            and typical_p is None and not dynatemp_supplied and not dry_supplied
            and not xtc_supplied and not mirostat_supplied and not adaptive_supplied):
        return None
    if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
        raise ClientError('Temperature must be finite and nonnegative')
    if min_p is not None and (not math.isfinite(min_p) or not 0 <= min_p <= 1):
        raise ClientError('Min-P must be a finite number in [0, 1]')
    if top_k is not None and (type(top_k) is not int or not 0 <= top_k <= 0xFFFFFFFFFFFFFFFF):
        raise ClientError('Top-K must be an unsigned 64-bit integer')
    if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ClientError('Top-P must be a finite number in (0, 1]')
    if typical_p is not None and (not math.isfinite(typical_p) or not 0 < typical_p <= 1):
        raise ClientError('Typical-P must be a finite number in (0, 1]')
    for name,value in (('XTC probability',xtc_probability),('XTC threshold',xtc_threshold)):
        if value is not None and (type(value) not in (int,float)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ClientError(name+' must be a finite number in [0, 1]')
    if mirostat is not None and (type(mirostat) is not int or mirostat not in (0,2)):
        raise ClientError('Mirostat must be integer mode 0 or 2')
    if mirostat_tau is not None and (type(mirostat_tau) not in (int,float)
            or not math.isfinite(mirostat_tau) or mirostat_tau <= 0
            or not math.isfinite(2*mirostat_tau)):
        raise ClientError('Mirostat tau must be positive with finite initial state')
    if mirostat_eta is not None and (type(mirostat_eta) not in (int,float)
            or not math.isfinite(mirostat_eta) or mirostat_eta < 0):
        raise ClientError('Mirostat eta must be finite and nonnegative')
    if adaptive_target is not None and (type(adaptive_target) not in (int,float)
            or not math.isfinite(adaptive_target) or adaptive_target > 1):
        raise ClientError('Adaptive-P target must be finite and at most 1; negative disables it')
    if adaptive_decay is not None and (type(adaptive_decay) not in (int,float)
            or not math.isfinite(adaptive_decay) or not 0 <= adaptive_decay <= .99):
        raise ClientError('Adaptive-P decay must be finite and in [0, 0.99]')
    if dynatemp_range is not None and (
            not math.isfinite(dynatemp_range) or dynatemp_range < 0):
        raise ClientError('DynaTemp range must be finite and nonnegative')
    if dynatemp_exponent is not None and (
            not math.isfinite(dynatemp_exponent) or dynatemp_exponent <= 0):
        raise ClientError('DynaTemp exponent must be finite and positive')
    if dynatemp_supplied:
        effective_temperature = .70 if temperature is None else temperature
        effective_range = 0.0 if dynatemp_range is None else dynatemp_range
        if not math.isfinite(effective_temperature + effective_range):
            raise ClientError('Temperature plus DynaTemp range must be finite')
    if dry_multiplier is not None and (not math.isfinite(dry_multiplier) or dry_multiplier < 0):
        raise ClientError('DRY multiplier must be finite and nonnegative')
    if dry_base is not None and (not math.isfinite(dry_base) or dry_base < 1):
        raise ClientError('DRY base must be finite and at least 1')
    if dry_allowed_length is not None and (
            type(dry_allowed_length) is not int or not 0 <= dry_allowed_length <= 0xFFFFFFFF):
        raise ClientError('DRY allowed length must be an unsigned 32-bit integer')
    if dry_penalty_last_n is not None and (
            type(dry_penalty_last_n) is not int or not 0 <= dry_penalty_last_n <= 0xFFFFFFFF):
        raise ClientError('DRY penalty-last-n must be an unsigned 32-bit integer')
    if (temperature == 0 and min_p is None and top_k is None and top_p is None
            and typical_p is None and not dynatemp_supplied and not dry_supplied
            and not xtc_supplied and not mirostat_supplied and not adaptive_supplied):
        return {'temperature':0}
    explicit = {key: value for key, value in
                (('temperature', temperature), ('min_p', min_p), ('top_k', top_k),
                 ('top_p', top_p), ('typical_p', typical_p),
                 ('dynatemp_range', dynatemp_range),
                 ('dynatemp_exponent', dynatemp_exponent),
                 ('xtc_probability', xtc_probability), ('xtc_threshold', xtc_threshold),
                 ('mirostat', mirostat), ('mirostat_tau', mirostat_tau), ('mirostat_eta', mirostat_eta),
                 ('adaptive_target', adaptive_target), ('adaptive_decay', adaptive_decay),
                 ('dry_multiplier', dry_multiplier),
                 ('dry_base', dry_base), ('dry_allowed_length', dry_allowed_length),
                 ('dry_penalty_last_n', dry_penalty_last_n)) if value is not None}
    if getattr(args, 'command', None) in ('continue','retry'):
        # The active controller supplies omitted values. Sending a complete
        # preset here would silently reset saved filters on a partial override.
        return explicit
    settings=dict(temperature=.70 if temperature is None else temperature,
                  min_p=(0. if mirostat==2 else .05) if min_p is None else min_p,
                  top_k=0 if top_k is None else top_k,
                  top_p=1.0 if top_p is None else top_p)
    if typical_p is not None:
        settings['typical_p'] = typical_p
    if mirostat_supplied:
        settings.update(mirostat=0 if mirostat is None else mirostat,
                        mirostat_tau=5. if mirostat_tau is None else mirostat_tau,
                        mirostat_eta=.1 if mirostat_eta is None else mirostat_eta)
    if adaptive_supplied:
        settings.update(adaptive_target=-1. if adaptive_target is None else adaptive_target,
                        adaptive_decay=.9 if adaptive_decay is None else adaptive_decay)
    if xtc_supplied:
        settings.update(xtc_probability=0.0 if xtc_probability is None else xtc_probability,
                        xtc_threshold=0.1 if xtc_threshold is None else xtc_threshold)
    if dynatemp_supplied:
        settings.update(
            dynatemp_range=0.0 if dynatemp_range is None else dynatemp_range,
            dynatemp_exponent=1.0 if dynatemp_exponent is None else dynatemp_exponent)
    if dry_supplied:
        settings.update(
            dry_multiplier=0.0 if dry_multiplier is None else dry_multiplier,
            dry_base=1.75 if dry_base is None else dry_base,
            dry_allowed_length=2 if dry_allowed_length is None else dry_allowed_length,
            dry_penalty_last_n=256 if dry_penalty_last_n is None else dry_penalty_last_n)
    return settings


def _training_ack(path,body,value):
    field='lesson_id' if path.endswith('/lessons') else 'job_id'
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    expected=hashlib.sha256(canonical([path,body])).hexdigest()
    identity=hashlib.sha256(canonical(body['lesson'])).hexdigest() if field=='lesson_id' else body['operation_id']
    if (type(value) is not dict or set(value)!={'schema','operation_id','request_sha256',field}
            or type(value['schema']) is not int or value['schema']!=1 or value['operation_id']!=body['operation_id']
            or value['request_sha256']!=expected or value[field]!=identity):
        raise ClientError('Training acceptance does not bind the original request')
    return value

def _training_post(client,journal,path,body):
    global _last_operation_id, _post_may_have_been_sent
    operation=body['operation_id'];previous=journal.get(operation)
    _last_operation_id=operation
    raw=json.dumps(body,ensure_ascii=False).encode()
    if previous is not None:
        if (previous['url']!=client.base_url or previous['path']!=path or previous['body_json']!=raw.decode()):
            raise ClientError('Operation ID already binds different training request bytes')
        record=previous
    else:record=journal.begin(operation,client.base_url,path,raw)
    if record['state']=='complete':return _training_ack(path,body,record['receipt'])
    print('operation_id='+operation,file=sys.stderr,flush=True)
    if previous is not None:
        _post_may_have_been_sent=True
        try:value=client.request('GET','/cow/training/operations/'+operation)
        except ClientError:raise ClientError('Original training acceptance is unknown; retain this operation ID and inspect it before any new update') from None
    else:
        try:value=client.request('POST',path,body)
        except ExplicitRejection:raise
        except ClientError:
            try:value=client.request('GET','/cow/training/operations/'+operation)
            except ClientError:raise ClientError('Training acceptance is unknown; original request retained for reconciliation') from None
    value=_training_ack(path,body,value);journal.complete(record,value);return value

def _training_run(args,client,journal):
    action=args.lesson_action
    if action in ('list','show','status'):
        resource='jobs' if action=='status' else 'lessons'
        identity=getattr(args,'identity',None)
        if identity is not None and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',identity) is None:raise ClientError('Invalid training identity')
        return _print_json(client.request('GET','/cow/training/'+resource+('' if identity is None else '/'+identity)))
    op=args.operation_id
    if re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,119}',op) is None:raise ClientError('Use an explicit operation ID of at most120 characters')
    if action=='submit':
        try:lesson=json.loads(args.file.read_bytes())
        except OSError as error:raise ClientError('Cannot read lesson file: '+str(error)) from None
        body=dict(operation_id=op,lesson=lesson)
        value=_training_post(client,journal,'/cow/training/lessons',body)
        if not args.train_now:return _print_json(value)
        ids=[value['lesson_id']];op+='-train'
        mode=args.mode;batch=1;steps=1;rate=args.learning_rate;rank=args.rank
    else:
        if (args.all or args.pending) and args.lesson_ids:raise ClientError('Choose explicit lesson IDs or a queue selection')
        saved=journal.get(op)
        frozen=json.loads(saved['body_json']) if saved is not None and saved['path']=='/cow/training/jobs' else None
        ids=(frozen['lesson_ids'] if frozen is not None and not args.lesson_ids else
             [r['lesson_id'] for r in client.request('GET','/cow/training/lessons')['lessons'] if not args.pending or r['pending']]
             if args.all or args.pending else args.lesson_ids)
        if not ids:raise ClientError('Select at least one lesson')
        mode=args.mode;batch=args.batch_size;steps=args.steps if args.steps is not None else frozen['steps'] if frozen else (len(ids)+batch-1)//batch;rate=args.learning_rate;rank=args.rank
    body=dict(operation_id=op,lesson_ids=ids,mode=mode,batch_size=batch,steps=steps,learning_rate=rate,rank=rank,parent_job_id=args.parent_job_id)
    _print_json(_training_post(client,journal,'/cow/training/jobs',body))

def run(args):
    global _last_operation_id
    settings=sampling_settings(args)
    client = CowClient(args.url, os.environ.get('COW_API_KEY'), args.timeout)
    journal = RequestJournal(args.state_dir or os.environ.get('COW_CLI_STATE_DIR') or default_state_dir())
    cmd = args.command
    if cmd == 'lesson':
        return _training_run(args,client,journal)
    if cmd == 'status':
        _print_json(client.request('GET', '/cow/status'))
    elif cmd in ('contexts', 'namespaces'):
        _print_json(client.request('GET', '/cow/namespaces'))
    elif cmd == 'branches':
        _print_json(client.request('GET', '/cow/branches'))
    elif cmd == 'history':
        _print_json(client.request('GET', '/cow/workspace/history?' + urlencode({'branch': args.branch})))
    elif cmd == 'operation':
        _print_json(client.request('GET', '/cow/operation?' + urlencode({'operation': args.operation_id})))
    elif cmd == 'upload-image':
        _print_json(client.upload_image(args.namespace_id,_image_bytes(args.file)))
    elif cmd == 'evaluate':
        op = _mutating_id(args)
        try:
            specification = json.loads(Path(args.spec).read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as error:
            raise ClientError(f'cannot read evaluation specification: {error}') from None
        if not isinstance(specification,dict) or 'fixtures' in specification:
            raise ClientError('evaluation specification must be an object; pass fixture bytes separately')
        fixtures = {}
        total = 0
        for item in args.fixture:
            if '=' not in item:
                raise ClientError('fixture arguments must use NAME=PATH')
            name,path = item.split('=',1)
            if not name or name in fixtures:
                raise ClientError('fixture names must be nonempty and unique')
            try:
                raw = Path(path).read_bytes()
            except OSError as error:
                raise ClientError(f'cannot read evaluation fixture {name}: {error}') from None
            total += len(raw)
            if total > 4*1024*1024:
                raise ClientError('total evaluation fixture bytes exceed 4 MiB')
            fixtures[name] = raw.hex()
        body = dict(snapshot_id=args.snapshot_id,operation_id=op,
                    specification=specification,fixtures=fixtures)
        result = _post_mutation(client,journal,op,'/cow/evaluate',body,streamed=False)
        _print_json(result)
    elif cmd == 'inspect-evaluation':
        _print_json(client.request('GET','/cow/evaluation/'+args.evaluation_id))
    elif cmd == 'select-evaluations':
        _print_json(client.request('POST','/cow/evaluation/select',
            dict(evaluation_ids=args.evaluation_ids,mandatory=args.mandatory)))
    elif cmd == 'adopt-evaluation':
        op = _mutating_id(args)
        body = dict(branch=args.branch,expected_revision=args.revision,
                    evaluation_ids=args.evaluation_ids,mandatory=args.mandatory,
                    operation_id=op)
        if args.presentation_source_operation is not None:
            body['presentation_source_operation'] = args.presentation_source_operation
        _print_json(_post_mutation(client,journal,op,'/cow/evaluation/adopt',body,streamed=False))
    elif cmd == 'retry-request':
        op = operation_id(args.operation_id)
        _last_operation_id = op
        _retry_request(client, journal, op,
                       (lambda _text: None) if args.machine_result else
                       (lambda text: print(text, end='', flush=True)),
                       machine_result=args.machine_result)
    elif cmd == 'retry':
        source_op=operation_id(args.from_operation)
        _last_operation_id=source_op
        settings=sampling_settings(args)
        seed=getattr(args,'seed',None)
        info_query={'operation':source_op}
        if settings is not None:
            info_query['settings']=json.dumps(settings,separators=(',',':'),sort_keys=True)
        if seed is not None:
            info_query['seed']=str(seed)
        if getattr(args,'stop',None) is not None:
            info_query['text_stops']=json.dumps(args.stop,separators=(',',':'))
        info=client.request('GET','/cow/text/retry-info?'+urlencode(info_query))
        branch=args.branch or info.get('branch')
        if not isinstance(branch,str): raise ClientError('retry source has no committed branch')
        if branch!=info.get('branch') and info.get('replay') is True:
            raise ClientError('exact answer replay is bound to its original branch')
        head=_head(client,branch)
        replay=info.get('replay') is True
        op=operation_id(args.operation_id) if args.operation_id else (source_op if replay else operation_id(None))
        # If the caller interrupts a changed retry after admission, recovery
        # instructions must point at the new operation, not its source answer.
        _last_operation_id=op
        body=dict(from_operation=source_op,operation_id=op,branch=branch,
                  revision=head['revision'],stream=True)
        if settings is not None: body['settings']=settings
        if seed is not None: body['seed']=seed
        if getattr(args,'stop',None) is not None: body['text_stops']=args.stop
        if replay:
            result=client.stream('/cow/text/retry',body,
                (lambda _text: None) if args.machine_result else
                (lambda text: print(text,end='',flush=True)))
            if result.get('replayed') and isinstance(result.get('answer'),dict):
                text=result['answer'].get('text')
                if not isinstance(text,str): raise ClientError('replayed answer lacks canonical text')
                if not args.machine_result:
                    print(text,end='',flush=True)
            _emit_stream_completion(result, machine_result=args.machine_result)
        else:
            result=_post_mutation(client,journal,op,'/cow/text/retry',body,streamed=True,
                expected_branch=branch,on_provisional=(lambda _text: None) if args.machine_result else
                    (lambda text: print(text,end='',flush=True)))
            _emit_stream_completion(result, machine_result=args.machine_result)
    elif cmd == 'namespace':
        op = _mutating_id(args)
        result = _post_mutation(client, journal, op, '/cow/namespace',
            {'title': args.title, 'operation_id': op}, streamed=False)
        _print_receipt(result)
    elif cmd == 'load':
        messages = (_structured_messages_file(args.messages_file) if args.messages_file is not None
                    else _messages_file(args.file))
        op = _mutating_id(args)
        body = {
            'branch': args.branch, 'messages': messages, 'namespace_id': args.namespace_id,
            'operation_id': op, 'stream': True}
        result = _post_mutation(client, journal, op, '/cow/text/context', body, streamed=True,
            expected_branch=args.branch, on_provisional=(lambda _text: None) if args.machine_result else
                (lambda text: print(text, end='', flush=True)))
        _emit_stream_completion(result, machine_result=args.machine_result)
    elif cmd == 'reseed':
        op = _mutating_id(args)
        head = _head(client,args.branch)
        supplied_revision = getattr(args, 'revision', None)
        if supplied_revision is not None and supplied_revision != head['revision']:
            raise ClientError('branch revision changed; re-read head and decide explicitly before sending')
        body = dict(branch=args.branch,
                    revision=head['revision'] if supplied_revision is None else supplied_revision,
                    operation_id=op,seed=args.seed)
        result = _post_mutation(client,journal,op,'/cow/reseed',body,streamed=False,
                                expected_branch=args.branch)
        _print_receipt(result)
    elif cmd in ('prepare', 'context', 'ask', 'answer', 'continue', 'append-user'):
        op = _mutating_id(args)
        path = {'prepare':'/cow/text/prepare', 'context':'/cow/text/context', 'ask':'/cow/text/ask', 'answer':'/cow/text/answer',
                'continue':'/cow/text/resume', 'append-user':'/cow/text/append-user'}[cmd]
        body = {'branch': args.branch, 'operation_id': op, 'stream': True}
        if cmd in ('prepare','context'):
            if journal.get(op) is not None:
                raise ClientError('operation ID already has a journal entry; use retry-request to replay it')
            body.update(namespace_id=args.namespace_id,
                        messages=_image_messages(client,args))
        else:
            body['max_tokens'] = args.max_tokens
            head = _head(client, args.branch)
            supplied_revision = getattr(args, 'revision', None)
            if supplied_revision is not None and supplied_revision != head['revision']:
                raise ClientError('branch revision changed; re-read head and decide explicitly before sending')
            body['revision'] = head['revision'] if supplied_revision is None else supplied_revision
        if cmd in ('ask', 'append-user'):
            if getattr(args,'image',None) or getattr(args,'parts_file',None):
                if journal.get(op) is not None:
                    raise ClientError('operation ID already has a journal entry; use retry-request to replay it')
                namespace=head.get('namespace_id')
                if not isinstance(namespace,str) or re.fullmatch('[0-9a-f]{32}',namespace) is None:
                    raise ClientError('selected branch has no image-upload namespace')
                body['content']=_image_messages(client,args,namespace_id=namespace)[0]['content']
            else:body['content'] = args.text
        if settings is not None:
            body['settings']=settings
        seed=getattr(args,'seed',None)
        if seed is not None:
            body['seed']=seed
        if getattr(args,'grammar_state',None) is not None:
            body['grammar_state']=args.grammar_state
        if getattr(args,'stop',None) is not None:
            body['text_stops']=args.stop
        result = _post_mutation(client, journal, op, path, body, streamed=True,
            expected_branch=args.branch, on_provisional=(lambda _text: None) if args.machine_result else
                (lambda text: print(text, end='', flush=True)))
        _emit_stream_completion(result, machine_result=args.machine_result)
    elif cmd == 'fork':
        op = _mutating_id(args)
        source = _head(client, args.source_branch)
        snapshot = args.snapshot or source['tip']
        body = {
            'name': args.name, 'snapshot': snapshot, 'source_branch': args.source_branch,
            'operation_id': op}
        result = _post_mutation(client, journal, op, '/cow/fork', body, streamed=False,
                                expected_branch=args.name)
        _print_receipt(result)
    elif cmd == 'rewind':
        op = _mutating_id(args)
        current = _head(client, args.branch)
        if args.revision is not None and args.revision != current['revision']:
            raise ClientError('branch revision changed; re-read head and decide explicitly before retrying')
        body = {
            'branch': args.branch, 'revision': current['revision'], 'snapshot': args.snapshot,
            'operation_id': op}
        result = _post_mutation(client, journal, op, '/cow/rewind', body, streamed=False,
                                expected_branch=args.branch)
        _print_receipt(result)
    elif cmd == 'retry-publication':
        op = operation_id(args.operation_id)
        _last_operation_id = op
        existing = journal.get(op)
        if existing is not None and existing['url'] != client.base_url:
            raise ClientError('request journal URL differs; specify the original service URL')
        if existing is not None and existing['state'] == 'complete':
            _print_receipt(existing['receipt'])
            return
        print(f'operation_id={op}', file=sys.stderr, flush=True)
        result = client.request('POST', '/cow/publication/retry', {'operation_id': op})
        if not isinstance(result, dict) or result.get('event') != 'complete':
            raise ClientError('publication retry did not return a complete receipt')
        _validate_receipt(result.get('receipt'), expected_operation=op)
        if existing is not None:
            body = json.loads(existing['body_json'])
            branch,revision,snapshot = _mutation_binding(existing['path'],body)
            _validate_receipt(result['receipt'], expected_branch=branch,
                              expected_operation=op,expected_revision=revision,
                              expected_snapshot=snapshot)
            journal.complete(existing, result['receipt'])
        _print_receipt(result['receipt'])
    else:
        raise ClientError('unknown command')


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument('--url', default='http://127.0.0.1:8099', help='existing COW service URL')
    root.add_argument('--timeout', type=positive_timeout, default=1800,
                      help='socket idle timeout in seconds (default: 1800)')
    root.add_argument('--state-dir', help='private client operation journal directory (or COW_CLI_STATE_DIR)')
    root.add_argument('--machine-result', action='store_true',
                      help='emit one exact JSON answer/receipt record for streamed mutations')
    subs = root.add_subparsers(dest='command', required=True)
    lesson=subs.add_parser('lesson',help='Queue checked lessons and request native single/batch updates.')
    actions=lesson.add_subparsers(dest='lesson_action',required=True)
    actions.add_parser('list')
    for name in ('show','status'):
        child=actions.add_parser(name);child.add_argument('identity')
    child=actions.add_parser('submit');child.add_argument('file',type=Path);child.add_argument('--operation-id',required=True)
    child.add_argument('--train-now',action='store_true');child.add_argument('--parent-job-id')
    child.add_argument('--mode',choices=('lora','full'),default='lora');child.add_argument('--learning-rate',type=float,default=.0001);child.add_argument('--rank',type=positive_int,default=4)
    child=actions.add_parser('train');child.add_argument('lesson_ids',nargs='*');selection=child.add_mutually_exclusive_group();selection.add_argument('--all',action='store_true');selection.add_argument('--pending',action='store_true')
    child.add_argument('--operation-id',required=True);child.add_argument('--parent-job-id')
    child.add_argument('--mode',choices=('lora','full'),default='lora');child.add_argument('--batch-size',type=positive_int,default=4)
    child.add_argument('--steps',type=positive_int,help='Default: cover the selected lessons once.');child.add_argument('--rank',type=positive_int,default=4)
    child.add_argument('--learning-rate',type=float,default=.0001)
    subs.add_parser('status')
    subs.add_parser('contexts', aliases=['namespaces'], help='list saved COW namespaces')
    subs.add_parser('branches')
    p=subs.add_parser('history');p.add_argument('--branch',required=True)
    p=subs.add_parser('operation');p.add_argument('--operation-id',required=True)
    p=subs.add_parser('namespace');p.add_argument('--title',required=True);p.add_argument('--operation-id')
    p=subs.add_parser('load');inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--file',help='plain UTF-8 text loaded as one system message')
    inputs.add_argument('--messages-file',help='explicit JSON role/content text messages, up to 1 MiB; preserves message order')
    p.add_argument('--namespace-id',required=True)
    p.add_argument('--branch',required=True);p.add_argument('--operation-id')
    p=subs.add_parser('upload-image',help='upload a local PNG/JPEG and print its immutable identity')
    p.add_argument('--file',required=True);p.add_argument('--namespace-id',required=True)
    for name in ('prepare','context'):
        p=subs.add_parser(name);p.add_argument('--branch',required=True);p.add_argument('--namespace-id',required=True)
        inputs=p.add_mutually_exclusive_group(required=True)
        inputs.add_argument('--text');inputs.add_argument('--parts-file',help='ordered JSON text/local-image/uploaded-ID parts; local paths are relative to this file and IDs require this namespace upload grant')
        p.add_argument('--image',action='append',help='local PNG/JPEG before the text; repeat up to eight times')
        p.add_argument('--operation-id')
    p=subs.add_parser('ask');p.add_argument('--branch',required=True)
    inputs=p.add_mutually_exclusive_group(required=True);inputs.add_argument('--text');inputs.add_argument('--parts-file')
    p.add_argument('--image',action='append',help='local PNG/JPEG before the question; repeat up to eight times')
    p.add_argument('--seed',type=nonnegative_int,help='deterministic unsigned 64-bit generation seed')
    p.add_argument('--revision',type=nonnegative_int,
                   help='expected branch revision; refuse a stale head and send this exact revision for server CAS')
    sampled_commands=[p]
    grammar_commands=[p]
    p.add_argument('--max-tokens',type=positive_int,default=128);p.add_argument('--operation-id')
    p=subs.add_parser('answer');p.add_argument('--branch',required=True);p.add_argument('--max-tokens',type=positive_int,default=128);p.add_argument('--operation-id');p.add_argument('--seed',type=nonnegative_int,help='deterministic unsigned 64-bit generation seed');p.add_argument('--revision',type=nonnegative_int,help='expected branch revision for server CAS')
    sampled_commands.append(p)
    grammar_commands.append(p)
    p=subs.add_parser('continue');p.add_argument('--branch',required=True);p.add_argument('--max-tokens',type=positive_int,default=128);p.add_argument('--operation-id');p.add_argument('--revision',type=nonnegative_int,help='expected branch revision for server CAS')
    sampled_commands.append(p)
    grammar_commands.append(p)
    p=subs.add_parser('append-user');p.add_argument('--branch',required=True)
    inputs=p.add_mutually_exclusive_group(required=True);inputs.add_argument('--text');inputs.add_argument('--parts-file')
    p.add_argument('--image',action='append',help='local PNG/JPEG before the question; repeat up to eight times')
    p.add_argument('--seed',type=nonnegative_int,help='deterministic unsigned 64-bit generation seed')
    p.add_argument('--revision',type=nonnegative_int,help='expected branch revision for server CAS')
    sampled_commands.append(p)
    grammar_commands.append(p)
    p.add_argument('--max-tokens',type=positive_int,default=128);p.add_argument('--operation-id')
    p=subs.add_parser('retry',help='replay or rerun a committed answer with changed sampler settings')
    p.add_argument('--from-operation',required=True);p.add_argument('--branch')
    p.add_argument('--operation-id');p.add_argument('--seed',type=nonnegative_int,
        help='explicit deterministic stream seed for a changed retry')
    sampled_commands.append(p)
    for command in grammar_commands:
        command.add_argument('--grammar-state',help='Select a named server-owned GBNF or JSON Schema state for this generation boundary')
    for command in sampled_commands:
        command.add_argument('--stop',action='append',default=None,
                             help='text stop marker (repeatable; hidden from output after it is consumed)')
    for command in sampled_commands:
        command.add_argument('--temperature',type=float,help='sampling temperature; defaults to 0.70 when sampling is enabled')
        command.add_argument('--min-p',type=float,help='Min-P threshold; defaults to 0.05 when sampling is enabled')
        command.add_argument('--top-k',type=nonnegative_int,help='optional vocabulary top-K; 0 disables it')
        command.add_argument('--top-p',type=probability,help='optional nucleus threshold in (0, 1]')
        command.add_argument('--typical-p',type=float,
                             help='Typical sampling threshold in (0, 1]')
        command.add_argument('--xtc-probability',type=float,
                             help='XTC activation probability in [0, 1]; default 0 disables it')
        command.add_argument('--xtc-threshold',type=float,
                             help='XTC candidate probability threshold in [0, 1]; default 0.1')
        command.add_argument('--mirostat',type=int,choices=(0,2),
                             help='Mirostat mode: 0 off, 2 feedback sampling; requires other probability filters off')
        command.add_argument('--mirostat-tau',type=float,
                             help='Mirostat target surprise in bits; default 5')
        command.add_argument('--mirostat-eta',type=float,
                             help='Mirostat feedback rate; default 0.1')
        command.add_argument('--adaptive-target',type=float,
                             help='Adaptive-P target probability, at most 1; negative disables it')
        command.add_argument('--adaptive-decay',type=float,
                             help='Adaptive-P feedback decay in [0, 0.99]; default 0.9')
        command.add_argument('--dynatemp-range',type=float,
                             help='DynaTemp range; defaults to 0 when DynaTemp is enabled')
        command.add_argument('--dynatemp-exponent',type=float,
                             help='DynaTemp exponent; defaults to 1 when DynaTemp is enabled')
        command.add_argument('--dry-multiplier',type=float,
                             help='opt-in DRY repetition penalty strength; default 0')
        command.add_argument('--dry-base',type=float,
                             help='DRY repetition penalty base; default 1.75')
        command.add_argument('--dry-allowed-length',type=nonnegative_int,
                             help='DRY repeated-sequence allowance; default 2')
        command.add_argument('--dry-penalty-last-n',type=nonnegative_int,
                             help='DRY lookback token limit; default 256 (maximum 4294967295)')
    p=subs.add_parser('fork');p.add_argument('--source-branch',required=True);p.add_argument('--snapshot');p.add_argument('--name',required=True);p.add_argument('--operation-id')
    p=subs.add_parser('rewind');p.add_argument('--branch',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--revision',type=int);p.add_argument('--operation-id')
    p=subs.add_parser('reseed',help='durably change only the active generation RNG stream')
    p.add_argument('--branch',required=True);p.add_argument('--seed',type=nonnegative_int,required=True)
    p.add_argument('--revision',type=nonnegative_int,
                   help='expected branch revision; refuse a stale head and send this exact revision for server CAS')
    p.add_argument('--operation-id')
    p=subs.add_parser('retry-publication');p.add_argument('--operation-id',required=True)
    p=subs.add_parser('retry-request',help='replay the exact original pending mutation from the local journal')
    p.add_argument('--operation-id',required=True)
    p=subs.add_parser('evaluate',help='evaluate a saved snapshot in the isolated server evaluator')
    p.add_argument('--snapshot-id',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--fixture',action='append',default=[],metavar='NAME=PATH')
    p.add_argument('--operation-id')
    p=subs.add_parser('inspect-evaluation',help='read a hash-checked immutable evaluation report')
    p.add_argument('--evaluation-id',required=True)
    p=subs.add_parser('select-evaluations',help='choose a winner from server-validated reports')
    p.add_argument('--evaluation-ids',nargs='+',required=True);p.add_argument('--mandatory',nargs='+',required=True)
    p=subs.add_parser('adopt-evaluation',help='CAS-adopt the server-selected evaluation winner')
    p.add_argument('--branch',required=True);p.add_argument('--revision',type=nonnegative_int,required=True)
    p.add_argument('--evaluation-ids',nargs='+',required=True);p.add_argument('--mandatory',nargs='+',required=True)
    p.add_argument('--presentation-source-operation')
    p.add_argument('--operation-id')
    return root


def _configure_standard_streams():
    """Make piped output portable across Windows console code pages."""
    for stream, errors in ((sys.stdout, 'strict'), (sys.stderr, 'backslashreplace')):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            reconfigure(encoding='utf-8', errors=errors)


def main(argv=None):
    global _last_operation_id, _post_may_have_been_sent
    _last_operation_id = None
    _post_may_have_been_sent = False
    args = None
    _configure_standard_streams()
    try:
        args = parser().parse_args(argv)
        run(args)
        return 0
    except KeyboardInterrupt:
        print('cow-cli: interrupted; no durable completion is claimed.', file=sys.stderr)
        if _last_operation_id:
            print(f'Check this operation ID before resubmitting: {_last_operation_id}', file=sys.stderr)
            print('Use the operation command to inspect it; the status may be process-local.', file=sys.stderr)
        return 130
    except (ClientError, KeyError, TypeError, ValueError) as exc:
        print(f'cow-cli: {exc}', file=sys.stderr)
        if args is not None and args.machine_result:
            outcome = ('rejected' if isinstance(exc, ExplicitRejection) or not _post_may_have_been_sent
                       else 'indeterminate')
            _print_json(dict(schema=1, error=dict(outcome=outcome,
                operation_id=_last_operation_id, message=str(exc))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
