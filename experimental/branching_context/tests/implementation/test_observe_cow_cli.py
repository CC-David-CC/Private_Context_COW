"""CPU controls for the standalone, transparent COW CLI event observer."""
from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPTS))

import observe_cow_cli as observer


def event_row(event, when):
    raw=(json.dumps(event,separators=(',',':'))+'\n').encode('utf-8')
    return {'arrival_monotonic_ns':when,'raw_line_b64':base64.b64encode(raw).decode('ascii')}


def trace_request(*events, start=100, headers=110, process_start=90,
                  eof=None, process_end=600, **extra):
    rows=list(events)
    if eof is None and not extra.get('read_error') and not extra.get('open_error'):
        eof=max((row.get('arrival_monotonic_ns',headers) for row in rows),default=headers)+1
    return {'request_started_monotonic_ns':start,'headers_monotonic_ns':headers,
            'events':rows,'process_started_monotonic_ns':process_start,
            'eof_monotonic_ns':eof,'process_finished_monotonic_ns':process_end,**extra}


def valid_receipt(snapshot='snap-1'):
    return {'snapshot':snapshot,'branch':{'name':'B','revision':2,'tip':snapshot,'root':'root-1'}}


def valid_events(tokens=(31,32,33)):
    receipt=valid_receipt()
    result=[event_row({'event':'text','token_id':tokens[0],'text':''},120),
            event_row({'event':'text','token_id':tokens[1],'text':'x'},200),
            event_row({'event':'turn_complete','token_id':tokens[2]},250),
            event_row({'event':'snapshot','tokens':list(tokens),'snapshot':'snap-1'},275),
            event_row({'event':'durable','receipt':receipt},300),
            event_row({'event':'complete','answer':{'tokens':list(tokens),'snapshot':'snap-1'},
                       'receipt':receipt},400)]
    return result


class FakeResponse:
    def __init__(self, lines, *, status=200, headers=None, fail_after=None):
        self.lines=list(lines); self.status=status; self.headers=headers or {'Content-Type':'application/x-ndjson'}
        self.index=0; self.closed=0; self.fail_after=fail_after; self.entered=0; self.exited=0
    def __enter__(self): self.entered+=1; return self
    def __exit__(self,*_args): self.exited+=1; self.close(); return False
    def __iter__(self): return self
    def __next__(self):
        if self.fail_after is not None and self.index==self.fail_after:
            raise OSError('injected stream read error')
        if self.index>=len(self.lines): raise StopIteration
        row=self.lines[self.index]; self.index+=1; return row
    def close(self): self.closed+=1


class FakeJsonResponse:
    status=200
    headers={'Content-Type':'application/json'}
    def __init__(self,value): self.value=json.dumps(value).encode('utf-8'); self.closed=0
    def __enter__(self): return self
    def __exit__(self,*_args): self.close(); return False
    def read(self): return self.value
    def close(self): self.closed+=1


class ObserverTests(unittest.TestCase):
    def test_proxy_passes_exact_lines_and_cli_exit_unchanged_and_only_wraps_ndjson(self):
        lines=[b'{"event":"token","token_id":5}\n',b'{"event":"text","text":""}\n']
        stream=FakeResponse(lines)
        json_response=FakeResponse([b'{"ok":true}'])
        calls=[]
        def urlopen(request,**kwargs):
            calls.append((request,kwargs))
            accept=request.get_header('Accept') or ''
            return stream if 'application/x-ndjson' in accept else json_response
        def main(argv):
            self.assertEqual(argv,['ask','--branch','B'])
            req=observer.make_request('http://127.0.0.1:8080/cow/text/ask',
                {'operation_id':'op-1','branch':'B'},'application/x-ndjson')
            response=cli.urlopen(req,timeout=3)
            with response:
                self.assertEqual(list(response),lines)
            # Ordinary JSON calls are passed to the exact original urlopen.
            ordinary=observer.make_request('http://127.0.0.1:8080/cow/head',{},'application/json')
            self.assertIs(cli.urlopen(ordinary),json_response)
            return 23
        cli=types.SimpleNamespace(urlopen=urlopen,main=main)
        with tempfile.TemporaryDirectory() as temp:
            result=observer.run_cli_with_observer(cli,['ask','--branch','B'],Path(temp)/'trace.json',
                                                  runtime_root=REPO,cli_path=REPO/'tools'/'cow_cli.py')
            saved=json.loads((Path(temp)/'trace.json').read_text(encoding='utf-8'))
        self.assertEqual(result,23)
        self.assertEqual(stream.closed,1)
        self.assertEqual(stream.entered,1); self.assertEqual(stream.exited,1)
        self.assertEqual(len(saved['requests']),1)
        request=saved['requests'][0]
        self.assertEqual(request['request_path'],'/cow/text/ask')
        self.assertEqual(request['operation_id'],'op-1'); self.assertEqual(request['branch'],'B')
        self.assertEqual([base64.b64decode(row['raw_line_b64']) for row in request['events']],lines)
        self.assertEqual(json_response.closed,0)

    def test_summary_reports_token_latency_durable_gap_and_empty_text_separately(self):
        request=trace_request(*valid_events(),start=100,headers=110,process_start=90,process_end=600)
        summary=observer.summarize_request(request)
        self.assertEqual(summary['validation']['status'],'valid')
        self.assertEqual(summary['stream_end_status'],'clean_eof')
        self.assertEqual(summary['token_count'],3)
        self.assertEqual(summary['empty_text_token_event_count'],1)
        self.assertEqual(summary['eos_event_count'],1)
        self.assertEqual([row['kind'] for row in summary['snapshots']],['snapshot','durable','complete'])
        self.assertEqual(summary['snapshots'][0]['snapshot'],'snap-1')
        metrics=summary['metrics_ns']
        self.assertEqual(metrics['request_to_first_token'],20)
        self.assertEqual(metrics['token_interval_mean'],65)
        self.assertEqual(metrics['token_interval_median'],65)
        self.assertEqual(metrics['token_interval_p95'],80)
        self.assertEqual(metrics['token_interval_max'],80)
        self.assertAlmostEqual(summary['generation_tokens_per_second'],2_000_000_000/130)
        self.assertEqual(metrics['last_token_to_durable'],50)
        self.assertEqual(metrics['request_to_durable'],200)
        self.assertEqual(metrics['process_elapsed'],510)

    def test_reducer_validates_full_monotonic_clock_chain_and_suppresses_invalid_tps(self):
        good=trace_request(*valid_events(),process_start=90,start=100,headers=110,eof=401,process_end=600)
        self.assertEqual(observer.summarize_request(good)['validation']['status'],'valid')
        bad=[]
        bad.append(trace_request(*valid_events(),process_start=101,start=100,headers=110,eof=401,process_end=600))
        bad.append(trace_request(*valid_events(),process_start=90,start=120,headers=110,eof=401,process_end=600))
        bad.append(trace_request(*valid_events(),process_start=90,start=100,headers=121,eof=401,process_end=600))
        bad.append(trace_request(*valid_events(),process_start=90,start=100,headers=110,eof=399,process_end=600))
        bad.append(trace_request(*valid_events(),process_start=90,start=100,headers=110,eof=401,process_end=400))
        for request in bad:
            with self.subTest(request=request):
                summary=observer.summarize_request(request)
                self.assertNotEqual(summary['validation']['status'],'valid')
                self.assertIsNone(summary['generation_tokens_per_second'])

    def test_snapshot_answer_shape_is_optional_but_when_present_must_match_tokens_and_order(self):
        good=trace_request(*valid_events())
        self.assertEqual(observer.summarize_request(good)['validation']['status'],'valid')
        without=[row for row in good['events'] if json.loads(base64.b64decode(row['raw_line_b64']))['event']!='snapshot']
        self.assertEqual(observer.summarize_request({**good,'events':without})['validation']['status'],'valid')
        for mutate in ('tokens','snapshot','order'):
            rows=copy_event_rows(good['events'])
            if mutate=='tokens':
                value=json.loads(base64.b64decode(rows[3]['raw_line_b64'])); value['tokens']=[1]
                rows[3]=event_row(value,275)
            elif mutate=='snapshot':
                value=json.loads(base64.b64decode(rows[3]['raw_line_b64'])); value['snapshot']='other'
                rows[3]=event_row(value,275)
            else:
                rows[3],rows[4]=rows[4],rows[3]
            summary=observer.summarize_request({**good,'events':rows})
            self.assertEqual(summary['validation']['status'],'invalid')
            self.assertIsNone(summary['generation_tokens_per_second'])

    def test_token_ids_and_structured_branch_receipt_are_strictly_validated(self):
        good=trace_request(*valid_events())
        for invalid in (-1, True):
            rows=copy_event_rows(good['events'])
            value=json.loads(base64.b64decode(rows[0]['raw_line_b64'])); value['token_id']=invalid
            rows[0]=event_row(value,120)
            self.assertEqual(observer.summarize_request({**good,'events':rows})['validation']['status'],'invalid')
        body={'branch':'B','operation_id':'op-rev','revision':7}
        request=observer.make_request('http://localhost/cow/text/ask',body,'application/x-ndjson')
        binding=observer._request_binding(request)
        self.assertEqual(binding.get('revision'),7)
        valid=trace_request(*valid_events(),branch='B',revision=1)
        self.assertEqual(observer.summarize_request(valid)['validation']['status'],'valid')
        for field,value in (('name','other'),('tip','other'),('root',''),('revision',9)):
            rows=copy_event_rows(good['events'])
            for index in (4,5):
                obj=json.loads(base64.b64decode(rows[index]['raw_line_b64']))
                obj['receipt']['branch'][field]=value
                rows[index]=event_row(obj,300 if index==4 else 400)
            bad={**good,'events':rows,'branch':'B','revision':7}
            self.assertEqual(observer.summarize_request(bad)['validation']['status'],'invalid')
        self.assertEqual(observer.summarize_request({**good,'branch':'B','revision':True})['validation']['status'],'invalid')

    def test_real_cli_entrypoint_keeps_validation_journal_and_exit_while_observing(self):
        cli,cli_path=observer.load_cli(REPO)
        op='op-observer-entrypoint'
        receipt={'snapshot':'snap-2','operation_id':op,
                 'branch':{'name':'B','revision':3,'tip':'snap-2','root':'root-1'}}
        stream=FakeResponse([json.dumps(row,separators=(',',':')).encode()+b'\n' for row in (
            {'event':'token','token_id':31}, {'event':'text','text':''},
            {'event':'token','token_id':32}, {'event':'durable','receipt':receipt},
            {'event':'complete','answer':{'tokens':[31,32],'snapshot':'snap-2'},'receipt':receipt})])
        head=FakeJsonResponse({'name':'B','revision':2,'tip':'snap-1','root':'root-1'})
        def urlopen(request,**_kwargs):
            if request.full_url.endswith('/cow/head?branch=B'):
                return head
            if request.get_header('Accept')=='application/x-ndjson':
                return stream
            self.fail('unexpected URL opened by actual CLI')
        with tempfile.TemporaryDirectory() as temp:
            trace_path=Path(temp)/'trace.json'
            args=['--url','http://127.0.0.1:8080','--state-dir',str(Path(temp)/'journal'),
                  'ask','--branch','B','--text','Say hi','--max-tokens','2','--operation-id',op]
            output=io.StringIO(); error=io.StringIO()
            with patch.object(cli,'urlopen',urlopen),contextlib.redirect_stdout(output),contextlib.redirect_stderr(error):
                code=observer.run_cli_with_observer(cli,args,trace_path,runtime_root=REPO,cli_path=cli_path)
            trace=json.loads(trace_path.read_text(encoding='utf-8'))
            journal=json.loads((Path(temp)/'journal'/'operations'/f'{op}.json').read_text(encoding='utf-8'))
        self.assertEqual(code,0)
        self.assertEqual(trace['cli_exit_code'],0)
        self.assertEqual(trace['cli_source_sha256'],observer._sha256(cli_path))
        self.assertEqual(trace['requests'][0]['request_path'],'/cow/text/ask')
        self.assertEqual(trace['requests'][0]['summary']['validation']['status'],'valid')
        self.assertEqual(trace['requests'][0]['summary']['token_ids'],[31,32])
        self.assertEqual(journal['state'],'complete')
        self.assertEqual(journal['receipt'],receipt)
        self.assertIn('Say hi',json.loads(journal['body_json'])['content'])

    def test_zero_or_one_token_has_null_interval_and_generation_metrics(self):
        for tokens in ((),(9,)):
            with self.subTest(tokens=tokens):
                receipt=valid_receipt()
                events=[event_row({'event':'token','token_id':token},100+i*100)
                        for i,token in enumerate(tokens)]
                events.extend((event_row({'event':'durable','receipt':receipt},300),
                               event_row({'event':'complete','answer':{'tokens':list(tokens),'snapshot':'snap-1'},
                                          'receipt':receipt},400)))
                summary=observer.summarize_request(trace_request(*events))
                self.assertIsNone(summary['metrics_ns']['token_interval_mean'])
                self.assertIsNone(summary['generation_tokens_per_second'])

    def test_reducer_rejects_malformed_reordered_duplicate_or_mismatched_stream(self):
        good=valid_events()
        bad_rows=[]
        malformed=copy_event_rows(good); malformed[0]['raw_line_b64']=base64.b64encode(b'{bad\n').decode(); bad_rows.append(malformed)
        reordered=copy_event_rows(good); reordered[-2],reordered[-1]=reordered[-1],reordered[-2]; bad_rows.append(reordered)
        duplicate=copy_event_rows(good); duplicate.append(copy_event_rows(good)[-1]); bad_rows.append(duplicate)
        duplicate_durable=copy_event_rows(good); duplicate_durable.insert(-1,copy_event_rows(good)[-2]); bad_rows.append(duplicate_durable)
        mismatch=copy_event_rows(good); obj=json.loads(base64.b64decode(mismatch[-1]['raw_line_b64'])); obj['receipt']['snapshot']='other'; mismatch[-1]=event_row(obj,450); bad_rows.append(mismatch)
        token_mismatch=copy_event_rows(good); obj=json.loads(base64.b64decode(token_mismatch[-1]['raw_line_b64'])); obj['answer']['tokens']=[99]; token_mismatch[-1]=event_row(obj,450); bad_rows.append(token_mismatch)
        nonmonotonic=copy_event_rows(good); nonmonotonic[1]['arrival_monotonic_ns']=49; bad_rows.append(nonmonotonic)
        for rows in bad_rows:
            with self.subTest(rows=rows):
                result=observer.summarize_request(trace_request(*rows))
                self.assertEqual(result['validation']['status'],'invalid')
                self.assertTrue(result['validation']['errors'])

    def test_streamed_error_and_truncation_are_recorded_without_cli_status_rewrite(self):
        for line,expected in ((b'{"event":"error","error":"failed"}\n','error'),
                              (b'{"event":"token","token_id":1}\n','truncated')):
            response=FakeResponse([line])
            def cli_main(_argv):
                request=observer.make_request('http://localhost/cow/text/ask',
                    {'operation_id':'op-x','branch':'B'},'application/x-ndjson')
                with cli.urlopen(request) as body:
                    for _line in body: pass
                return 2
            cli=types.SimpleNamespace(urlopen=lambda *_args,**_kwargs:response,main=cli_main)
            with tempfile.TemporaryDirectory() as temp:
                code=observer.run_cli_with_observer(cli,[],Path(temp)/'trace.json',runtime_root=REPO,
                    cli_path=REPO/'tools'/'cow_cli.py')
                trace=json.loads((Path(temp)/'trace.json').read_text(encoding='utf-8'))
            self.assertEqual(code,2)
            self.assertEqual(trace['requests'][0]['summary']['validation']['status'],expected)

    def test_read_exception_passes_through_and_is_observed_as_transport_error(self):
        response=FakeResponse([],fail_after=0)
        def cli_main(_argv):
            request=observer.make_request('http://localhost/cow/text/ask',
                {'operation_id':'op-read-error','branch':'B'},'application/x-ndjson')
            with cli.urlopen(request) as body:
                list(body)
            return 0
        cli=types.SimpleNamespace(urlopen=lambda *_args,**_kwargs:response,main=cli_main)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.json'
            with self.assertRaisesRegex(OSError,'injected stream read error'):
                observer.run_cli_with_observer(cli,[],path,runtime_root=REPO,cli_path=REPO/'tools'/'cow_cli.py')
            trace=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(trace['requests'][0]['summary']['validation']['status'],'transport_error')
        self.assertEqual(trace['requests'][0]['summary']['stream_end_status'],'read_error')
        self.assertEqual(trace['requests'][0]['read_error']['type'],'OSError')
        self.assertEqual(response.closed,1)

    def test_open_exception_is_timestamped_transport_error_without_rate(self):
        def main(_argv):
            request=observer.make_request('http://localhost/cow/text/ask',
                {'operation_id':'op-open-error','branch':'B'},'application/x-ndjson')
            try: cli.urlopen(request)
            except OSError: return 7
            self.fail('open error was swallowed')
        def fail_open(*_args,**_kwargs): raise OSError('connect failed')
        cli=types.SimpleNamespace(urlopen=fail_open,main=main)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.json'
            code=observer.run_cli_with_observer(cli,[],path,runtime_root=REPO,cli_path=REPO/'tools'/'cow_cli.py')
            request=json.loads(path.read_text(encoding='utf-8'))['requests'][0]
        self.assertEqual(code,7)
        summary=request['summary']
        self.assertEqual(summary['validation']['status'],'transport_error')
        self.assertEqual(summary['stream_end_status'],'open_error')
        self.assertIsNone(summary['generation_tokens_per_second'])
        self.assertLessEqual(request['process_started_monotonic_ns'],request['request_started_monotonic_ns'])
        self.assertLessEqual(request['request_started_monotonic_ns'],request['open_error_monotonic_ns'])
        self.assertLessEqual(request['open_error_monotonic_ns'],request['process_finished_monotonic_ns'])

    def test_read_error_after_complete_still_invalidates_transport_and_tps(self):
        request=trace_request(*valid_events(),read_error={'type':'OSError','message':'late socket error'},
                              read_error_monotonic_ns=450,eof=None)
        summary=observer.summarize_request(request)
        self.assertEqual(summary['validation']['status'],'transport_error')
        self.assertEqual(summary['stream_end_status'],'read_error')
        self.assertIsNone(summary['generation_tokens_per_second'])

    def test_observer_capture_error_cannot_be_marked_valid(self):
        request=trace_request(*valid_events(),observer_error='capture failed')
        summary=observer.summarize_request(request)
        self.assertEqual(summary['validation']['status'],'invalid')
        self.assertIsNone(summary['generation_tokens_per_second'])

    def test_trace_write_failure_does_not_rewrite_cli_exit_code(self):
        cli=types.SimpleNamespace(urlopen=lambda *_args,**_kwargs:None,main=lambda _argv:17)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.json'; path.write_text('keep',encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                code=observer.run_cli_with_observer(cli,[],path,runtime_root=REPO,
                    cli_path=REPO/'tools'/'cow_cli.py')
            self.assertEqual(path.read_text(encoding='utf-8'),'keep')
        self.assertEqual(code,17)

    def test_reducer_failure_does_not_rewrite_cli_exit_code(self):
        response=FakeResponse([b'{"event":"token","token_id":1}\n'])
        cli=types.SimpleNamespace(urlopen=lambda *_args,**_kwargs:response)
        def main(_argv):
            request=observer.make_request('http://localhost/cow/text/ask',
                {'operation_id':'op-summary-failure','branch':'B'},'application/x-ndjson')
            with cli.urlopen(request) as stream: list(stream)
            return 19
        cli.main=main
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(observer,'summarize_request',side_effect=RuntimeError('injected reducer failure')):
                with contextlib.redirect_stderr(io.StringIO()):
                    code=observer.run_cli_with_observer(cli,[],Path(temp)/'trace.json',runtime_root=REPO,
                        cli_path=REPO/'tools'/'cow_cli.py')
        self.assertEqual(code,19)
        self.assertEqual(response.closed,1)

    def test_actual_cli_stream_validation_still_rejects_duplicate_complete(self):
        path=REPO/'tools'/'cow_cli.py'
        spec=importlib.util.spec_from_file_location('_cow_cli_validation_test',path)
        cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        receipt=valid_receipt()
        complete={'event':'complete','answer':{'tokens':[1],'snapshot':'snap-1'},'receipt':receipt}
        lines=[json.dumps(x,separators=(',',':')).encode()+b'\n' for x in (
            {'event':'token','token_id':1}, {'event':'durable','receipt':receipt}, complete, complete)]
        response=FakeResponse(lines)
        monitored=observer._Observer(lambda *_args,**_kwargs:response)
        with patch.object(cli,'urlopen',monitored.urlopen):
            client=cli.CowClient('http://127.0.0.1:8080')
            with self.assertRaises(cli.ClientError):
                client.stream('/cow/text/ask',{'branch':'B','operation_id':'op-1'},lambda _text:None)
        self.assertGreaterEqual(response.closed,1)
        self.assertEqual(observer.summarize_request(monitored.requests[0])['validation']['status'],'invalid')


def copy_event_rows(rows):
    return [dict(row) for row in rows]


if __name__=='__main__':
    unittest.main()
