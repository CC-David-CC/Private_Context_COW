import json
import unittest
import urllib.request

from serve.server import serve
from tools.web_demo import SAMPLE, sample_service


class SampleApp(unittest.TestCase):
    def test_sample_is_labelled_and_works_with_thinking_on_and_off(self):
        service=sample_service()
        self.assertEqual(service.model,'Scripted sample (no AI model)')
        server=serve(service,port=0)
        try:
            for effort in ['none','high']:
                with self.subTest(effort=effort):
                    body={'messages':[{'role':'user','content':'hello'}],
                          'reasoning_effort':effort,'max_tokens':512}
                    req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=10) as response:
                        answer=json.loads(response.read())['choices'][0]['message']['content']
                    self.assertEqual(answer,SAMPLE)
        finally:
            server.shutdown()
            server.server_close()
