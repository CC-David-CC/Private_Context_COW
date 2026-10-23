"""Authenticated discovery observes runtime facts without admitting model work."""
import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from serve.cow_http import make_server


class CapabilityHTTPTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        owner=self
        class Adapter:
            def capabilities(self):
                owner.calls.append('observe')
                return dict(workers=[],features={},limits={},scheduling='fifo')
        self.server=make_server(Adapter(),api_key='secret',connection_info=dict(
            model_contract_id='a'*64,execution_profile_id='b'*64,eos_tokens=[1]))
        self.thread=threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.thread.join,5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url='http://127.0.0.1:'+str(self.server.server_port)

    def request(self,path,key='secret'):
        request=Request(self.url+path,headers={'Authorization':'Bearer '+key})
        with urlopen(request,timeout=5) as response:return json.load(response)

    def test_authentication_and_query_validation_precede_observation(self):
        for path,key,code in [('/cow/capabilities','wrong',401),('/cow/capabilities?x=1','secret',400)]:
            with self.subTest(path=path),self.assertRaises(HTTPError) as error:
                self.request(path,key)
            self.assertEqual(error.exception.code,code)
            error.exception.close()
        self.assertEqual(self.calls,[])

    def test_versioned_report_and_legacy_info_are_separate(self):
        before=self.request('/cow/info')
        value=self.request('/cow/capabilities')
        self.assertEqual(value['schema'],1)
        self.assertEqual(value['model_contract_id'],before['model_contract_id'])
        self.assertFalse(value['features']['text']['enabled'])
        self.assertEqual(self.request('/cow/info'),before)
        self.assertEqual(set(before),{'schema','service','model_contract_id','execution_profile_id','eos_tokens','capabilities'})
        self.assertEqual(self.calls,['observe'])


if __name__=='__main__':unittest.main()
