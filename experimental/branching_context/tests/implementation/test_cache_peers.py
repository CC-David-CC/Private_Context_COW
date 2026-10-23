"""Authenticated CPU HTTP transfers; real native restore is qualified separately."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import socket
import threading
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import test_checkpoint_transfer as fixture
from context_time_travel.cache_inventory import InventoryJournal
from context_time_travel.cache_peers import ConfiguredPeer,PeerCaches,load_peers
from context_time_travel.store import IntegrityError
from serve.cow_http import make_server


class PeerCacheTests(unittest.TestCase):
    setUp=fixture.CheckpointTransferTests.setUp
    model=fixture.CheckpointTransferTests.model
    transport=fixture.CheckpointTransferTests.transport
    canonical=fixture.CheckpointTransferTests.canonical
    assert_released=fixture.CheckpointTransferTests.assert_released

    def serving(self,*,enabled=True):
        journal=InventoryJournal('cpu-peer')
        journal.reconcile([dict(model='a'*64,execution='b'*64,snapshot=self.snapshot,
            tier='disk',worker='shared',checksum=self.envelope['manifest']['payload_sha256'],
            bytes=len(self.payload))])
        self.journal=journal
        self.source.pool_transfer_enabled=enabled
        def export(snapshot,model_contract_id,execution_profile_id):
            if not enabled:raise ValueError('disabled')
            return self.outgoing.export(self.source,snapshot,model_contract_id,execution_profile_id)
        adapter=SimpleNamespace(
            export_checkpoint=export,cache_transfer_enabled=lambda:enabled,
            cache_inventory=lambda **kw:journal.read(**kw),
            capabilities=lambda:dict(features={name:dict(supported=True,enabled=value,qualification='experimental')
                for name,value in [('peer_transfer',enabled),('cache_inventory',True)]}))
        server=make_server(adapter,api_key='private-test-token',connection_info=dict(
            model_contract_id='a'*64,execution_profile_id='b'*64,eos_tokens=[3]))
        thread=threading.Thread(target=server.serve_forever);thread.start()
        self.addCleanup(lambda:(server.shutdown(),server.server_close(),thread.join(5)))
        url='http://127.0.0.1:'+str(server.server_port)
        # HTTP framing metadata is accounted in the same RAM pool.
        self.incoming=self.transport(self.target,ram_bytes=16*1024**2)
        return url,PeerCaches([ConfiguredPeer('source',url,'private-test-token')],self.incoming)

    def test_real_http_roundtrip_and_repeated_recovery_leave_heads_unchanged(self):
        url,peers=self.serving();before=self.canonical(self.target)
        result=peers.recover(self.target,self.snapshot)
        self.assertEqual(result['transferred_bytes'],len(self.payload))
        self.assertEqual(result['peer'],'source')
        self.assertEqual(self.target.validate_checkpoint(self.snapshot),self.envelope['manifest'])
        self.assertEqual(self.canonical(self.target),before)
        self.assertEqual(peers.recover(self.target,self.snapshot)['transferred_bytes'],0)
        self.assert_released(self.incoming);self.assert_released(self.outgoing)

    def test_transfer_requires_authentication_and_explicit_enable(self):
        url,peers=self.serving(enabled=False)
        path='/cow/cache/checkpoint/'+self.snapshot+'?model_contract_id='+('a'*64)+'&execution_profile_id='+('b'*64)
        with self.assertRaises(HTTPError) as error:urlopen(url+path)
        self.assertEqual(error.exception.code,401)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(url+path,headers={'Authorization':'Bearer private-test-token'}))
        self.assertEqual(error.exception.code,404)
        self.assertIsNone(peers.recover(self.target,self.snapshot))

    def test_wrong_peer_key_is_an_error_not_a_cache_miss(self):
        url,_=self.serving()
        peers=PeerCaches([ConfiguredPeer('wrong',url,'wrong')],self.incoming)
        with self.assertRaisesRegex(IntegrityError,'HTTP 401'):peers.recover(self.target,self.snapshot)
        self.assert_released(self.incoming)

    def test_incompatible_peer_never_supplies_cache_bytes(self):
        _,peers=self.serving();self.target.profile['execution_profile_id']='c'*64
        with self.assertRaisesRegex(IntegrityError,'model/execution identity'):peers.recover(self.target,self.snapshot)
        self.assertFalse((self.target.checkpoint_directory/(self.snapshot+'.json')).exists())

    def test_stale_inventory_and_unreachable_peer_allow_existing_rebuild_path(self):
        _,peers=self.serving()
        # Leave the inventory hint but remove the source's optional image.
        (self.source.checkpoint_directory/(self.snapshot+'.json')).unlink()
        self.assertIsNone(peers.recover(self.target,self.snapshot))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        absent=PeerCaches([ConfiguredPeer('absent',f'http://127.0.0.1:{port}','test')],self.incoming)
        self.assertIsNone(absent.recover(self.target,self.snapshot))
        self.assertEqual(self.target.store.head('main').tip,self.snapshot)

    def test_present_source_corruption_is_not_fallback(self):
        _,peers=self.serving()
        path=self.source.checkpoint_directory/(self.envelope['manifest']['payload_sha256']+'.state')
        path.write_bytes(b'bad')
        with self.assertRaisesRegex(IntegrityError,'HTTP 409'):peers.recover(self.target,self.snapshot)
        self.assertFalse((self.target.checkpoint_directory/(self.snapshot+'.json')).exists())
        self.assert_released(self.incoming);self.assert_released(self.outgoing)

    def test_missing_payload_of_present_manifest_is_integrity_failure(self):
        _,peers=self.serving()
        (self.source.checkpoint_directory/(self.envelope['manifest']['payload_sha256']+'.state')).unlink()
        with self.assertRaisesRegex(IntegrityError,'HTTP 409'):peers.recover(self.target,self.snapshot)

    def test_redirect_cannot_forward_peer_credentials(self):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                if self.path=='/stolen':captured.append(self.headers.get('Authorization'))
                self.send_response(302);self.send_header('Location','/stolen');self.end_headers()
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        self.addCleanup(lambda:(server.shutdown(),server.server_close(),thread.join(5)))
        peer=ConfiguredPeer('redirect','http://127.0.0.1:'+str(server.server_port),'private-test-token')
        with self.assertRaisesRegex(IntegrityError,'HTTP 302'):
            PeerCaches([peer],self.incoming).recover(self.target,self.snapshot)
        self.assertEqual(captured,[])

    def test_launcher_keeps_peer_features_independently_disabled_by_default(self):
        from tools.cow_server import configured_argv
        from tools.cow_server_setup import cache_peer_config
        config=dict(profile='/profile.json',store='/store',host='127.0.0.1',port=12345,
            ram_gib=100,disk_gib=100,min_available_gib=16,checkpoint_idle_seconds=1800)
        original=configured_argv(self.base,config,'private-key')
        self.assertNotIn('--cache-transfer',original);self.assertNotIn('--cache-peers',original)
        self.assertEqual(cache_peer_config(SimpleNamespace()),{})
        enabled=configured_argv(self.base,dict(config,cache_transfer=True),'private-key')
        self.assertEqual(enabled,original+['--cache-transfer'])
        with self.assertRaises(ValueError):configured_argv(self.base,dict(config,cache_transfer='yes'),'private-key')

    def test_serving_refuses_unauthenticated_peer_exports_even_on_localhost(self):
        with self.assertRaisesRegex(ValueError,'API key'):
            make_server(SimpleNamespace(cache_transfer_enabled=lambda:True))

    def test_bounded_private_configuration_and_credentials(self):
        key=self.base/'peer.key';key.write_text('test-key\n');key.chmod(0o600)
        config=self.base/'peers.json'
        row=dict(name='cache',url='http://127.0.0.1:12345',api_key_file=str(key))
        config.write_text(json.dumps(dict(schema=1,peers=[row])));config.chmod(0o600)
        peers=load_peers(config)
        self.assertEqual(len(peers),1);self.assertNotIn('test-key',repr(peers))
        key.chmod(0o644)
        with self.assertRaises(ValueError):load_peers(config)
        key.chmod(0o600)
        for changed in ([row,row],[dict(row,url='http://user:password@localhost')],[dict(row,url='file:///tmp')],
                        [dict(row,url='http://127.0.0.1/path')]):
            config.write_text(json.dumps(dict(schema=1,peers=changed)))
            with self.assertRaises(ValueError):load_peers(config)

    def test_identical_new_root_uses_peer_before_fresh_prefill(self):
        from test_root_computation_reuse import CPUBackend
        from context_time_travel.real import BranchingModel
        from context_time_travel.coordinator import ContextCoordinator
        from context_time_travel.pools import PoolConfig
        profile=dict(model_contract_id='a'*64,execution_profile_id='b'*64,
                     model_weights_sha256='c'*64,execution={'GPU':[0]})
        self.source=BranchingModel(self.base/'root-source',CPUBackend(profile),profile)
        self.snapshot=self.source.create((1,2,3))
        prepared=self.source.prepare_checkpoint(self.snapshot)
        self.source.publish_prepared_checkpoint(prepared)
        self.envelope=json.loads(prepared.envelope)
        self.payload=(self.source.checkpoint_directory/(prepared.checkpoint.checksum+'.state')).read_bytes()
        self.outgoing=self.transport(self.source)
        url,_=self.serving()
        directory=self.base/'root-target';backends={}
        def factory(gpu):
            backend=CPUBackend(profile);backends[gpu]=backend
            return BranchingModel(directory,backend,profile)
        coordinator=ContextCoordinator(directory,PoolConfig(ram_bytes=16*1024**2,disk_bytes=1024**2,
            min_available_bytes=0),factory,cache_peers=[ConfiguredPeer('source',url,'private-test-token')])
        self.addCleanup(coordinator.close);coordinator.ready(5)
        receipt=coordinator.create('new-chat',(1,2,3),'peer-root').durable.result(5)
        self.assertEqual(receipt['snapshot'],self.snapshot)
        self.assertEqual(backends[0].stats['forward_tokens'],0)
        self.assertEqual(coordinator.metrics()['cache_transfers']['received_bytes'],len(self.payload))


if __name__=='__main__':unittest.main()
