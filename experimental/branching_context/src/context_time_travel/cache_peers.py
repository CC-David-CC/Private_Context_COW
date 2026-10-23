"""Explicit authenticated cache peers; inventories are expiring placement hints.

Inspired by DeepInfra's inventory recovery and Dynamo/KVCR placement. Peers
provide numerical bytes only. They cannot install history or publish a branch.
"""
from dataclasses import dataclass,field
import os
from pathlib import Path
import re
import socket
import stat
import struct
import threading
import time
from urllib.error import HTTPError,URLError
from urllib.parse import urlencode,urlsplit
from urllib.request import Request,build_opener,HTTPRedirectHandler,ProxyHandler

from .cache_inventory import InventoryView,RecoveryRequired
from .store import IntegrityError,_decode

MAX_METADATA=4*1024**2


@dataclass(frozen=True)
class ConfiguredPeer:
    name: str
    url: str
    api_key: str=field(repr=False)

    def __post_init__(self):
        if type(self.name) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',self.name) is None:
            raise ValueError('Invalid cache peer name')
        if type(self.url) is not str or len(self.url)>2048 or any(ord(c)<33 or ord(c)>126 for c in self.url):
            raise ValueError('Invalid cache peer URL')
        parsed=urlsplit(self.url)
        if (parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ('','/') or parsed.query or parsed.fragment
                or '%' in parsed.netloc or (parsed.port is not None and not 0<parsed.port<65536)):
            raise ValueError('Cache peers require an HTTP(S) origin without credentials or a path')
        if (type(self.api_key) is not str or not 1<=len(self.api_key)<=512
                or any(ord(c)<33 or ord(c)>126 for c in self.api_key)):
            raise ValueError('Cache peer key must be one printable token')
        object.__setattr__(self,'url',self.url.rstrip('/'))


def _private_file(path,limit):
    path=Path(path)
    if not path.is_absolute():raise ValueError('Cache peer files require absolute paths')
    try:
        descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(descriptor,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid() or info.st_mode&0o077:
                raise ValueError('Cache peer files must be private regular files owned by this account')
            raw=stream.read(limit+1)
            if len(raw)>limit:raise ValueError('Cache peer file exceeds size limit')
            return raw
    except OSError as error:raise ValueError('Cannot open private cache peer file') from error


def load_peers(path):
    document=_decode(_private_file(path,65536))
    if (set(document)!={'schema','peers'} or type(document['schema']) is not int or document['schema']!=1
            or type(document['peers']) is not list or not 1<=len(document['peers'])<=16):
        raise ValueError('Cache peer configuration requires schema 1 and 1..16 explicit peers')
    result=[]
    for row in document['peers']:
        if type(row) is not dict or set(row)!={'name','url','api_key_file'}:
            raise ValueError('Unknown cache peer configuration fields')
        key=_private_file(row['api_key_file'],1024).decode('ascii').strip()
        peer=ConfiguredPeer(row['name'],row['url'],key)
        if any(peer.name==p.name or peer.url==p.url for p in result):raise ValueError('Duplicate cache peer')
        result.append(peer)
    return tuple(result)


class _Unavailable(Exception):pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class PeerCaches:
    def __init__(self,peers,transfers,*,fresh_seconds=5):
        self.peers=tuple(peers)
        if (not 1<=len(self.peers)<=16 or any(type(p) is not ConfiguredPeer for p in self.peers)
                or len({p.name for p in self.peers})!=len(self.peers)
                or len({p.url for p in self.peers})!=len(self.peers)
                or type(fresh_seconds) not in (int,float) or not 0<fresh_seconds<=60):
            raise ValueError('Invalid configured cache peers')
        self.transfers=transfers;self.fresh_seconds=fresh_seconds
        self._lock=threading.RLock();self._slots=threading.BoundedSemaphore(2)
        self._rows={p.name:dict(view=None,at=0,identity=None,rate=None,status='unobserved') for p in self.peers}
        self._locks={p.name:threading.Lock() for p in self.peers}
        # Ignore ambient proxy configuration and never forward credentials after
        # a redirect. Only explicitly configured origins receive these keys.
        self._opener=build_opener(ProxyHandler({}),_NoRedirect())

    def metrics(self):
        with self._lock:
            return {name:dict(status=row['status'],seconds_per_byte=row['rate'],
                inventory_fresh=row['at']>0 and time.monotonic()-row['at']<self.fresh_seconds)
                for name,row in self._rows.items()}

    def _request(self,peer,path,deadline):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise _Unavailable('Peer discovery/transfer deadline exceeded')
        try:
            return self._opener.open(Request(peer.url+path,headers={
                'Authorization':'Bearer '+peer.api_key}),timeout=min(10,remaining))
        except HTTPError as error:
            error.close()
            if error.code in (404,429,503):raise _Unavailable('Peer cache is unavailable') from None
            if error.code==409 and path.startswith('/cow/cache/inventory'):
                raise RecoveryRequired('Peer snapshot cursor expired') from None
            raise IntegrityError('Cache peer rejected authenticated transfer: HTTP '+str(error.code)) from None
        except (URLError,OSError,TimeoutError) as error:
            raise _Unavailable('Cache peer is unreachable') from error

    def _json(self,peer,path,deadline):
        with self._request(peer,path,deadline) as response:
            raw=response.read(MAX_METADATA+1)
            if len(raw)>MAX_METADATA:raise IntegrityError('Peer metadata exceeds limit')
            try:return _decode(raw)
            except (ValueError,TypeError) as error:raise IntegrityError('Invalid peer metadata') from error

    def _refresh(self,peer,identity,deadline):
        # Concurrent requests may use a fresh view; they never queue an unbounded
        # set of network inventory refreshes behind the same peer.
        if not self._locks[peer.name].acquire(blocking=False):raise _Unavailable('Peer refresh busy')
        try:
            with self._lock:
                row=self._rows[peer.name]
                if row['identity']==identity and row['at']>0 and time.monotonic()-row['at']<self.fresh_seconds:
                    return row['view'].entries
                view=row['view'] if row['identity']==identity else None
            cap=self._json(peer,'/cow/capabilities',deadline)
            if (cap.get('schema')!=1 or cap.get('service')!='context-cow'
                    or (cap.get('model_contract_id'),cap.get('execution_profile_id'))!=identity):
                raise IntegrityError('Configured peer model/execution identity differs')
            if any(cap.get('features',{}).get(name,{}).get('enabled') is not True
                   for name in ('peer_transfer','cache_inventory')):
                raise _Unavailable('Peer cache features are disabled')
            cursor={} if view is None else view.cursor
            # At most 16 pages for 16,384 locations, plus one expired-cursor
            # recovery. Repeated expiry is an unavailable peer, not an endless loop.
            restarted=False
            for _ in range(34):
                try:
                    packet=self._json(peer,'/cow/cache/inventory?'+urlencode(dict(cursor,page_size=1024)),deadline)
                    if view is None:view=InventoryView(packet.get('node'))
                    view.apply(packet)
                except RecoveryRequired:
                    if restarted:raise _Unavailable('Peer inventory repeatedly expired')
                    cursor={};view=None;restarted=True;continue
                except (ValueError,TypeError,KeyError) as error:
                    raise IntegrityError('Peer inventory validation failed') from error
                following=packet.get('next_cursor')
                if following is not None:
                    cursor=dict(cursor=following);continue
                with self._lock:
                    row.update(view=view,at=time.monotonic(),identity=identity,status='ready')
                return view.entries
            raise _Unavailable('Peer inventory exceeded recovery page bound')
        finally:self._locks[peer.name].release()

    @staticmethod
    def _read_exact(stream,size):
        result=bytearray()
        while len(result)<size:
            value=stream.read(min(1024**2,size-len(result)))
            if not value:raise IntegrityError('Incomplete peer checkpoint header')
            result.extend(value)
        return bytes(result)

    def _download(self,peer,model,snapshot,identity,deadline,cancelled):
        query=urlencode(dict(zip(('model_contract_id','execution_profile_id'),identity)))
        with self.transfers.pool.reserve(2*MAX_METADATA),self._request(
                peer,'/cow/cache/checkpoint/'+snapshot+'?'+query,deadline) as response:
            if (response.headers.get('Content-Type')!='application/x-cow-checkpoint'
                    or response.headers.get('Transfer-Encoding') is not None):
                raise IntegrityError('Unknown peer checkpoint wire format')
            try:total=int(response.headers['Content-Length'])
            except (KeyError,TypeError,ValueError):raise IntegrityError('Peer checkpoint requires an exact extent') from None
            if not 4<total<=4+MAX_METADATA+self.transfers.max_bytes:
                raise IntegrityError('Peer checkpoint extent exceeds bound')
            length=struct.unpack('!I',self._read_exact(response,4))[0]
            if not 0<length<=MAX_METADATA:raise IntegrityError('Peer manifest extent exceeds bound')
            envelope=_decode(self._read_exact(response,length))
            manifest=model.validate_checkpoint_envelope(snapshot,envelope)
            if total!=4+length+manifest['state_bytes']+4*manifest['logits_count']:
                raise IntegrityError('Peer checkpoint framing differs from manifest')
            return self.transfers.receive(model,snapshot,envelope,response,cancelled=cancelled)

    def recover(self,model,snapshot,*,cancelled=None,rebuild_seconds=None):
        if not self._slots.acquire(blocking=False):return None
        identity=(model.profile['model_contract_id'],model.profile['execution_profile_id'])
        deadline=time.monotonic()+self.transfers.timeout
        try:
            candidates=[]
            for peer in self.peers:
                if cancelled is not None and cancelled():
                    from concurrent.futures import CancelledError
                    raise CancelledError('Peer recovery cancelled')
                try:
                    entries=self._refresh(peer,identity,deadline)
                    matches=[e for e in entries if (e['model'],e['execution'])==identity
                             and e['snapshot']==snapshot and e['tier']=='disk']
                    if matches:
                        with self._lock:rate=self._rows[peer.name]['rate']
                        estimate=None if rate is None else rate*matches[0]['bytes']
                        if rebuild_seconds is None or estimate is None or estimate<=rebuild_seconds:
                            candidates.append((peer,estimate))
                except (_Unavailable,OSError,TimeoutError,socket.timeout):
                    with self._lock:self._rows[peer.name].update(at=0,status='unavailable')
            # Unknown rates retain configuration order. Measured peers are
            # ordered only when every eligible peer has a usable observation.
            if candidates and all(cost is not None for _,cost in candidates):candidates.sort(key=lambda p:p[1])
            for peer,_ in candidates:
                started=time.monotonic()
                try:receipt=self._download(peer,model,snapshot,identity,deadline,cancelled)
                except (_Unavailable,OSError,TimeoutError,socket.timeout):
                    with self._lock:self._rows[peer.name].update(at=0,status='unavailable')
                    continue
                seconds=time.monotonic()-started;nbytes=receipt['transferred_bytes']
                if nbytes:
                    sample=seconds/nbytes
                    with self._lock:
                        row=self._rows[peer.name];old=row['rate']
                        row['rate']=sample if old is None else .75*old+.25*sample
                return dict(receipt,peer=peer.name,seconds=seconds)
            return None
        finally:self._slots.release()
