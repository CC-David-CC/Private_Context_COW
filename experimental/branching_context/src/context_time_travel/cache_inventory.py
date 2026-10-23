"""Recoverable, bounded location hints; never checkpoint or history authority.

Inspired by DeepInfra's kv-local-indexer and NVIDIA Dynamo/KVCR. This is an
independent implementation of snapshot recovery after missed inventory events.
"""
from collections import deque, OrderedDict
import copy
import hashlib
import json
import math
import threading
import time
import uuid


class RecoveryRequired(ValueError):
    """Discard the delta and obtain a fresh complete inventory."""


def _text(value):
    if not isinstance(value,str) or not 1<=len(value)<=128 or not value.isprintable():
        raise ValueError('Invalid inventory identity')
    return value


def _entry(value):
    fields={'model','execution','snapshot','tier','worker','checksum','bytes'}
    if type(value) is not dict or set(value)!=fields:
        raise ValueError('Unknown inventory entry layout')
    for field in fields-{'bytes'}:_text(value[field])
    if value['tier'] not in ('native','ram','disk'):
        raise ValueError('Unknown inventory tier')
    if (len(value['checksum'])!=64 or any(c not in '0123456789abcdef' for c in value['checksum'])
            or type(value['bytes']) is not int or not 0<value['bytes']<2**63):
        raise ValueError('Invalid inventory extent or checksum')
    return dict(value)


def _key(row):
    return tuple(row[k] for k in ('model','execution','snapshot','tier','worker'))


def _entries(values,limit):
    if type(values) is not list or len(values)>limit:
        raise ValueError('Inventory capacity exceeded or invalid entry collection')
    result={}
    for raw in values:
        row=_entry(raw); key=_key(row)
        if key in result:raise ValueError('Duplicate inventory location')
        result[key]=row
    return result


class InventoryJournal:
    def __init__(self,node,*,event_limit=1024,entry_limit=16384,
                 snapshot_limit=4,snapshot_ttl=60,clock=time.monotonic):
        self.node=_text(node)
        if any(type(n) is not int or n<=0 for n in (event_limit,entry_limit,snapshot_limit)):
            raise ValueError('Invalid inventory bounds')
        if (type(snapshot_ttl) not in (int,float) or not math.isfinite(snapshot_ttl)
                or snapshot_ttl<=0 or not callable(clock)):
            raise ValueError('Invalid snapshot lifetime')
        self.epoch=uuid.uuid4().hex
        self.entry_limit=entry_limit
        self._entries={}; self._events=deque(maxlen=event_limit); self._seq=0
        self._lock=threading.RLock()
        self._snapshots=OrderedDict()
        self._snapshot_limit,self._snapshot_ttl,self._clock=snapshot_limit,snapshot_ttl,clock

    def reconcile(self,entries):
        """Publish a fresh availability observation; changes are atomic."""
        next_entries=_entries(entries,self.entry_limit)
        with self._lock:
            changes=[('drop',self._entries[k]) for k in sorted(self._entries.keys()-next_entries.keys())]
            changes += [('put',v) for k,v in sorted(next_entries.items()) if self._entries.get(k)!=v]
            for action,row in changes:
                self._seq+=1
                self._events.append(dict(seq=self._seq,action=action,entry=row))
            self._entries=next_entries

    def read(self,*,epoch=None,since=None,page_size=None,cursor=None):
        if (epoch is None)!=(since is None):raise ValueError('Use both epoch and since')
        if since is not None and (type(since) is not int or since<0):
            raise ValueError('Invalid inventory cursor')
        if epoch is not None:_text(epoch)
        if page_size is not None and (type(page_size) is not int or not 1<=page_size<=1024):
            raise ValueError('Page size must be between 1 and 1024')
        if cursor is not None and (page_size is None or epoch is not None):
            raise ValueError('Page cursor requires page_size and no event cursor')
        with self._lock:
            if cursor is not None:return self._page(page_size,cursor)
            packet=dict(schema=1,node=self.node,epoch=self.epoch,seq=self._seq)
            oldest=self._events[0]['seq']-1 if self._events else self._seq
            if epoch==self.epoch and oldest<=since<=self._seq:
                packet.update(kind='delta',base=since,
                              events=[e for e in self._events if e['seq']>since])
            else:
                if page_size is not None:return self._page(page_size,None)
                packet.update(kind='snapshot',entries=[v for _,v in sorted(self._entries.items())])
            return copy.deepcopy(packet)

    def _page(self,size,cursor):
        # Called under the journal lock. A page never mixes two observations.
        now=self._clock()
        if type(now) not in (int,float) or not math.isfinite(now):
            raise ValueError('Invalid inventory clock')
        for identity,row in list(self._snapshots.items()):
            if now>=row['expires']:del self._snapshots[identity]
        if cursor is None:
            while len(self._snapshots)>=self._snapshot_limit:self._snapshots.popitem(last=False)
            identity=uuid.uuid4().hex; offset=0
            row=dict(seq=self._seq,expires=now+self._snapshot_ttl,size=size,
                entries=[dict(v) for _,v in sorted(self._entries.items())])
            self._snapshots[identity]=row
        else:
            _text(cursor)
            try:
                identity,number=cursor.split(':'); offset=int(number)
                if number!=str(offset) or offset<=0:raise ValueError()
            except ValueError:raise ValueError('Invalid page cursor') from None
            row=self._snapshots.get(identity)
            if row is None:raise RecoveryRequired('Inventory snapshot expired or was evicted')
            if size!=row['size'] or offset%size or offset>=len(row['entries']):
                raise ValueError('Page cursor does not match its snapshot')
        total=len(row['entries']); stop=min(total,offset+size)
        return dict(schema=2,node=self.node,epoch=self.epoch,seq=row['seq'],
            kind='snapshot_page',snapshot=identity,offset=offset,total=total,
            entries=[dict(v) for v in row['entries'][offset:stop]],
            next_cursor=identity+':'+str(stop) if stop<total else None)


class InventoryView:
    """A consumer detects gaps before exposing any part of an update."""
    def __init__(self,node,*,entry_limit=16384,event_limit=32768):
        self.node=_text(node); self.entry_limit=entry_limit; self.event_limit=event_limit
        self._epoch=None; self._seq=0; self._entries={}; self._retired=set()
        self._pending_page=None; self._completed_page=None

    @property
    def cursor(self):return dict(epoch=self._epoch,since=self._seq) if self._epoch else {}

    @property
    def entries(self):return [dict(v) for _,v in sorted(self._entries.items())]

    def apply(self,packet):
        if type(packet) is dict and type(packet.get('schema')) is int and packet['schema']==2:
            return self._apply_page(packet)
        if type(packet) is not dict or packet.get('schema')!=1 or type(packet.get('schema')) is not int:
            raise ValueError('Unsupported inventory protocol')
        if packet.get('node')!=self.node:raise ValueError('Inventory node mismatch')
        epoch=_text(packet.get('epoch')); seq=packet.get('seq')
        if type(seq) is not int or seq<0:raise ValueError('Invalid inventory sequence')
        common={'schema','node','epoch','seq','kind'}
        if epoch in self._retired:raise RecoveryRequired('Retired inventory epoch')
        if packet.get('kind')=='snapshot':
            if set(packet)!=common|{'entries'}:raise ValueError('Unknown snapshot fields')
            next_entries=_entries(packet['entries'],self.entry_limit)
            if epoch==self._epoch and seq<self._seq:raise RecoveryRequired('Stale snapshot')
        elif packet.get('kind')=='delta':
            if set(packet)!=common|{'base','events'}:raise ValueError('Unknown delta fields')
            events=packet['events']; base=packet['base']
            if (type(base) is not int or base<0 or type(events) is not list
                    or len(events)>self.event_limit or seq-base!=len(events)):
                raise RecoveryRequired('Invalid or incomplete delta')
            checked=[]
            for i,event in enumerate(events,base+1):
                if (type(event) is not dict or set(event)!={'seq','action','entry'}
                        or type(event['seq']) is not int or event['seq']!=i
                        or event['action'] not in ('put','drop')):
                    raise RecoveryRequired('Inventory sequence gap')
                checked.append((event['action'],_entry(event['entry'])))
            if epoch!=self._epoch:raise RecoveryRequired('Snapshot required for new epoch')
            if seq<=self._seq:return  # Validated repeated delivery; no mutation.
            if base!=self._seq:raise RecoveryRequired('Inventory sequence gap')
            next_entries=dict(self._entries)
            for action,row in checked:
                if action=='put':next_entries[_key(row)]=row
                else:next_entries.pop(_key(row),None)
            if len(next_entries)>self.entry_limit:raise ValueError('Inventory capacity exceeded')
        else:raise ValueError('Unknown inventory message kind')
        if self._epoch is not None and epoch!=self._epoch:
            if len(self._retired)>=128:raise RecoveryRequired('Reconnect with a fresh consumer')
            self._retired.add(self._epoch)
        self._entries,self._epoch,self._seq=next_entries,epoch,seq
        self._pending_page=None

    def _apply_page(self,packet):
        fields={'schema','node','epoch','seq','kind','snapshot','offset','total','entries','next_cursor'}
        if (set(packet)!=fields or packet['node']!=self.node or packet['kind']!='snapshot_page'):
            raise ValueError('Unsupported inventory page')
        epoch=_text(packet['epoch']); identity=_text(packet['snapshot'])
        seq,offset,total=packet['seq'],packet['offset'],packet['total']
        if (any(type(n) is not int or n<0 for n in (seq,offset,total))
                or total>self.entry_limit or offset>total):
            raise ValueError('Invalid inventory page extent')
        checked=_entries(packet['entries'],min(1024,self.entry_limit)); stop=offset+len(checked)
        if (stop>total or (not checked and total!=0)
                or packet['next_cursor']!=(identity+':'+str(stop) if stop<total else None)):
            raise ValueError('Invalid inventory page continuation')
        if epoch in self._retired or (epoch==self._epoch and seq<self._seq):
            raise RecoveryRequired('Stale inventory page')
        digest=hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        signature=(epoch,seq,identity,total)
        complete=self._completed_page
        if complete is not None and complete['signature']==signature:
            if complete['hashes'].get(offset)!=digest:raise RecoveryRequired('Changed repeated inventory page')
            return
        pending=self._pending_page
        if pending is not None and pending['signature']==signature and offset in pending['hashes']:
            if pending['hashes'][offset]!=digest:raise RecoveryRequired('Changed repeated inventory page')
            return
        if offset==0:
            pending=dict(signature=signature,entries={},hashes={})
        if pending is None or pending['signature']!=signature or offset!=len(pending['entries']):
            raise RecoveryRequired('Inventory page gap; restart recovery')
        if pending['entries'].keys() & checked.keys():raise ValueError('Duplicate paged inventory location')
        entries=dict(pending['entries']); entries.update(checked)
        hashes=dict(pending['hashes']); hashes[offset]=digest
        if stop==total:
            self.apply(dict(schema=1,node=self.node,epoch=epoch,seq=seq,kind='snapshot',entries=list(entries.values())))
            self._completed_page=dict(signature=signature,hashes=hashes)
        else:
            self._pending_page=dict(signature=signature,entries=entries,hashes=hashes)
