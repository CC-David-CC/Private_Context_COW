"""Normalized SQLite presentation history and branch catalog.

This is not a numerical snapshot store. Model execution currently replays the
selected message history through ordinary serving; no native COW claim is made.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid


class WorkspaceConflict(ValueError):
    pass


class WorkspaceNotFound(ValueError):
    pass


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def label(value):
    if not isinstance(value,str) or not value.strip() or len(value)>120:
        raise ValueError('A name must have 1 to 120 characters')
    return value.strip()


SCHEMA = '''
CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at REAL NOT NULL,archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN(0,1)));
CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,role TEXT NOT NULL CHECK(role IN('user','assistant')),text TEXT NOT NULL,reasoning TEXT NOT NULL,time REAL NOT NULL,meta TEXT,error TEXT,stopped INTEGER NOT NULL CHECK(stopped IN(0,1)),think_secs REAL,tool_limit INTEGER);
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,kind TEXT NOT NULL CHECK(kind IN('file','image')),name TEXT NOT NULL,content TEXT);
CREATE TABLE IF NOT EXISTS message_assets(message_id TEXT NOT NULL REFERENCES messages(id),ordinal INTEGER NOT NULL CHECK(ordinal>=0),asset_id TEXT NOT NULL REFERENCES assets(id),PRIMARY KEY(message_id,ordinal));
CREATE TABLE IF NOT EXISTS tool_records(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS message_tools(message_id TEXT NOT NULL REFERENCES messages(id),ordinal INTEGER NOT NULL CHECK(ordinal>=0),tool_id TEXT NOT NULL REFERENCES tool_records(id),PRIMARY KEY(message_id,ordinal));
CREATE TABLE IF NOT EXISTS history_nodes(id TEXT PRIMARY KEY,parent_id TEXT REFERENCES history_nodes(id),message_id TEXT REFERENCES messages(id),CHECK((parent_id IS NULL)=(message_id IS NULL)),CHECK(parent_id IS NULL OR parent_id<>id),UNIQUE(parent_id,message_id));
CREATE TABLE IF NOT EXISTS chat_roots(chat_id TEXT PRIMARY KEY REFERENCES chats(id),node_id TEXT NOT NULL UNIQUE REFERENCES history_nodes(id));
CREATE TABLE IF NOT EXISTS branches(id TEXT PRIMARY KEY,chat_id TEXT NOT NULL REFERENCES chats(id),name TEXT NOT NULL,created_at REAL NOT NULL,UNIQUE(chat_id,name));
CREATE TABLE IF NOT EXISTS branch_heads(branch_id TEXT PRIMARY KEY REFERENCES branches(id),tip TEXT NOT NULL REFERENCES history_nodes(id),revision INTEGER NOT NULL CHECK(revision>=0));
CREATE TABLE IF NOT EXISTS branch_moves(branch_id TEXT NOT NULL REFERENCES branches(id),revision INTEGER NOT NULL CHECK(revision>0),from_node TEXT NOT NULL REFERENCES history_nodes(id),to_node TEXT NOT NULL REFERENCES history_nodes(id),at REAL NOT NULL,PRIMARY KEY(branch_id,revision));
CREATE TABLE IF NOT EXISTS operation_receipts(id TEXT PRIMARY KEY,request_hash TEXT NOT NULL,result TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,branch_id TEXT NOT NULL REFERENCES branches(id),input_node TEXT NOT NULL REFERENCES history_nodes(id),input_revision INTEGER NOT NULL,settings TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN('pending','completed','stopped','failed','interrupted')),result_node TEXT REFERENCES history_nodes(id));
CREATE UNIQUE INDEX IF NOT EXISTS one_pending_request_per_branch ON requests(branch_id) WHERE status='pending';
CREATE INDEX IF NOT EXISTS nodes_parent ON history_nodes(parent_id);
CREATE INDEX IF NOT EXISTS branch_chat ON branches(chat_id);
'''


class ChatWorkspace:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self._owner=open(str(self.path)+'.lock','a+b')
        self._owner.seek(0,2)
        if not self._owner.tell():self._owner.write(b'0');self._owner.flush()
        self._owner.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self._owner.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self._owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise WorkspaceConflict('Another server owns this chat database') from None
        with self.connection() as db:
            version=db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0,1):raise ValueError('Unsupported chat database version')
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript(SCHEMA)
            for table in ['messages','assets','message_assets','tool_records','message_tools','history_nodes','branch_moves','operation_receipts','chat_roots']:
                for action in ['UPDATE','DELETE']:
                    db.execute(f'CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,"immutable history"); END')
            db.execute('PRAGMA user_version=1')
            # Constructor is server-start recovery, not a per-request connection.
            db.execute("UPDATE requests SET status='interrupted' WHERE status='pending'")
        if self.path.exists():self.path.chmod(0o600)

    def close(self):
        self._owner.close()

    def __del__(self):
        if hasattr(self,'_owner'):self._owner.close()

    @contextmanager
    def connection(self,write=False):
        if self._owner.closed:raise WorkspaceConflict('This chat database owner is closed')
        db=sqlite3.connect(self.path,timeout=20,isolation_level=None)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA synchronous=FULL')
        try:
            if write:db.execute('BEGIN IMMEDIATE')
            yield db
            if write:db.commit()
        except BaseException:
            if write:db.rollback()
            raise
        finally:db.close()

    def _receipt(self,db,op,request,result=None):
        if not isinstance(op,str) or not 1<=len(op)<=128:raise ValueError('An operation ID is required')
        digest=identity(request)
        old=db.execute('SELECT * FROM operation_receipts WHERE id=?',(op,)).fetchone()
        if old:
            if old['request_hash']!=digest:raise ValueError('Operation ID reused for a different request')
            return json.loads(old['result'])
        if result is not None:
            db.execute('INSERT INTO operation_receipts VALUES(?,?,?)',(op,digest,canonical(result)))
        return None

    def _head(self,db,branch):
        row=db.execute('SELECT b.*,h.tip,h.revision FROM branches b JOIN branch_heads h ON h.branch_id=b.id WHERE b.id=?',(branch,)).fetchone()
        if row is None:raise WorkspaceNotFound('Branch not found')
        return dict(row)

    def _not_busy(self,db,branch):
        if db.execute("SELECT 1 FROM requests WHERE branch_id=? AND status='pending'",(branch,)).fetchone():
            raise WorkspaceConflict('This branch has an unfinished answer')

    def _move(self,db,head,revision,target):
        if type(revision) is not int or revision<0:raise ValueError('A branch revision is required')
        if head['revision']!=revision:raise WorkspaceConflict('The branch changed; reload before editing it')
        if revision>=2**63-2:raise WorkspaceConflict('Branch revision exhausted')
        root=self._root(db,target)
        own=db.execute('SELECT node_id FROM chat_roots WHERE chat_id=?',(head['chat_id'],)).fetchone()[0]
        if root!=own:raise ValueError('History belongs to another chat')
        db.execute('UPDATE branch_heads SET tip=?,revision=? WHERE branch_id=?',(target,revision+1,head['id']))
        db.execute('INSERT INTO branch_moves VALUES(?,?,?,?,?)',(head['id'],revision+1,head['tip'],target,time.time()))
        return {**head,'tip':target,'revision':revision+1}

    def _root(self,db,node):
        seen=set()
        while node:
            if node in seen:raise ValueError('History cycle')
            seen.add(node)
            row=db.execute('SELECT parent_id FROM history_nodes WHERE id=?',(node,)).fetchone()
            if row is None:raise WorkspaceNotFound('History point not found')
            if row['parent_id'] is None:return node
            node=row['parent_id']

    def create_chat(self,title,op):
        title=label(title);request=['create',title]
        with self.connection(True) as db:
            old=self._receipt(db,op,request)
            if old:return old
            chat=uuid.uuid4().hex;root=identity(['chat-root',chat]);branch=uuid.uuid4().hex;now=time.time()
            db.execute('INSERT INTO chats(id,title,created_at) VALUES(?,?,?)',(chat,title,now))
            db.execute('INSERT INTO history_nodes VALUES(?,NULL,NULL)',(root,))
            db.execute('INSERT INTO chat_roots VALUES(?,?)',(chat,root))
            db.execute('INSERT INTO branches VALUES(?,?,?,?)',(branch,chat,'Main',now))
            db.execute('INSERT INTO branch_heads VALUES(?,?,0)',(branch,root))
            result={'id':chat,'title':title,'branch':self._head(db,branch)}
            self._receipt(db,op,request,result);return result

    def list_chats(self,include_archived=False):
        with self.connection() as db:
            db.execute('BEGIN')
            chats=[dict(r) for r in db.execute('SELECT * FROM chats '+('' if include_archived else 'WHERE archived=0 ')+'ORDER BY created_at DESC,id')]
            branches=self._graph_branches(db)
            nodes,_,_=self._graph_nodes(db,branches,previews=False)
            groups={}
            for branch in branches:groups.setdefault(branch['chat_id'],[]).append(branch)
            shared={}
            for node in nodes.values():
                count=node['current_branch_count']
                if count:shared.setdefault((node['_root'],count),0);shared[(node['_root'],count)]=max(shared[(node['_root'],count)],node['depth'])
            for chat in chats:
                group=groups.get(chat['id'],[])
                chat['branch_count']=len(group)
                chat['max_saved_messages']=max((nodes[b['tip']]['depth'] for b in group),default=0)
                chat['shared_prefix_messages']=shared.get((nodes[group[0]['tip']]['_root'],len(group)),0) if len(group)>1 else 0
            return chats

    def _graph_branches(self,db,chat=None):
        sql='''SELECT b.*,h.tip,h.revision,r.id AS pending_request FROM branches b
               JOIN branch_heads h ON h.branch_id=b.id
               LEFT JOIN requests r ON r.branch_id=b.id AND r.status='pending' '''
        return [dict(r) for r in db.execute(sql+('WHERE b.chat_id=? ' if chat else '')+'ORDER BY b.created_at,b.id',(chat,) if chat else ())]

    def _graph_nodes(self,db,branches,chat=None,include_previous=False,previews=True):
        """One ancestor-union query, then iterative linear tree accounting."""
        condition='WHERE b.chat_id=?' if chat else ''
        seeds='SELECT h.tip FROM branch_heads h JOIN branches b ON b.id=h.branch_id '+condition
        params=(chat,) if chat else ()
        if include_previous:
            seeds+=' UNION SELECT m.from_node FROM branch_moves m JOIN branches b ON b.id=m.branch_id '+condition
            params+=((chat,) if chat else ())
        fields=",m.role,substr(m.text,1,160) AS preview" if previews else ''
        join=' LEFT JOIN messages m ON m.id=n.message_id' if previews else ''
        rows=db.execute('''WITH RECURSIVE reachable(id) AS ('''+seeds+'''
            UNION SELECT n.parent_id FROM history_nodes n JOIN reachable r ON n.id=r.id WHERE n.parent_id IS NOT NULL)
            SELECT n.id,n.parent_id'''+fields+' FROM reachable r JOIN history_nodes n ON n.id=r.id'+join,params)
        nodes={r['id']:{**dict(r),'current_branch_count':0} for r in rows}
        children={node:[] for node in nodes};ordered=[]
        for node in nodes.values():
            if node['parent_id'] is None:
                node['depth']=0;node['_root']=node['id'];ordered.append(node['id'])
            elif node['parent_id'] in children:children[node['parent_id']].append(node['id'])
            else:raise ValueError('History parent missing')
        # Breadth-first iteration handles long histories without Python recursion.
        for node_id in ordered:
            node=nodes[node_id]
            for child_id in children[node_id]:
                child=nodes[child_id];child['depth']=node['depth']+1;child['_root']=node['_root'];ordered.append(child_id)
        if len(ordered)!=len(nodes):raise ValueError('History cycle')
        for branch in branches:
            if branch['tip'] not in nodes:raise ValueError('History tip missing')
            nodes[branch['tip']]['current_branch_count']+=1
        for node_id in reversed(ordered):
            node=nodes[node_id]
            if node['parent_id']:nodes[node['parent_id']]['current_branch_count']+=node['current_branch_count']
        return nodes,children,ordered

    def graph(self,chat,include_previous=False):
        with self.connection() as db:
            db.execute('BEGIN')  # Heads, counts, previews and request indicators share one WAL snapshot.
            root=db.execute('SELECT node_id FROM chat_roots WHERE chat_id=?',(chat,)).fetchone()
            if root is None:raise WorkspaceNotFound('Chat not found')
            root=root[0];branches=self._graph_branches(db,chat)
            nodes,children,ordered=self._graph_nodes(db,branches,chat,include_previous)
            if any(n['_root']!=root for n in nodes.values()):raise ValueError('History belongs to another chat')
            for branch in branches:branch['saved_messages']=nodes[branch['tip']]['depth']
            previous=[]
            if include_previous:
                # Only abandoned terminal paths need previous-tip badges, not every old append.
                terminal={n for n in ordered if not nodes[n]['current_branch_count'] and not children[n]}
                seen=set()
                for row in db.execute('''SELECT m.branch_id,m.from_node,m.revision FROM branch_moves m
                    JOIN branches b ON b.id=m.branch_id WHERE b.chat_id=? ORDER BY m.revision DESC,m.branch_id''',(chat,)):
                    key=(row['branch_id'],row['from_node'])
                    if row['from_node'] in terminal and key not in seen:
                        seen.add(key);previous.append({'branch_id':row['branch_id'],'node_id':row['from_node'],'revision':row['revision']})
            common=max((n['depth'] for n in nodes.values() if n['current_branch_count']==len(branches)),default=0) if len(branches)>1 else 0
            return {'chat_id':chat,'root_id':root,'branches':branches,
                'nodes':[{k:v for k,v in nodes[node].items() if k!='_root'} for node in ordered],
                'shared_prefix_messages':common,'previous_tips':previous,
                'execution_mode':'ordinary-serving-history-replay','native_cow_connected':False}

    def list_branches(self,chat):
        with self.connection() as db:
            return [dict(r) for r in db.execute('SELECT b.*,h.tip,h.revision FROM branches b JOIN branch_heads h ON h.branch_id=b.id WHERE chat_id=? ORDER BY created_at,id',(chat,))]

    def _message(self,db,message):
        if not isinstance(message,dict) or message.get('role') not in ('user','assistant'):raise ValueError('A user or assistant message is required')
        values={k:message.get(k) for k in ['meta','error','thinkSecs','limit']}
        text=message.get('text','');reasoning=message.get('reasoning','');stamp=message.get('time',time.time()*1000)
        if not isinstance(text,str) or not isinstance(reasoning,str) or len(text)+len(reasoning)>8*1024*1024:raise ValueError('Message text must be a bounded string')
        if type(stamp) not in (int,float) or not math.isfinite(stamp):raise ValueError('Invalid message time')
        for key in ['meta','error']:
            if values[key] is not None and not isinstance(values[key],str):raise ValueError('Invalid '+key)
        for key in ['thinkSecs','limit']:
            if values[key] is not None and (type(values[key]) not in (int,float) or not math.isfinite(values[key])):raise ValueError('Invalid '+key)
        normalized={**values,'role':message['role'],'text':text,'reasoning':reasoning,'time':stamp,'stopped':bool(message.get('stopped',False)),
                    'files':message.get('files',[]),'images':message.get('images',[]),'tools':message.get('tools',[])}
        mid=identity(normalized)
        db.execute('INSERT OR IGNORE INTO messages VALUES(?,?,?,?,?,?,?,?,?,?)',(mid,normalized['role'],text,reasoning,stamp,values['meta'],values['error'],int(normalized['stopped']),values['thinkSecs'],values['limit']))
        ordinal=0
        for field,kind,content_key in [('files','file','text'),('images','image','url')]:
            if not isinstance(normalized[field],list) or len(normalized[field])>64:raise ValueError('Invalid attachments')
            for item in normalized[field]:
                if not isinstance(item,dict) or not isinstance(item.get('name'),str):raise ValueError('Invalid attachment')
                content=item.get(content_key)
                if content is not None and not isinstance(content,str):raise ValueError('Invalid attachment content')
                asset=identity([kind,item['name'],content])
                db.execute('INSERT OR IGNORE INTO assets VALUES(?,?,?,?)',(asset,kind,item['name'],content))
                db.execute('INSERT OR IGNORE INTO message_assets VALUES(?,?,?)',(mid,ordinal,asset));ordinal+=1
        if not isinstance(normalized['tools'],list) or len(normalized['tools'])>128:raise ValueError('Invalid tool records')
        for ordinal,tool in enumerate(normalized['tools']):
            if not isinstance(tool,dict):raise ValueError('Invalid tool record')
            tid=identity(tool);db.execute('INSERT OR IGNORE INTO tool_records VALUES(?,?)',(tid,canonical(tool)))
            db.execute('INSERT OR IGNORE INTO message_tools VALUES(?,?,?)',(mid,ordinal,tid))
        return mid

    def _append(self,db,head,revision,message):
        mid=self._message(db,message);node=identity([head['tip'],mid])
        db.execute('INSERT OR IGNORE INTO history_nodes VALUES(?,?,?)',(node,head['tip'],mid))
        return self._move(db,head,revision,node)

    def append(self,branch,revision,message,op):
        request=['append',branch,revision,message]
        with self.connection(True) as db:
            old=self._receipt(db,op,request)
            if old:return old
            head=self._head(db,branch);self._not_busy(db,branch)
            result=self._append(db,head,revision,message);self._receipt(db,op,request,result);return result

    def fork(self,branch,node,name,op):
        name=label(name);request=['fork',branch,node,name]
        with self.connection(True) as db:
            old=self._receipt(db,op,request)
            if old:return old
            parent=self._head(db,branch)
            own=db.execute('SELECT node_id FROM chat_roots WHERE chat_id=?',(parent['chat_id'],)).fetchone()[0]
            if self._root(db,node)!=own:raise ValueError('Cannot fork another chat history')
            new=uuid.uuid4().hex
            try:db.execute('INSERT INTO branches VALUES(?,?,?,?)',(new,parent['chat_id'],name,time.time()))
            except sqlite3.IntegrityError:raise WorkspaceConflict('A branch already has this name') from None
            db.execute('INSERT INTO branch_heads VALUES(?,?,0)',(new,node))
            result=self._head(db,new);self._receipt(db,op,request,result);return result

    def rewind(self,branch,revision,node,op):
        request=['rewind',branch,revision,node]
        with self.connection(True) as db:
            old=self._receipt(db,op,request)
            if old:return old
            head=self._head(db,branch);self._not_busy(db,branch)
            result=self._move(db,head,revision,node);self._receipt(db,op,request,result);return result

    def clear(self,branch,revision,op):
        with self.connection() as db:
            head=self._head(db,branch);node=db.execute('SELECT node_id FROM chat_roots WHERE chat_id=?',(head['chat_id'],)).fetchone()[0]
        return self.rewind(branch,revision,node,op)

    def _read_message(self,db,mid,node):
        row=dict(db.execute('SELECT * FROM messages WHERE id=?',(mid,)).fetchone())
        message={'role':row['role'],'text':row['text'],'time':row['time'],'node_id':node}
        for target,key in [('reasoning','reasoning'),('meta','meta'),('error','error'),('thinkSecs','think_secs'),('limit','tool_limit')]:
            if row[key] not in (None,''):message[target]=row[key]
        if row['stopped']:message['stopped']=True
        for asset in db.execute('SELECT a.* FROM message_assets m JOIN assets a ON a.id=m.asset_id WHERE message_id=? ORDER BY ordinal',(mid,)):
            field,key=('files','text') if asset['kind']=='file' else ('images','url')
            item={'name':asset['name']}
            if asset['content'] is not None:item[key]=asset['content']
            message.setdefault(field,[]).append(item)
        tools=[json.loads(r[0]) for r in db.execute('SELECT t.payload FROM message_tools m JOIN tool_records t ON t.id=m.tool_id WHERE message_id=? ORDER BY ordinal',(mid,))]
        if tools:message['tools']=tools
        return message

    def history(self,branch):
        with self.connection() as db:
            head=self._head(db,branch);node=head['tip'];messages=[];seen=set()
            while node:
                if node in seen:raise ValueError('History cycle')
                seen.add(node);row=db.execute('SELECT * FROM history_nodes WHERE id=?',(node,)).fetchone()
                if row is None:raise WorkspaceNotFound('History point missing')
                if row['message_id']:messages.append(self._read_message(db,row['message_id'],node))
                node=row['parent_id']
            moves=[dict(r) for r in db.execute('SELECT * FROM branch_moves WHERE branch_id=? ORDER BY revision DESC LIMIT 30',(branch,))]
            pending=db.execute("SELECT id FROM requests WHERE branch_id=? AND status='pending'",(branch,)).fetchone()
            return {**head,'messages':list(reversed(messages)),'moves':moves,'pending_request':pending[0] if pending else None,
                    'execution_mode':'ordinary-serving-history-replay','native_cow_connected':False}

    def rename_chat(self,chat,title):
        with self.connection(True) as db:
            if db.execute('UPDATE chats SET title=? WHERE id=?',(label(title),chat)).rowcount!=1:raise WorkspaceNotFound('Chat not found')

    def rename_branch(self,branch,name):
        with self.connection(True) as db:
            try:
                if db.execute('UPDATE branches SET name=? WHERE id=?',(label(name),branch)).rowcount!=1:raise WorkspaceNotFound('Branch not found')
            except sqlite3.IntegrityError:raise WorkspaceConflict('A branch already has this name') from None

    def archive_chat(self,chat,archived):
        if type(archived) is not bool:raise ValueError('archived must be true or false')
        with self.connection(True) as db:
            if db.execute("SELECT 1 FROM requests r JOIN branches b ON b.id=r.branch_id WHERE b.chat_id=? AND r.status='pending'",(chat,)).fetchone():
                raise WorkspaceConflict('Wait for this chat answer before archiving')
            if db.execute('UPDATE chats SET archived=? WHERE id=?',(int(archived),chat)).rowcount!=1:raise WorkspaceNotFound('Chat not found')

    def claim(self,branch,revision,message,settings,op):
        request=['claim',branch,revision,message,settings]
        with self.connection(True) as db:
            old=self._receipt(db,op,request)
            if old:return {**old,'replayed':True}
            head=self._head(db,branch);self._not_busy(db,branch)
            effective=dict(settings)
            if db.execute('SELECT archived FROM chats WHERE id=?',(head['chat_id'],)).fetchone()[0]:effective['strata_checkpoint']=False
            result=self._append(db,head,revision,message)
            db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?,NULL)',(op,branch,result['tip'],result['revision'],canonical(effective),'pending'))
            result={**result,'request_id':op,'input_node':result['tip'],'settings':effective}
            self._receipt(db,op,request,result);return result

    def request(self,op):
        with self.connection() as db:
            row=db.execute('SELECT * FROM requests WHERE id=?',(op,)).fetchone()
            if row is None:raise WorkspaceNotFound('Request not found')
            return dict(row)

    def finish(self,op,message,status):
        if status not in ('completed','stopped','failed'):raise ValueError('Invalid request result')
        with self.connection(True) as db:
            request=db.execute('SELECT * FROM requests WHERE id=?',(op,)).fetchone()
            if request is None:raise WorkspaceNotFound('Request not found')
            if request['status']!='pending':return self._head(db,request['branch_id'])
            head=self._head(db,request['branch_id'])
            if head['revision']!=request['input_revision'] or head['tip']!=request['input_node']:raise WorkspaceConflict('Reserved branch moved')
            if message is not None:head=self._append(db,head,head['revision'],message)
            db.execute('UPDATE requests SET status=?,result_node=? WHERE id=?',(status,head['tip'],op))
            return head
