"""Prepare the existing coding workflow; COW and RuntimeLog remain authoritative."""
import hashlib,json,os,re,stat,tempfile
from pathlib import Path
from .run_cli import _config,_pairs,execute,_source_notice
from .cow_client import CowClient,HTTPTransport,OperationJournal,OperationIndeterminate
PROGRAM="""FLOW coding
REPEAT attempts exhausted=stop max=32 until=verified
EXPAND decide from=current profile=control task=task -> proposed
PLAN freeze against=world from=proposed -> plan
ACT perform plan=plan -> outcome
TRANSFORM advance contract=cow-source input=proposed -> current
TRANSFORM feedback contract=tool-feedback input=outcome -> task
VERIFY check contract=command-evidence-and-finish input=outcome -> verified
END
END
"""
MAX_LAUNCH_BYTES=1024*1024
def _connection(text):
    try:
        return json.loads(text,object_pairs_hook=_pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite connection value')))
    except ValueError as exc:
        raise ValueError('connection must be JSON data without duplicate fields or executable expressions') from exc
def _bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def build_launch(project,task,connection,*,branch,read_paths=(),edit_paths=(),create_paths=(),test_commands=(),from_branch=None,from_snapshot=None,shell_timeout=120):
    project=Path(project).resolve(strict=True)
    if not project.is_dir():raise ValueError('project must be a directory')
    keys={'endpoint_url','endpoint_pin','model_contract_id','execution_profile_id','eos_tokens'}
    if type(connection) is not dict or set(connection)!=keys:
        raise ValueError('connection needs only endpoint/profile identities and EOS IDs; credentials belong in BW_COW_API_KEY')
    if type(task) is not str or not task.strip() or len(task.encode('utf8'))>262144:raise ValueError('bounded nonempty task text required')
    if type(branch) is not str or not 1<=len(branch)<=128:raise ValueError('bounded branch name required')
    if not test_commands or any(type(c) is not str or not c.strip() for c in test_commands):
        raise ValueError('at least one explicit verification command is required')
    if from_snapshot is not None and from_branch is None:raise ValueError('a snapshot requires its source branch')
    if from_branch is not None and (type(from_branch) is not str or not from_branch):raise ValueError('source branch required')
    if from_snapshot is not None and (type(from_snapshot) is not str or not re.fullmatch('[0-9a-f]{64}',from_snapshot)):raise ValueError('invalid source snapshot')
    HTTPTransport(connection['endpoint_url'])
    cow=dict(connection,branch=branch,mode='ask',bootstrap='coding',grammar_registration='inline')
    fork=None
    if from_branch is not None:
        cow.pop('bootstrap');cow['mode']='append_user'
        fork=dict(source_branch=from_branch,snapshot=from_snapshot)
    config=dict(schema=1,source=str(project),run_id=branch,
        grant=dict(read_paths=sorted(set(read_paths)|set(edit_paths)|set(create_paths)),
                   edit_paths=list(edit_paths),create_paths=list(create_paths),allow_tests=False,allow_shell=True),
        bindings=dict(task=dict(type='Text',value=task)),cow=cow,
        shell=dict(timeout=shell_timeout),
        verification=dict(schema=1,commands=[dict(command=c,disposition='exited',exit_code=0) for c in test_commands]))
    config=_config(_bytes(config).decode())
    return dict(schema=1,program=PROGRAM,config=config,fork=fork)
def _validate(launch):
    if type(launch) is not dict or set(launch)!={'schema','program','config','fork'} or type(launch['schema']) is not int or launch['schema']!=1 or launch['program']!=PROGRAM:
        raise ValueError('unsupported coding launch')
    _config(_bytes(launch['config']).decode())
    fork=launch['fork']
    if fork is not None:
        if (type(fork) is not dict or set(fork)!={'source_branch','snapshot'}
                or type(fork['source_branch']) is not str or not fork['source_branch']
                or type(fork['snapshot']) is not str or not re.fullmatch('[0-9a-f]{64}',fork['snapshot'])
                or launch['config']['cow'].get('mode')!='append_user'
                or 'bootstrap' in launch['config']['cow']):
            raise ValueError('retained launch must pin an exact source snapshot')
    elif launch['config']['cow'].get('bootstrap')!='coding':
        raise ValueError('new coding launch requires its bootstrap')
def _state(state,*,create=False,source=None):
    state=Path(state).absolute()
    if state.is_symlink():raise ValueError('state must be a real owned directory')
    if source is not None and state.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('run state must be outside the source project')
    if create and not state.exists():state.mkdir(mode=0o700)
    info=state.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o700 or info.st_uid!=os.getuid():
        raise ValueError('state must be an owned directory with mode0700')
    return state
def save_launch(state,launch):
    import fcntl
    _validate(launch)
    state=_state(state,create=True,source=launch['config']['source'])
    raw=_bytes(dict(launch=launch,sha256=hashlib.sha256(_bytes(launch)).hexdigest()))+b'\n'
    if len(raw)>MAX_LAUNCH_BYTES:raise ValueError('coding launch is too large')
    path=state/'coding-launch.json'
    # Serialize trusted setup publishers; atomic rename leaves one file link
    # even if the process stops before staging cleanup. No workflow lives here.
    lockfd=os.open(state/'.coding-launch.lock',os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(lockfd,'r+b') as lock:
        info=os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError('invalid coding setup lock')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if path.exists() or path.is_symlink():
            if read_launch(state)!=launch:raise ValueError('run setup changed; use a new run state')
            return
        fd,name=tempfile.mkstemp(prefix='.coding-launch-',dir=state);temporary=Path(name)
        try:
            with os.fdopen(fd,'wb') as stream:
                os.fchmod(stream.fileno(),0o600);stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
            directory=os.open(state,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            if temporary.exists():temporary.unlink()
def read_launch(state):
    state=_state(state)
    fd=os.open(state/'coding-launch.json',os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError('coding launch must be a private owned regular file')
        raw=stream.read(MAX_LAUNCH_BYTES+1)
    if len(raw)>MAX_LAUNCH_BYTES:raise ValueError('coding launch is too large')
    envelope=json.loads(raw)
    if type(envelope) is not dict or set(envelope)!={'launch','sha256'} or envelope['sha256']!=hashlib.sha256(_bytes(envelope['launch'])).hexdigest():
        raise ValueError('coding launch integrity check failed')
    _validate(envelope['launch'])
    return envelope['launch']
def _client(state,launch):
    cow=launch['config']['cow']
    return CowClient.http(cow['endpoint_url'],journal=OperationJournal(Path(state)/'cow-operations'),api_key=os.environ.get('BW_COW_API_KEY'),endpoint_pin=cow['endpoint_pin'],notice_handler=_source_notice)
def run_saved(state,*,max_steps=1024):
    if type(max_steps) is not int or max_steps<=0:raise ValueError('finite positive step budget required')
    launch=read_launch(state)
    if launch['fork'] is not None:
        fork=launch['fork'];operation='bw-code-fork-'+hashlib.sha256(_bytes(launch)).hexdigest()[:40]
        # The original pinned source and journal ID survive retries. An uncertain
        # fork remains uncertain; never select a newer head or invent a new ID.
        try:
            _client(state,launch).fork(fork['source_branch'],fork['snapshot'],launch['config']['cow']['branch'],operation)
        except OperationIndeterminate as exc:
            raise OperationIndeterminate('Original fork operation '+operation+' is unresolved; reconcile its original journal intent before continuing') from exc
    return execute('run',launch['program'],_bytes(launch['config']).decode(),state,max_steps=max_steps)
def start(args):
    if type(args.max_steps) is not int or args.max_steps<=0:raise ValueError('finite positive step budget required')
    from .__main__ import _source
    connection=_connection(_source(args.connection))
    launch=build_launch(args.project,_source(args.task_file),connection,branch=args.branch,
        read_paths=args.read,edit_paths=args.edit,create_paths=args.create,
        test_commands=args.test_command,from_branch=args.from_branch,
        from_snapshot=args.from_snapshot,shell_timeout=args.shell_timeout)
    existing=Path(args.state)/'coding-launch.json'
    if launch['fork'] is not None and launch['fork']['snapshot'] is None:
        if existing.exists():
            prior=read_launch(args.state)
            if prior['fork'] is not None and prior['fork']['source_branch']==args.from_branch:
                launch['fork']['snapshot']=prior['fork']['snapshot']
        if launch['fork']['snapshot'] is None:
            # Read-only selection before publication. The snapshot,not a live
            # branch name,is saved before any mutating setup/inference request.
            launch['fork']['snapshot']=_client(args.state,launch).head(args.from_branch).tip
    save_launch(args.state,launch)
    return run_saved(args.state,max_steps=args.max_steps)
