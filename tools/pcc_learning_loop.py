"""Drive PCC's existing durable learning-run API with its ordinary request journal."""
from pathlib import Path
import argparse,json,os,re,sys
from tools.cow_cli import CowClient,RequestJournal,ClientError,_training_post,default_state_dir


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8099')
    parser.add_argument('--state-dir',type=Path)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('policies');sub.add_parser('list')
    p=sub.add_parser('status');p.add_argument('run_id')
    p=sub.add_parser('round');p.add_argument('run_id');p.add_argument('round_index',type=int)
    p=sub.add_parser('start');p.add_argument('--operation-id',required=True);p.add_argument('--policy-id',required=True)
    p.add_argument('--parent-job-id');p.add_argument('--expected-parent-sha256')
    for action in ('advance','reconcile','stop','preserve'):
        p=sub.add_parser(action);p.add_argument('run_id');p.add_argument('--operation-id',required=True)
        p.add_argument('--revision',type=int,help='Default: current run revision; retries retain the original revision')
        if action=='advance':p.add_argument('lesson_ids',nargs='*')
        if action=='preserve':p.add_argument('--name',required=True);p.add_argument('--notes-file',type=Path)
    args=parser.parse_args(argv)
    try:
        if hasattr(args,'run_id') and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,127}',args.run_id) is None:raise ClientError('Invalid learning run ID')
        if args.action=='round' and not 1<=args.round_index<=100000:raise ClientError('Positive bounded round number required')
        client=CowClient(args.url,os.environ.get('COW_API_KEY'))
        journal=RequestJournal(args.state_dir or os.environ.get('COW_CLI_STATE_DIR') or default_state_dir())
        if args.action=='policies':value=client.request('GET','/cow/training/learning-policies')
        elif args.action=='list':value=client.request('GET','/cow/training/learning-runs')
        elif args.action=='status':value=client.request('GET','/cow/training/learning-runs/'+args.run_id)
        elif args.action=='round':value=client.request('GET','/cow/training/learning-runs/'+args.run_id+'/rounds/'+str(args.round_index))
        elif args.action=='start':
            body=dict(operation_id=args.operation_id,policy_id=args.policy_id,parent_job_id=args.parent_job_id,
                expected_parent_sha256=args.expected_parent_sha256)
            value=_training_post(client,journal,'/cow/training/learning-runs',body)
        else:
            prior=journal.get(args.operation_id)
            if prior is not None and prior['path']!='/cow/training/learning-steps':raise ClientError('Operation belongs to another request')
            frozen=json.loads(prior['body_json']) if prior is not None else None
            revision=(args.revision if args.revision is not None else frozen['expected_revision'] if frozen is not None
                else client.request('GET','/cow/training/learning-runs/'+args.run_id)['revision'])
            body=dict(operation_id=args.operation_id,run_id=args.run_id,expected_revision=revision,action=args.action)
            if args.action=='advance' and args.lesson_ids:body['lesson_ids']=args.lesson_ids
            if args.action=='preserve':body.update(name=args.name,notes='' if args.notes_file is None else args.notes_file.read_text())
            value=_training_post(client,journal,'/cow/training/learning-steps',body)
        if args.action in ('status','round') and (value.get('run_id')!=args.run_id or value.get('promoted') is not False):raise ClientError('Learning read belongs to another run')
        if args.action=='round' and value.get('round')!=args.round_index:raise ClientError('Learning round identity differs')
        print(json.dumps(value,sort_keys=True));return 0
    except (ClientError,ValueError,OSError,KeyError) as error:
        print('Cannot advance learning run: '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
