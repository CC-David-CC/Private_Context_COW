"""Check completed restored branches against already executed fresh baselines."""
import argparse
import array
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('campaign',type=Path)
    args=parser.parse_args();root=args.campaign
    def read(name): return json.loads((root/name).read_text())
    prepared,resumed,baseline=read('prepare.json'),read('resume.json'),read('calibration.json')
    result=dict(status='PASS',scope='early completed-branch audit; final campaign still required',comparisons=[])
    width=4*248320
    for name,actual in resumed['branches'].items():
        expected=prepared['branches'][name]
        if (actual['generated'] != expected['generated']
                or [s['controller'] for s in actual['states']] != [s['controller'] for s in expected['states']]
                or [s['frontier_sha256'] for s in actual['states']] != [s['frontier_sha256'] for s in expected['states']]):
            raise AssertionError('Restored/uninterrupted branch mismatch: '+name)
        observation=next(row['first'] for row in baseline['comparisons'] if row['label']==name)
        command=observation['command'];tokens=Path(command[command.index('--tokens-file')+1])
        prefix=tokens.with_suffix('.f32')
        prompt=Path(str(prefix)+'.requests.f32');generated=Path(str(prefix)+'.decode.f32')
        prompt_end=len(actual['states'][0]['history'])
        if actual['generated'] != observation['work']['generated']:
            raise AssertionError('Restored/fresh baseline tokens mismatch: '+name)
        for state in [prepared['root']]+actual['states']:
            if state['consumed'] < prompt_end: path,index=prompt,0
            elif state['consumed'] == prompt_end: path,index=prompt,1
            else: path,index=generated,state['consumed']-prompt_end-1
            with path.open('rb') as stream: stream.seek(index*width);reference=stream.read(width)
            data=(root/state['frontier']).read_bytes()
            if len(data)!=width or len(reference)!=width: raise ValueError('Incomplete frontier')
            a,b=array.array('f'),array.array('f');a.frombytes(data);b.frombytes(reference)
            maximum=max(abs(x-y) for x,y in zip(a,b))
            result['comparisons'].append(dict(branch=name,consumed=state['consumed'],max_logit_error=maximum))
            if maximum!=0: raise AssertionError('Restored/fresh baseline frontier mismatch: '+name)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
