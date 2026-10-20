"""Locate the first unequal mixed activation in complete finite batched traces."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def read(path, position, stages=(0,1)):
    rows={}
    with path.open('rb') as f:
        while header:=f.read(32):
            if len(header)!=32: raise ValueError('Truncated trace header')
            pos,layer,half,count=struct.unpack('<4q',header)
            if pos!=position or layer<0 or half not in stages or not 0<count<=100000:
                raise ValueError('Unexpected trace coordinates')
            raw=f.read(4*count)
            if len(raw)!=4*count: raise ValueError('Truncated activation')
            values=array.array('f');values.frombytes(raw)
            if sys.byteorder!='little': values.byteswap()
            if not all(map(math.isfinite,values)): raise ValueError('Nonfinite activation')
            key=(layer,half)
            if key in rows: raise ValueError('Duplicate activation')
            rows[key]=values
    return rows


def compare(a,b,position,layers,stages=(0,1)):
    x,y=read(a,position,stages),read(b,position,stages)
    required={(l,h) for l in range(layers) for h in stages}
    if set(x)!=required or set(y)!=required: raise ValueError('Incomplete layer trace')
    records=[]
    for layer,half in sorted(required):
        u,v=x[layer,half],y[layer,half]
        if len(u)!=len(v): raise ValueError('Activation widths differ')
        i=max(range(len(u)),key=lambda k:abs(u[k]-v[k]))
        records.append(dict(layer=layer,half=half,count=len(u),max_absolute_error=abs(u[i]-v[i]),
                            worst_index=i,actual=u[i],reference=v[i]))
    return dict(position=position,first_difference=next((r for r in records if r['max_absolute_error']),None),
        comparisons=records,trace_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('actual',type=Path);p.add_argument('reference',type=Path)
    p.add_argument('--position',type=int,required=True);p.add_argument('--layers',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(a.actual,a.reference,a.position,a.layers)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['first_difference']))
