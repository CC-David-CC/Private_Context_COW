"""Owned signed-division lesson variations and independent executable checks.

This bounded code-completion benchmark is not general BW/coding qualification.
The checker uses exact fractions, never the training completion as its oracle.
"""
import ast
from fractions import Fraction
import hashlib
import itertools
import math
import random
import string

def cases(count=1000,held_out=48,preservation=16,blind_test=16,seed=20261023):
    rng=random.Random(seed)
    templates=['# ceil({a}/{b})\n','\x23 round {a}/{b} upward\n','# signed ceiling division\n']
    pool=list(itertools.product(['ceildiv','updiv','cdiv','quot_up'],string.ascii_lowercase,string.ascii_lowercase,templates))
    rng.shuffle(pool);rows=[];seen=set()
    def add(split,kind,prompt,parameters,completion):
        identity=hashlib.sha256(prompt.encode()).hexdigest()
        if identity in seen:return False
        seen.add(identity);rows.append(dict(id=identity,split=split,kind=kind,prompt=prompt,parameters=parameters,completion=completion));return True
    for function,a,b,template in pool:
        if a==b:continue
        prompt=template.format(a=a,b=b)+'def '+function+'('+a+','+b+'):\n    return'
        add('train','ceil',prompt,[a,b],' -(-'+a+'//'+b+')\n')
        if len(rows)==count:break
    if len(rows)!=count:raise ValueError('Requested too many distinct training variations')
    for index in range(held_out):
        a,b=[('numerator','denominator'),('top','bottom'),('value','divisor'),('left','right')][index%4]
        prompt='# Round the exact ratio toward positive infinity. Case '+str(index)+'.\ndef fresh_'+str(index)+'('+a+','+b+'):\n    return'
        add('held_out','ceil',prompt,[a,b],' -(-'+a+'//'+b+')\n')
    for index in range(preservation):
        kind=['add','floor','multiply','abs'][index%4]
        a,b='x','y';params=[a] if kind=='abs' else [a,b]
        explanation={'add':'sum of integers','floor':'floor of signed ratio','multiply':'product of integers','abs':'absolute value'}[kind]
        prompt='# '+explanation+'. Case '+str(index)+'.\ndef control_'+str(index)+'('+','.join(params)+'):\n    return'
        answer={'add':' x+y\n','floor':' x//y\n','multiply':' x*y\n','abs':' abs(x)\n'}[kind]
        add('preservation',kind,prompt,params,answer)
    for index in range(blind_test):
        a,b=[('dividend','divisor'),('aa','bb')][index%2]
        prompt='# Smallest integer not below '+a+'/'+b+'. Final '+str(index)+'.\ndef final_'+str(index)+'('+a+','+b+'):\n    return'
        add('blind_test','ceil',prompt,[a,b],' -(-'+a+'//'+b+')\n')
    return rows

_NODES=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Name,ast.Load,ast.Constant,ast.Add,ast.Sub,ast.Mult,
    ast.FloorDiv,ast.Div,ast.Mod,ast.USub,ast.UAdd,ast.Call,ast.IfExp,ast.Compare,ast.Eq,ast.NotEq,
    ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.BoolOp,ast.And,ast.Or,ast.Not,ast.BitXor)

def check(row,completion):
    """Check an expression with fixed independent edge cases and fresh integers.

    Only a bounded arithmetic AST is evaluated. No attributes, imports, loops,
    comprehensions, filesystem, hidden test execution or arbitrary model code.
    """
    expression=completion.split('\n',1)[0].strip()
    try:
        if not expression or len(expression)>512:raise ValueError('Missing or oversized return expression')
        tree=ast.parse(expression,mode='eval');nodes=list(ast.walk(tree))
        if len(nodes)>128 or any(not isinstance(node,_NODES) for node in nodes):raise ValueError('Unsupported return expression')
        functions={'abs':abs,'int':int,'min':min,'max':max}
        for node in nodes:
            if isinstance(node,ast.Name) and node.id not in row['parameters'] and node.id not in functions:raise ValueError('Unbound name')
            if isinstance(node,ast.Constant) and (type(node.value) is not int or abs(node.value)>2**63):raise ValueError('Unsupported constant')
            if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in functions or node.keywords):raise ValueError('Unsupported call')
        code=compile(tree,'<owned-arithmetic-check>','eval')
        rng=random.Random(7419+int(row['id'][:8],16))
        pairs=[(a,b) for a in [-(2**60)+3,-7,-1,0,1,7,2**60+9] for b in [-97,-5,-2,-1,1,2,5,97]]
        pairs += [(rng.randrange(-10**9,10**9),rng.choice([-1,1])*rng.randrange(1,10000)) for _ in range(16)]
        for index,(a,b) in enumerate(pairs):
            bound=dict(zip(row['parameters'],[a,b]));actual=eval(code,{'__builtins__':{},**functions},bound)
            kind=row['kind']
            expected=(math.ceil(Fraction(a,b)) if kind=='ceil' else a//b if kind=='floor' else a+b if kind=='add' else a*b if kind=='multiply' else abs(a))
            if type(actual) is not int or actual!=expected:return dict(passed=False,reason='independent-integer-check',checks=index+1)
        return dict(passed=True,checks=len(pairs))
    except (ValueError,SyntaxError,TypeError,ZeroDivisionError,OverflowError,NameError) as error:
        return dict(passed=False,reason=type(error).__name__)
