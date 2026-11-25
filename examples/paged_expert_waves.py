"""Toy wave planner and weighted MoE combine, no GPU."""
from fractions import Fraction
def plan(routes,k,resident=()):
    if k<=0:
        raise ValueError("slot capacity")
    union=set(e for row in routes for e,_ in row)
    preferred=list(dict.fromkeys(e for e in resident if e in union))
    order=preferred+sorted(union-set(preferred))
    return [order[i:i+k] for i in range(0,len(order),k)]
def combine(routes,waves,outputs):
    seen=set()
    result=[Fraction(0) for _ in routes]
    for wave in waves:
        for i,row in enumerate(routes):
            for j,(e,g) in enumerate(row):
                if e in wave:
                    if (i,j) in seen:
                        raise ValueError("duplicate entry")
                    seen.add((i,j))
                    result[i]+=g*outputs[i,e]
    if len(seen)!=sum(map(len,routes)):
        raise ValueError("missing entry")
    return result
def selftest():
    routes=[[(1,Fraction(1,2)),(4,Fraction(1,2))],[(4,Fraction(1,4)),(6,Fraction(3,4))],[(2,Fraction(1,2)),(7,Fraction(1,2))]]
    out={(i,e):e for i,row in enumerate(routes) for e,_ in row}
    w=plan(routes,3,[1,4])
    assert len(w)==2 and combine(routes,w,out)==combine(routes,plan(routes,5),out)
    assert plan([],3)==[]
    try:
        combine(routes,[],out)
    except ValueError:
        pass
    else:
        raise AssertionError("missing entries")
if __name__=="__main__":
    selftest()
    print(plan([[(1,1),(4,1)],[(6,1),(2,1),(7,1)]],3,[1,4]))
