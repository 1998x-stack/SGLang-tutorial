"""CPU reference for stable softmax partition merge; not SGLang collectives."""
import math
def partial(scores,values):
    if len(scores)!=len(values): raise ValueError("shape")
    if not scores: return (0.0,-math.inf)
    m=max(scores); weights=[math.exp(x-m) for x in scores]; z=sum(weights)
    return sum(w*v for w,v in zip(weights,values))/z,m+math.log(z)
def merge(parts):
    valid=[(o,l) for o,l in parts if l!=-math.inf]
    if not valid: raise ValueError("empty context")
    m=max(l for o,l in valid)
    z=sum(math.exp(l-m) for o,l in valid)
    return sum(o*math.exp(l-m) for o,l in valid)/z
def placement(v,c):
    if v<0 or c<=0: raise ValueError("address")
    return v%c,v//c
def selftest():
    assert placement(13,4)==(1,3)
    p=[partial([0],[2]),partial([math.log(3)],[10])]
    assert math.isclose(merge(p),8)
    scores=[1000,998,-1000,999]; values=[2,3,7,4]
    assert math.isclose(merge([partial(scores[::2],values[::2]),partial(scores[1::2],values[1::2])]),partial(scores,values)[0])
    assert merge([partial([],[]),partial([0],[7])])==7
    try: merge([])
    except ValueError: pass
    else: raise AssertionError("empty")
if __name__=="__main__":
    selftest()
    print(merge([partial([0],[2]),partial([math.log(3)],[10])]))
