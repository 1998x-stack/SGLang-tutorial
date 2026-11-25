"""Toy provenance validator; not a PPO or SGLang implementation."""
from math import exp,isfinite
def validate_spans(n,spans):
    if n<0:
        raise ValueError("negative token count")
    end=0
    for version,a,b in spans:
        if not version or a!=end or not a<b<=n:
            raise ValueError("invalid version span")
        end=b
    if end!=n:
        raise ValueError("incomplete provenance")
    return True
def ratios(new,behavior,mask):
    if not len(new)==len(behavior)==len(mask):
        raise ValueError("token alignment")
    if any(m not in (0,1) for m in mask) or not all(isfinite(x) and x<=0 for x in list(new)+list(behavior)):
        raise ValueError("invalid mask/logprob")
    return [exp(p-q) if m else None for p,q,m in zip(new,behavior,mask)]
def selftest():
    assert validate_spans(4,[("41",0,2),("42",2,4)])
    assert validate_spans(0,[])
    assert ratios([-1,-2],[-1,-3],[1,0])==[1,None]
    for spans in [[("41",0,1),("42",2,4)],[("41",0,3),("42",2,4)]]:
        try:
            validate_spans(4,spans)
        except ValueError:
            pass
        else:
            raise AssertionError("bad coverage")
if __name__=="__main__":
    selftest()
    print(ratios([-1,-2],[-2,-2],[1,1]))
