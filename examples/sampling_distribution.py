"""CPU reference of selected torch fallback filter semantics; not a GPU sampler."""
import math
def softmax(z,t):
    if not z or t<=0 or any(math.isnan(v) for v in z): raise ValueError("logits")
    m=max(z)/t
    if not math.isfinite(m): raise ValueError("no finite maximum")
    w=[math.exp(v/t-m) for v in z]; s=sum(w)
    return [v/s for v in w]
def filter_probs(p,k,top_p,min_p):
    if not p or not 1<=k<=len(p) or not 0<top_p<=1 or not 0<=min_p<=1 or min(p)<0 or not math.isclose(sum(p),1): raise ValueError("parameters")
    order=sorted(range(len(p)),key=lambda i:(-p[i],i)); out=[0.0]*len(p); prefix=0.0
    for rank,i in enumerate(order):
        csum=prefix+p[i]; w=p[i] if rank<k else 0.0
        if csum-w<=top_p and w>=p[order[0]]*min_p: out[i]=w
        prefix=csum
    s=sum(out)
    if s==0: raise ValueError("empty support")
    return [w/s for w in out]
def selftest():
    p=[.5,.25,.15,.1]; q=filter_probs(p,3,.7,.4)
    assert math.isclose(q[0],2/3) and math.isclose(q[1],1/3) and q[2:]==[0,0]
    assert filter_probs(p,1,1,0)==[1,0,0,0]
    assert softmax([1000,1000],1)==[.5,.5]
    assert softmax([0,-math.inf],1)==[1,0]
    try: softmax([-math.inf,-math.inf],1)
    except ValueError: pass
    else: raise AssertionError("empty support")
if __name__=="__main__":
    selftest()
    print(filter_probs([.5,.25,.15,.1],3,.7,.4))
