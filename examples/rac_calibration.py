"""CPU teaching example for activation distribution and latency; not a pruner."""
def squared_error(w,wp,rows):
    if len(w)!=len(wp) or any(len(x)!=len(w) for x in rows): raise ValueError("shape")
    return sum(sum((a-b)*v for a,b,v in zip(w,wp,x))**2 for x in rows)
def latency(prefill,tokens,per_token):
    if min(prefill,tokens,per_token)<0: raise ValueError("negative")
    return prefill+tokens*per_token
def selftest():
    w=[1,1]; keep1=[1,0]; keep2=[0,1]
    prompt=[[3,0]]; decode=[[0,4]]
    assert squared_error(w,keep1,prompt)==0
    assert squared_error(w,keep2,prompt)==9
    assert squared_error(w,keep1,prompt+decode)==16
    assert squared_error(w,keep2,prompt+decode)==9
    assert latency(100,1000,2)==2100 and latency(100,3000,1)==3100
    try: squared_error(w,keep1,[[1]])
    except ValueError: pass
    else: raise AssertionError("shape")
if __name__=="__main__":
    selftest()
    print("prompt-only vs joint errors: [0,9] vs [16,9]")
