"""Toy checkpoint eligibility and recompute count."""
def reuse(k,states):
    if k<0 or any(s<0 for s in states): raise ValueError("length")
    r=max([0]+[s for s in states if s<=k])
    return r,k-r
def selftest():
    assert reuse(850,[400,1000])==(400,450)
    assert reuse(850,[400,800,1000])==(800,50)
    assert reuse(850,[])==(0,850)
    assert reuse(0,[100])==(0,0)
if __name__=="__main__":
    selftest()
    print(reuse(850,[400,800,1000]))
