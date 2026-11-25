"""CPU streaming resource model; no cloud network."""
def peak(c,b,m0):
    if c<0 or b<=0 or m0<0: raise ValueError("budget")
    return c*b+m0
def max_streams(m,b,m0):
    if m<0 or b<=0 or m0<0: raise ValueError("budget")
    return max(0,(m-m0)//b)
def lower_bound(w,band):
    if w<0 or band<=0: raise ValueError("band")
    return w/band
def selftest():
    assert max_streams(1024,128,256)==6
    assert [peak(c,128,256) for c in [2,6,8]]==[512,1024,1280]
    assert lower_bound(32,4)==8
    assert max_streams(100,128,256)==0
if __name__=="__main__":
    selftest()
    print(max_streams(1024,128,256))
