"""CPU recurrence and quadratic chunk sizing; not real PP runtime."""
from math import sqrt,isclose
def timeline(durations,chunks):
    if chunks<0 or not durations or min(durations)<=0: raise ValueError("dimensions")
    end=[[0]*chunks for _ in durations]
    for i,t in enumerate(durations):
        for j in range(chunks):
            end[i][j]=max(end[i-1][j] if i else 0,end[i][j-1] if j else 0)+t
    return end
def predict(a,b,L,T):
    if a<=0 or b<0 or L<0 or T<=0: raise ValueError("fit")
    B=2*a*L+b
    return 2*T/(B+sqrt(B*B+4*a*T))
def selftest():
    assert timeline([1,1,1],4)[-1][-1]==6
    assert timeline([1,2,1],4)[-1][-1]==10
    assert timeline([1,1],0)==[[],[]]
    assert isclose(predict(.001,.1,0,20),100)
    x=predict(.001,.1,100,20)
    assert 0<x<100 and isclose(.001*x*x+.3*x,20)
    try: predict(0,1,1,1)
    except ValueError: pass
    else: raise AssertionError("fit")
if __name__=="__main__":
    selftest()
    print(timeline([1,1,1],4),predict(.001,.1,100,20))
