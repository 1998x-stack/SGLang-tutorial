"""Toy latency estimate and ready filtering; not llm-d EPP or Kubernetes."""
from dataclasses import dataclass
@dataclass
class Worker:
    name:str
    cached:int
    rate:float
    queue:float
    ready:bool=True
def estimate(n,w):
    if n<0 or not 0<=w.cached<=n or w.rate<=0 or w.queue<0: raise ValueError("model")
    return w.queue+(n-w.cached)/w.rate
def choose(n,workers):
    eligible=[w for w in workers if w.ready]
    return min(eligible,key=lambda w:(estimate(n,w),w.name)).name if eligible else None
def selftest():
    a=Worker("A",800,100,10); b=Worker("B",0,100,0)
    assert estimate(1000,a)==12 and estimate(1000,b)==10
    assert choose(1000,[a,b])=="B"
    a.queue=1; assert choose(1000,[a,b])=="A"
    a.ready=False; assert choose(1000,[a,b])=="B"
    b.ready=False; assert choose(1000,[a,b]) is None
if __name__=="__main__":
    selftest()
    print("cache locality, queue and readiness checks passed")
