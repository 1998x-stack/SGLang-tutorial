"""Toy priorities and high-water T-LRU budget; not a radix implementation."""
from dataclasses import dataclass
@dataclass
class Leaf:
    name:str
    time:int
    hits:int
    locked:bool=False
def order(nodes,policy,threshold=2):
    keys={"lru":lambda n:(n.time,),"lfu":lambda n:(n.hits,n.time),"slru":lambda n:(n.hits>=threshold,n.time)}
    return [n.name for n in sorted((n for n in nodes if not n.locked),key=keys[policy])]
def tel_safe(history,cached,leaf,next_prompt,threshold):
    if min(history,cached,leaf,next_prompt,threshold)<0 or not leaf<=cached<=history or threshold<=next_prompt: raise ValueError("budget")
    return cached-leaf>=max(history+next_prompt-threshold,0)
def selftest():
    a=[Leaf("A",1,6),Leaf("B",2,1),Leaf("C",3,0)]
    assert order(a,"lru")==["A","B","C"]
    assert order(a,"lfu")==["C","B","A"]
    assert order(a,"slru")==["B","C","A"]
    assert tel_safe(1000,1000,100,100,300)
    assert not tel_safe(1000,800,100,100,300)
    a[0].locked=True
    assert order(a,"lru")==["B","C"]
if __name__=="__main__":
    selftest()
    print("policy, hard eligibility and T-LRU budget checks passed")
