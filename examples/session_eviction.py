"""Toy Full-component eviction among otherwise eligible candidates."""
from dataclasses import dataclass
@dataclass
class Node:
    name:str
    refs:set
    age:int
    locked:bool=False
def eviction_order(nodes):
    return sorted((n for n in nodes if not n.locked),key=lambda n:(bool(n.refs),len(n.refs),n.age,n.name))
def close(nodes,sid):
    for n in nodes: n.refs.discard(sid)
def accepts_registration(active_generation,request_generation):
    return active_generation is not None and active_generation==request_generation
def selftest():
    nodes=[Node("X",{"A","B"},0),Node("Y",{"A"},1),Node("Z",set(),2),Node("locked",set(),0,True)]
    assert [n.name for n in eviction_order(nodes)]==["Z","Y","X"]
    close(nodes,"A")
    assert [n.name for n in eviction_order(nodes)]==["Y","Z","X"]
    assert len(nodes)==4
    assert not accepts_registration(2,1) and not accepts_registration(None,1)
    assert accepts_registration(2,2)
if __name__=="__main__":
    selftest()
    print("soft protection, close and stale-generation checks passed")
