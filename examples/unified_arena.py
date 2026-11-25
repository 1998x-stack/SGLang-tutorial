"""CPU teaching: capacity lower bound and safe virtual page moves."""
def fixed_fit(demand,caps):
    return len(demand)==len(caps) and all(0<=d<=c for d,c in zip(demand,caps))
def byte_bound(demand,units,budget,fixed=0):
    if len(demand)!=len(units) or min([budget,fixed,*demand,*units])<0:
        raise ValueError("dimensions")
    return fixed+sum(d*b for d,b in zip(demand,units))<=budget
class Pages:
    def __init__(self):
        self.mapping={7:5}; self.pending={5}; self.fixed=set()
    def move(self,v,dest):
        old=self.mapping[v]
        if old in self.pending or old in self.fixed or dest in self.pending or dest in self.fixed or dest in self.mapping.values():
            raise RuntimeError("hazard")
        self.mapping[v]=dest
def selftest():
    assert not fixed_fit([6,4],[4,8]) and byte_bound([6,4],[1,1],12)
    assert not byte_bound([6,4],[1,1],12,3)
    p=Pages()
    try: p.move(7,2)
    except RuntimeError: pass
    else: raise AssertionError("pending hazard")
    assert p.mapping=={7:5}
    p.pending.clear(); p.move(7,2)
    assert p.mapping=={7:2}
if __name__=="__main__":
    selftest()
    print("capacity bound and hazard-safe remap passed")
