"""CPU teaching for readiness and ideal bandwidth bound, not a remote loader."""
def transfer_lower_bound(weights,bands,shared=None):
    if len(weights)!=len(bands) or not weights or min(weights)<0 or min(bands)<=0 or (shared is not None and shared<=0): raise ValueError("bandwidth")
    t=max(w/b for w,b in zip(weights,bands))
    return max(t,sum(weights)/shared) if shared is not None else t
class Gate:
    def __init__(self,ranks,version):
        if ranks<=0: raise ValueError("ranks")
        self.expected=set(range(ranks)); self.version=version; self.complete=set(); self.failed=False; self.initialized=False
    def record(self,rank,version):
        if rank not in self.expected or version!=self.version:
            self.failed=True
            raise ValueError("rank or revision")
        self.complete.add(rank)
    @property
    def ready(self):
        return not self.failed and self.initialized and self.complete==self.expected
def selftest():
    assert transfer_lower_bound([8]*4,[4]*4)==2
    assert transfer_lower_bound([8]*4,[4]*4,4)==8
    g=Gate(2,"v1"); g.record(0,"v1"); g.initialized=True
    assert not g.ready
    g.record(1,"v1"); assert g.ready
    try: g.record(1,"v2")
    except ValueError: pass
    else: raise AssertionError("revision")
    assert not g.ready
if __name__=="__main__":
    selftest()
    print("rank readiness and bandwidth bounds passed")
