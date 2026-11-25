"""Toy contribution state, not distributed runtime."""
class Stream:
    def __init__(self,r): self.r=r; self.pending=None; self.done=None
    def record(self,x):
        if self.pending is not None or self.done is not None: raise ValueError("pending")
        self.pending=tuple(x)
    def complete(self):
        if self.pending is None: raise ValueError("nothing owed")
        self.done=sum(self.pending); self.pending=None
    def fold(self):
        if self.done is None: raise ValueError("not complete")
        self.r+=self.done; self.done=None
        return self.r
def selftest():
    s=Stream(10); s.record([2,3])
    try: s.fold()
    except ValueError: pass
    else: raise AssertionError("pending")
    s.complete(); assert s.fold()==15
    try: s.fold()
    except ValueError: pass
    else: raise AssertionError("double fold")
if __name__=="__main__":
    selftest()
    print("single completion and residual update passed")
