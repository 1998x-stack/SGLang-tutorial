"""Toy forced termination with multi-token marker; not ReasonerGrammarObject."""
class Budget:
    def __init__(self,limit,marker):
        if limit<0 or not marker: raise ValueError("budget")
        self.limit=limit; self.marker=tuple(marker); self.n=0; self.m=0; self.phase="think"
    def allowed(self):
        if self.phase=="answer": return None
        if self.m or self.n>=self.limit: return {self.marker[self.m]}
        return None
    def accept(self,t):
        if self.phase=="answer": return
        allowed=self.allowed()
        if allowed is not None and t not in allowed: raise ValueError("budget")
        if t==self.marker[self.m]:
            self.m+=1
            if self.m==len(self.marker): self.phase="answer"
        else: self.n+=1
def selftest():
    b=Budget(3,[90,91])
    for t in [1,2,3]: b.accept(t)
    assert b.allowed()=={90}
    try: b.accept(4)
    except ValueError: pass
    else: raise AssertionError("budget")
    b.accept(90); assert b.allowed()=={91}
    b.accept(91); assert b.phase=="answer"
    z=Budget(0,[90]); assert z.allowed()=={90}
    z.accept(90); assert z.phase=="answer"
if __name__=="__main__":
    selftest()
    print("budget and terminator invariants passed")
