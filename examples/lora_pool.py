"""Toy GPU adapter slot pool; base-model None slot is reserved for teaching."""
class PoolFull(RuntimeError): pass
class AdapterPool:
    def __init__(self,max_loras_per_batch,eviction_policy="lru"):
        if max_loras_per_batch<1: raise ValueError("capacity must be positive")
        if eviction_policy not in {"lru","fifo"}: raise ValueError("policy must be lru or fifo")
        self.capacity=max_loras_per_batch; self.policy=eviction_policy; self.clock=0
        self.slots={"<base>":(0,0)}; self.pinned={"<base>"}
    def acquire(self,adapter,pinned=False):
        name=adapter or "<base>"; self.clock+=1
        if name in self.slots:
            loaded,last=self.slots[name]; self.slots[name]=(loaded,self.clock)
            if pinned: self.pinned.add(name)
            return None
        victim=None
        if len(self.slots)>=self.capacity:
            candidates=[n for n in self.slots if n not in self.pinned]
            if not candidates: raise PoolFull("no evictable adapter slot")
            victim=min(candidates,key=lambda n:self.slots[n][1] if self.policy=="lru" else self.slots[n][0])
            del self.slots[victim]
        self.slots[name]=(self.clock,self.clock)
        if pinned: self.pinned.add(name)
        return victim
    def unpin(self,adapter): self.pinned.discard(adapter)
    @property
    def active(self): return tuple(self.slots)
if __name__=="__main__":
    p=AdapterPool(3); p.acquire("A"); p.acquire("B"); print("evicted:",p.acquire("C")); print("active:",p.active)
    q=AdapterPool(2); q.acquire("A",pinned=True)
    try: q.acquire("B")
    except PoolFull as e: print("wait:",e)
