"""Toy Multi-LoRA GPU slot pool; None represents the shared base-model ID."""
from collections import OrderedDict
class PoolFull(RuntimeError): pass
class AdapterPool:
    def __init__(self, max_loras_per_batch):
        if max_loras_per_batch<1: raise ValueError("pool needs at least the base slot")
        self.capacity=max_loras_per_batch
        self.slots=OrderedDict([(None,False)])  # None is an ID, not a pinned adapter
    def load(self, name, pinned=False):
        if name in self.slots:
            self.slots.move_to_end(name); self.slots[name]=self.slots[name] or pinned; return
        pinned_count=sum(1 for key,value in self.slots.items() if key is not None and value)
        if pinned and pinned_count >= self.capacity-1: raise PoolFull("pinned adapter limit reached")
        while len(self.slots)>=self.capacity:
            victim=next((k for k,v in self.slots.items() if k is not None and not v),None)
            if victim is None: raise PoolFull("all adapter slots are pinned or base")
            del self.slots[victim]
        self.slots[name]=pinned
    def touch(self,name):
        if name not in self.slots: raise KeyError(name)
        self.slots.move_to_end(name)
    def distinct_ids_fit(self,ids): return len(set(ids))<=self.capacity
if __name__=="__main__":
    p=AdapterPool(3); p.load("A",pinned=True); p.load("B"); p.load("C")
    print("slots:",list(p.slots))
