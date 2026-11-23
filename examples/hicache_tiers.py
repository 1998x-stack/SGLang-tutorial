"""Toy HiCache tier lookup; not a storage benchmark."""
class HiCacheToy:
    def __init__(self): self.l1={}; self.l2={}; self.l3={}
    def put(self, instance, key, tier, namespace=None):
        if tier=="L1": self.l1.setdefault(instance,set()).add(key)
        elif tier=="L2": self.l2.setdefault(instance,set()).add(key)
        elif tier=="L3":
            if namespace is None: raise ValueError("L3 needs backend namespace")
            self.l3.setdefault(namespace,set()).add(key)
        else: raise ValueError("tier must be L1/L2/L3")
    def lookup(self, instance, key, namespace=None):
        if key in self.l1.get(instance,set()): return "L1",("local lookup","GPU attention")
        if key in self.l2.get(instance,set()): return "L2",("local lookup","Host → GPU","GPU attention")
        if namespace is not None and key in self.l3.get(namespace,set()): return "L3",("backend lookup","L3 → L2","L2 → L1","GPU attention")
        return "MISS",("local lookup","prefill missing prefix")

def write_levels(policy, is_hot=False):
    if policy=="write_through": return ("L2","L3")
    if policy=="write_through_selective": return ("L2","L3") if is_hot else ()
    if policy=="write_back": return ()
    raise ValueError("unknown write policy")

if __name__=="__main__":
    c=HiCacheToy(); c.put("A","doc1","L1"); c.put("A","doc2","L2"); c.put("A","doc3","L3","cluster")
    print("L1",c.lookup("A","doc1","cluster")); print("L2",c.lookup("A","doc2","cluster"))
    print("L3",c.lookup("B","doc3","cluster")); print("miss",c.lookup("B","doc2","cluster"))
