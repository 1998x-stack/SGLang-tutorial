"""Deterministic top-k MoE dispatch/combine toy; no GPU communication is run."""
def dispatch_topk(routes, expert_to_rank):
    buckets = {rank: [] for rank in sorted(set(expert_to_rank.values()))}
    loads = {expert: 0 for expert in expert_to_rank}
    for token, route in enumerate(routes):
        if not route or len({e for e, _ in route}) != len(route): raise ValueError("routes need unique experts")
        if abs(sum(w for _, w in route)-1.0) > 1e-9: raise ValueError("gate weights must sum to one")
        for expert, weight in route:
            if expert not in expert_to_rank: raise KeyError(expert)
            buckets[expert_to_rank[expert]].append((token, expert, weight)); loads[expert] += 1
    return buckets, loads

def combine_outputs(routes, expert_outputs):
    return [sum(weight * expert_outputs[(token, expert)] for expert, weight in route) for token, route in enumerate(routes)]

def imbalance_ratio(loads):
    if not loads or sum(loads.values()) == 0: return 0.0
    return max(loads.values()) / (sum(loads.values()) / len(loads))

if __name__ == "__main__":
    routes=[[(0,.7),(2,.3)],[(1,.4),(3,.6)],[(0,.5),(3,.5)]]
    buckets,loads=dispatch_topk(routes,{0:0,1:0,2:1,3:1})
    values={(0,0):2,(0,2):4,(1,1):1,(1,3):3,(2,0):6,(2,3):2}
    print("rank buckets:",buckets); print("expert loads:",loads,"imbalance:",round(imbalance_ratio(loads),3))
    print("combined:",combine_outputs(routes,values))
