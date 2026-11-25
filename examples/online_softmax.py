"""CPU-only stable tiled weighted softmax with explicit masked-score domain."""
from math import exp, isclose, isfinite, inf

def _validate(scores, values):
    if not scores or len(scores) != len(values):
        raise ValueError("scores and values must be non-empty and equally sized")
    if any(not isfinite(s) and s != -inf for s in scores) or any(not isfinite(v) for v in values):
        raise ValueError("non-finite value or invalid score")
    if all(s == -inf for s in scores):
        raise ValueError("all scores masked: normalization undefined")

def online_weighted_sum(scores, values, tile_size=1):
    _validate(scores, values)
    if tile_size <= 0:
        raise ValueError("tile size")
    m, l, u = -inf, 0.0, 0.0
    for start in range(0, len(scores), tile_size):
        pairs = [(s,v) for s,v in zip(scores[start:start+tile_size],values[start:start+tile_size],strict=True) if s != -inf]
        if not pairs:
            continue
        mb = max(s for s,_ in pairs)
        weights = [(exp(s-mb),v) for s,v in pairs]
        lb = sum(w for w,_ in weights)
        ub = sum(w*v for w,v in weights)
        new_m = max(m, mb)
        old_scale = 0.0 if m == -inf else exp(m-new_m)
        tile_scale = exp(mb-new_m)
        l, u = old_scale*l + tile_scale*lb, old_scale*u + tile_scale*ub
        m = new_m
    return u/l

def full_weighted_sum(scores, values):
    _validate(scores, values)
    m = max(scores)
    weights = [0.0 if s == -inf else exp(s-m) for s in scores]
    return sum(w*v for w,v in zip(weights,values,strict=True))/sum(weights)

def demo():
    scores=[-inf,-inf,2.0,-1.0,3.5,.25,-inf]
    values=[99,99,10,20,-4,7,99]
    full=full_weighted_sum(scores,values)
    for size in (1,2,3,7):
        assert isclose(online_weighted_sum(scores,values,size),full,rel_tol=1e-12,abs_tol=1e-12)
    for func in (online_weighted_sum,full_weighted_sum):
        for invalid in ([-inf,-inf],[float("nan"),0],[inf,0]):
            try: func(invalid,[1,2])
            except ValueError: pass
            else: raise AssertionError("invalid normalization")
    print(f"online={online_weighted_sum(scores,values,2):.12f}; full={full:.12f}")

if __name__=="__main__":
    demo()
