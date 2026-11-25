"""Toy sparse residency, not SGLang/CUDA."""
def swap_in(resident, selected, capacity):
    if capacity < 0 or len(resident) > capacity or len(set(resident)) != len(resident):
        raise ValueError("invalid pool")
    selected = list(dict.fromkeys(selected))
    if len(selected) > capacity:
        raise ValueError("working set exceeds capacity")
    slots = list(resident)
    misses = [x for x in selected if x not in slots]
    for x in misses:
        if len(slots) < capacity:
            slots.append(x)
        else:
            victim = next(i for i, v in enumerate(slots) if v not in selected)
            slots[victim] = x
    return slots, misses, [slots.index(x) for x in selected]

def selftest():
    assert swap_in([0,1,2,3],[1,3],4)[1] == []
    assert swap_in([0,1,2,3],[1,4],4) == ([4,1,2,3],[4],[1,0])
    assert swap_in([],[],0) == ([],[],[])
    try:
        swap_in([], [1,2], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("capacity")
if __name__ == "__main__":
    selftest()
    print(swap_in([0,1,2,3],[1,4,6],4))
