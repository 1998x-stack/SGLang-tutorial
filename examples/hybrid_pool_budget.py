"""Toy integer byte budgeting for two distinct pools."""
from fractions import Fraction
def budget(total, ratio, state_bytes, kv_bytes, slots_per_request):
    if total < 0 or ratio < 0 or min(state_bytes, kv_bytes, slots_per_request) <= 0:
        raise ValueError("invalid budget")
    r = Fraction(ratio)
    state = int(total*r/(1+r))
    slots = state//state_bytes
    return slots, (total-state)//kv_bytes, slots//slots_per_request
def reusable(kv_prefix, checkpoint_prefix):
    return kv_prefix == checkpoint_prefix and kv_prefix is not None
def selftest():
    assert budget(480,Fraction(1,2),8,1,4) == (20,320,5)
    assert budget(480,1,8,1,4) == (30,240,7)
    assert budget(0,0,8,1,4) == (0,0,0)
    assert reusable((1,2),(1,2)) and not reusable((1,2),(1,))
    assert not reusable(None,None)
    try:
        budget(1,-1,8,1,4)
    except ValueError:
        pass
    else:
        raise AssertionError("negative ratio")
if __name__ == "__main__":
    selftest()
    print(budget(480,2,8,1,4))
