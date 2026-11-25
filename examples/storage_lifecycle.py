"""Toy DP fanout and cleanup order, not HiCache storage."""
def attach(ranks):
    state=[]
    for idle,ok in ranks: state.append(bool(idle and ok))
    return all(state),state
def detach(stop_ok):
    log=["drain","stop"]
    if stop_ok: log+=["release","drain"]
    return stop_ok,log
def selftest():
    assert attach([(True,True),(True,False)])==(False,[True,False])
    assert attach([(False,True)])==(False,[False])
    assert detach(True)==(True,["drain","stop","release","drain"])
    assert detach(False)==(False,["drain","stop"])
if __name__=="__main__":
    selftest()
    print(detach(True))
