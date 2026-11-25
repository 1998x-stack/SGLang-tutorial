"""CPU launch coverage and bandwidth teaching model."""
def grid(n,b):
    if n<0 or b<=0: raise ValueError("grid")
    return (n+b-1)//b
def active(n,b):
    return [sum(i<n for i in range(j*b,(j+1)*b)) for j in range(grid(n,b))]
def selftest():
    assert grid(130,64)==3 and active(130,64)==[64,64,2]
    assert active(0,64)==[] and active(64,64)==[64]
    assert sum(active(129,64))==129
    try: grid(1,0)
    except ValueError: pass
    else: raise AssertionError("block")
if __name__=="__main__":
    selftest()
    print(active(130,64))
