"""CPU teaching: ownership and dense causal score counts, not a GPU backend."""
def zigzag(n,c):
    if n<0 or c<=0: raise ValueError("invalid dimensions")
    q,r=divmod(n,2*c)
    blocks=[]; start=0
    for i in range(2*c):
        end=start+q+(i<r)
        blocks.append(list(range(start,end))); start=end
    return [blocks[i]+blocks[2*c-1-i] for i in range(c)]
def interleave(lengths,c):
    if c<=0 or any(n<0 for n in lengths): raise ValueError("invalid dimensions")
    flat=[(req,i) for req,n in enumerate(lengths) for i in range(n)]
    return [flat[r::c] for r in range(c)]
def work(groups,p=0):
    if p<0: raise ValueError("prefix")
    return [sum(p+i+1 for i in g) for g in groups]
def selftest():
    assert work(zigzag(16,4))==[34]*4
    assert sorted(sum(zigzag(17,4),[]))==list(range(17))
    assert interleave([5,3],4)[1]==[(0,1),(1,0)]
    assert zigzag(0,2)==[[],[]]
    try: zigzag(1,0)
    except ValueError: pass
    else: raise AssertionError("invalid size")
if __name__=="__main__":
    selftest()
    print(zigzag(16,4),work(zigzag(16,4)))
