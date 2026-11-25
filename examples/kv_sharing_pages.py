"""Toy page placement and conservative prefix matching."""
def placement(x,p,c):
    if x < 0 or p <= 0 or c <= 0:
        raise ValueError("invalid address")
    return (x%(p*c))//p, (x//(p*c))*p+x%p
def inverse(owner,local,p,c):
    if not 0 <= owner < c or local < 0 or p <= 0:
        raise ValueError("invalid local row")
    return (local//p)*p*c+owner*p+local%p
def prefix(a,b,namespace_a,namespace_b,p):
    if p <= 0:
        raise ValueError("page size")
    if namespace_a != namespace_b:
        return 0
    n=0
    for x,y in zip(a,b):
        if x != y:
            break
        n+=1
    return n//p*p
def selftest():
    assert placement(13,2,4)==(2,3)
    for x in range(128):
        assert inverse(*placement(x,2,4),2,4)==x
    assert prefix([11,12,13,91],[11,12,13,73],"v1","v1",2)==2
    assert prefix([1,2],[1,2],"v1","v2",2)==0
    assert prefix([],[],"v1","v1",2)==0
if __name__ == "__main__":
    selftest()
    print([placement(x,2,4) for x in range(16)])
