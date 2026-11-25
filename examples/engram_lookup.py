"""Toy hash IDs and equal-range table placement, not DeepSeek Engram."""
def hash_row(ids,multipliers,p,offset):
    if len(ids)!=len(multipliers) or len(ids)<2 or p<=0 or min(ids)<0 or offset<0: raise ValueError("hash")
    h=0
    for u,m in zip(ids,multipliers): h^=u*m
    return offset+h%p
def owner(row,total,ranks):
    if ranks<=0 or total<=0 or total%ranks or not 0<=row<total: raise ValueError("equal shard")
    width=total//ranks
    return row//width,row%width
def lookup(shards,row):
    o,i=owner(row,sum(map(len,shards)),len(shards))
    if len(set(map(len,shards)))!=1: raise ValueError("unequal shard")
    return shards[o][i]
def selftest():
    assert hash_row([3,5],[7,11],13,100)==108
    assert owner(6,8,4)==(3,0)
    assert lookup([[0,1],[2,3],[4,5],[6,7]],6)==6
    try: owner(8,8,4)
    except ValueError: pass
    else: raise AssertionError("out of range")
if __name__=="__main__":
    selftest()
    print(hash_row([3,5],[7,11],13,100))
