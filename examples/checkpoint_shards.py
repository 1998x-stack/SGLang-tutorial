"""Toy checkpoint index and equal output-axis partition; not a safetensors reader."""
from math import prod
def nbytes(shape,itemsize):
    if itemsize<=0 or any(d<0 for d in shape): raise ValueError("shape")
    return prod(shape)*itemsize
def partition(rows,cols,ranks):
    if ranks<=0 or rows<=0 or cols<=0 or rows%ranks: raise ValueError("equal partition")
    d=rows//ranks
    return [(r*d,(r+1)*d,cols) for r in range(ranks)]
def validate_index(weight_map,files,expected):
    missing=set(expected)-set(weight_map)
    absent={f for f in weight_map.values() if f not in files}
    return missing,absent
def selftest():
    assert nbytes([8,4],2)==64
    p=partition(8,4,2)
    assert p==[(0,4,4),(4,8,4)]
    assert sum(nbytes([b-a,c],2) for a,b,c in p)==64
    assert validate_index({"W":"part-02"},{"part-02"},{"W"})==(set(),set())
    assert validate_index({"W":"part-02"},set(),{"W","bias"})==({"bias"},{"part-02"})
    try: partition(7,4,2)
    except ValueError: pass
    else: raise AssertionError("shape")
if __name__=="__main__":
    selftest()
    print(partition(8,4,2))
