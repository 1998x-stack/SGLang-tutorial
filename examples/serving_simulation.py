"""Toy FCFS logical clock; not SGLang's continuous-batching simulator."""
def simulate(requests):
    now=0; result=[]
    for arrival,prefill,n,itl in sorted(requests,key=lambda r:r[0]):
        if min(arrival,prefill,itl)<0 or n<1: raise ValueError("request")
        start=max(now,arrival); first=start+prefill; last=first+(n-1)*itl
        result.append({"queue":start-arrival,"ttft":first-arrival,"tpot":(last-first)/(n-1) if n>1 else None,"last":last})
        now=last
    return result
def selftest():
    a,b=simulate([(0,4,3,2),(3,2,2,2)])
    assert a=={"queue":0,"ttft":4,"tpot":2,"last":8}
    assert b=={"queue":5,"ttft":7,"tpot":2,"last":12}
    assert simulate([(0,1,1,2)])[0]["tpot"] is None
    assert simulate([])==[]
    assert simulate([(0,6,3,2),(3,2,2,2)])[1]["ttft"]==9
if __name__=="__main__":
    selftest()
    print(simulate([(0,4,3,2),(3,2,2,2)]))
