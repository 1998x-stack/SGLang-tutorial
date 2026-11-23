"""Toy EPD stage-capacity model; not a serving benchmark."""
from dataclasses import dataclass
@dataclass(frozen=True)
class StageCapacity:
    encoder_per_worker: float; encoder_workers: int
    prefill_per_worker: float; prefill_workers: int
    decode_per_worker: float; decode_workers: int
    def totals(self):
        return {"encoder":self.encoder_per_worker*self.encoder_workers,"prefill":self.prefill_per_worker*self.prefill_workers,"decode":self.decode_per_worker*self.decode_workers}
    def bottleneck(self):
        totals=self.totals(); name=min(totals,key=totals.get); return name,totals[name]
def end_to_end_latency_ms(encoder,embedding_transfer,prefill,kv_transfer,decode,queue=0):
    values=(encoder,embedding_transfer,prefill,kv_transfer,decode,queue)
    if any(x<0 for x in values): raise ValueError("latencies must be non-negative")
    return sum(values)
if __name__=="__main__":
    s=StageCapacity(8,1,10,2,3,4); print("capacities:",s.totals()); print("bottleneck:",s.bottleneck())
    print("toy e2e ms:",end_to_end_latency_ms(40,5,70,8,120,12))
