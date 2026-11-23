"""Deterministic CPU model of SGLang router policy decisions; not the Rust router."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Worker:
    name: str
    stage: str
    prefix_tokens: int = 0
    pending_prefill_tokens: int = 0
    waiting: int = 0
    running: int = 0
    kv_usage: float = 0.0
    inflight: int = 0

def admitted(w, max_waiting=8, max_pending=8192, max_kv=0.95, max_inflight=64):
    return w.waiting < max_waiting and w.pending_prefill_tokens < max_pending and w.kv_usage < max_kv and w.inflight < max_inflight

def load_key(w):
    if w.stage in ("prefill", "plain"):
        return (w.pending_prefill_tokens, w.waiting, w.running, w.inflight)
    return (w.waiting, w.running, w.kv_usage, w.inflight)

def power_of_two(workers):
    eligible=[w for w in workers if admitted(w)]
    if not eligible: return None
    # Deterministic two-candidate sample for reproducible teaching; production sampling is randomized.
    sampled=sorted(eligible,key=lambda w:w.name)[:2]
    return min(sampled,key=load_key).name

def cache_aware(workers, min_prefix=1024, mode="prefer", factor=2.0, gap=1024):
    eligible=[w for w in workers if admitted(w)]
    if not eligible: return None
    owner=max(eligible,key=lambda w:w.prefix_tokens)
    if owner.prefix_tokens < min_prefix: return power_of_two(eligible)
    alternative=min((w for w in eligible if w.name!=owner.name),key=load_key,default=None)
    if mode=="balanced" and alternative is not None:
        a=owner.pending_prefill_tokens; b=alternative.pending_prefill_tokens
        if a > factor*b and a-b > gap: return alternative.name
    return owner.name

if __name__=="__main__":
    workers=[Worker("A","prefill",2048,9000),Worker("B","prefill",512,400),Worker("C","prefill",0,900)]
    print("power_of_two:",power_of_two(workers))
    print("cache_aware:",cache_aware(workers))
    print("balanced:",cache_aware(workers,mode="balanced"))
