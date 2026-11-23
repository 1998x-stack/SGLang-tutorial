"""Toy communication/compute overlap and expert skew model; not a GPU benchmark."""
def mode_for_phase(phase):
    if phase=="prefill": return "normal"
    if phase=="decode": return "low_latency"
    return "auto"
def overlapped_time(comm_ms, compute_ms, overlap_ms):
    if min(comm_ms,compute_ms,overlap_ms)<0 or overlap_ms>min(comm_ms,compute_ms):
        raise ValueError("overlap must be between zero and min(comm, compute)")
    return comm_ms+compute_ms-overlap_ms
def skew(loads):
    if not loads or any(x<0 for x in loads): raise ValueError("loads must be non-negative and non-empty")
    mean=sum(loads)/len(loads)
    return max(loads)/mean if mean else 0.0
if __name__=="__main__":
    print("modes:",mode_for_phase("prefill"),mode_for_phase("decode"),mode_for_phase("other"))
    print("toy layer ms:",overlapped_time(4,7,3))
    print("expert max/mean:",skew([10,12,8,30]))
