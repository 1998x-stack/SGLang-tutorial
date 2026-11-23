"""Type-separated Mooncake teaching model; no network or Mooncake client is used."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ObjectRef:
    plane: str
    namespace: str
    key: str
    payload_type: str
class PlaneStore:
    def __init__(self): self.objects={}
    def put(self, ref, value):
        if ref.plane not in {"pd-transfer","hicache-l3","epd-embedding"}: raise ValueError("unknown plane")
        self.objects[(ref.plane,ref.namespace,ref.key,ref.payload_type)]=value
    def get(self, ref): return self.objects.get((ref.plane,ref.namespace,ref.key,ref.payload_type))
if __name__=="__main__":
    s=PlaneStore()
    s.put(ObjectRef("pd-transfer","req-7","kv","language-kv"),"decode landing buffer")
    s.put(ObjectRef("hicache-l3","model-v1","prefix-ab","kv-pages"),"reusable KV pages")
    s.put(ObjectRef("epd-embedding","processor-v3","image-abc","vit-embedding"),"encoder embedding")
    print("objects:",len(s.objects))
