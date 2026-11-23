"""CPU model of CUDA-graph shape keys, static buffers and breakable segments."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphKey:
    shape: tuple[int, ...]
    segmentation: tuple[int, ...]


class GraphCache:
    def __init__(self):
        self._captures = {}

    def run(self, key: GraphKey, static_buffer_id: str):
        if not key.shape or any(x <= 0 for x in key.shape):
            raise ValueError("graph shape dimensions must be positive")
        if not static_buffer_id:
            raise ValueError("a stable static-buffer identity is required")
        if key not in self._captures:
            self._captures[key] = static_buffer_id
            return "capture"
        if self._captures[key] != static_buffer_id:
            raise RuntimeError("replay must reuse the capture-time static buffer")
        return "replay"

    @property
    def graph_count(self):
        return len(self._captures)


def replay_segments(segments, eager_ops):
    if len(segments) != len(eager_ops) + 1:
        raise ValueError("N captured segments require N-1 eager operations")
    events = []
    for i, segment in enumerate(segments):
        events.append("graph:" + segment)
        if i < len(eager_ops):
            events.append("eager:" + eager_ops[i])
    return events


if __name__ == "__main__":
    cache = GraphCache()
    key = GraphKey((8, 4096), (0, 1024, 4096))
    print(cache.run(key, "static-buffer-A"), cache.run(key, "static-buffer-A"))
    print(cache.run(GraphKey((16, 4096), (0, 2048, 4096)), "static-buffer-B"))
    print("breakable replay:", replay_segments(["attention-1", "attention-2"], ["dynamic-op"]))
    print("captured graphs:", cache.graph_count)
