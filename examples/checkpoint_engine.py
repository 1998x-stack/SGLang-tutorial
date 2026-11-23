"""CPU topology/readiness model for checkpoint distribution; no IPC or network is used."""
from dataclasses import dataclass


def split_shards(shards, worker_count):
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    assignments = [[] for _ in range(worker_count)]
    for index, shard in enumerate(sorted(shards)):
        assignments[index % worker_count].append(shard)
    return assignments


def transfer_plan(method, loader_ranks, inference_ranks):
    if method not in {"broadcast", "p2p", "all"}:
        raise ValueError("method must be broadcast, p2p or all")
    if loader_ranks < 1 or inference_ranks < 1:
        raise ValueError("rank counts must be positive")
    plan = []
    if method in {"broadcast", "all"}:
        plan.append(("broadcast", loader_ranks, inference_ranks))
    if method in {"p2p", "all"}:
        plan.append(("p2p", loader_ranks, inference_ranks))
    return plan


@dataclass
class WeightReadiness:
    expected_ranks: int
    admin_key: str
    received: dict = None

    def __post_init__(self):
        if self.expected_ranks < 1 or not self.admin_key:
            raise ValueError("expected ranks and admin key are required")
        self.received = {}

    def accept_rank(self, rank, payload, supplied_key):
        if supplied_key != self.admin_key:
            raise PermissionError("invalid admin key")
        if not 0 <= rank < self.expected_ranks:
            raise ValueError("rank out of range")
        if not payload:
            raise ValueError("empty weight payload")
        self.received[rank] = payload

    @property
    def ready(self):
        return len(self.received) == self.expected_ranks


if __name__ == "__main__":
    assignments = split_shards(["shard-0", "shard-1", "shard-2", "shard-3"], 2)
    readiness = WeightReadiness(expected_ranks=2, admin_key="local-test")
    readiness.accept_rank(0, "weights-A", "local-test")
    print("ready after rank 0:", readiness.ready)
    readiness.accept_rank(1, "weights-B", "local-test")
    print("ready after rank 1:", readiness.ready)
    print("worker shard assignments:", assignments)
