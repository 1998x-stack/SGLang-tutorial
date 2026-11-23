"""CPU-only teaching model of request rows and paged physical KV slots.

This is deliberately smaller than SGLang's CUDA allocator. It demonstrates
only the index contract: logical token position -> physical slot ID.
"""
from dataclasses import dataclass, field


@dataclass
class RequestRow:
    row_id: int
    slots: list[int] = field(default_factory=list)
    page_ids: list[int] = field(default_factory=list)


class ToyPagedAllocator:
    def __init__(self, page_size: int = 4, free_pages=(2, 4, 7, 9), rows: int = 8):
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.free_pages = list(sorted(free_pages))
        self.free_rows = list(range(1, rows + 1))
        self.requests: dict[str, RequestRow] = {}

    def create_request(self, request_id: str) -> int:
        if request_id in self.requests:
            raise ValueError(f"duplicate request: {request_id}")
        if not self.free_rows:
            raise MemoryError("no free request rows")
        row_id = self.free_rows.pop(0)
        self.requests[request_id] = RequestRow(row_id=row_id)
        return row_id

    def append(self, request_id: str, token_count: int) -> list[int]:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        row = self.requests[request_id]
        added = []
        for _ in range(token_count):
            logical_pos = len(row.slots)
            offset = logical_pos % self.page_size
            if offset == 0:
                if not self.free_pages:
                    raise MemoryError("KV page pool exhausted")
                page_id = self.free_pages.pop(0)
                row.page_ids.append(page_id)
            else:
                page_id = row.page_ids[-1]
            slot_id = page_id * self.page_size + offset
            row.slots.append(slot_id)
            added.append(slot_id)
        return added

    def release(self, request_id: str) -> list[int]:
        row = self.requests.pop(request_id)
        self.free_rows.append(row.row_id)
        self.free_rows.sort()
        self.free_pages.extend(row.page_ids)
        self.free_pages.sort()
        return list(row.page_ids)

    def row(self, request_id: str) -> RequestRow:
        return self.requests[request_id]


def demo() -> None:
    pool = ToyPagedAllocator(page_size=4)
    row_id = pool.create_request("A")
    first = pool.append("A", 6)
    second = pool.append("A", 2)
    assert row_id == 1
    assert first == [8, 9, 10, 11, 16, 17]
    assert second == [18, 19]
    print(f"request A row={row_id}, token slots={pool.row('A').slots}")
    print(f"request A pages={pool.row('A').page_ids}")
    released = pool.release("A")
    assert released == [2, 4]
    assert pool.free_pages == [2, 4, 7, 9]
    print(f"released pages={released}, free pages={pool.free_pages}")


if __name__ == "__main__":
    demo()
