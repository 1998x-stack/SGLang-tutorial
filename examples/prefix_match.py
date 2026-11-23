"""CPU-only prefix cache teaching model, not production SGLang implementation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Namespace:
    model: str
    tokenizer: str
    adapter: str = ""
    tenant: str = ""


@dataclass(frozen=True)
class RequestKey:
    tokens: tuple[int, ...]
    namespace: Namespace


def longest_common_prefix(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Number of equal token IDs before the first mismatch."""
    result = 0
    for x, y in zip(a, b):
        if x != y:
            break
        result += 1
    return result


def reusable_prefix(a: RequestKey, b: RequestKey, page_size: int = 1) -> int:
    """Toy example: require same namespace and complete pages.

    Production code additionally depends on KV residency, cache validity and
    model/runtime-specific conditions. This function allocates no GPU memory.
    """
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if a.namespace != b.namespace:
        return 0
    common = longest_common_prefix(a.tokens, b.tokens)
    return (common // page_size) * page_size


def demo() -> None:
    ns = Namespace("toy-model", "toy-tokenizer", tenant="A")
    base = RequestKey((11, 12, 13, 91, 92), ns)
    candidates = [
        RequestKey((11, 12, 13, 91, 92), ns),
        RequestKey((11, 12, 13, 73), ns),
        RequestKey((42, 12, 13), ns),
    ]
    results = [reusable_prefix(base, c) for c in candidates]
    assert results == [5, 3, 0]
    assert reusable_prefix(base, candidates[1], page_size=2) == 2
    for candidate, match in zip(candidates, results):
        print(f"candidate={candidate.tokens}; prefix_len={match}")


if __name__ == "__main__":
    demo()
