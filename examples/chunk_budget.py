"""CPU-only toy model for budget-limited chunk sizing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkBudget:
    remaining_input: int
    token_budget: int
    memory_budget: int
    policy_cap: int | None = None
    alignment: int = 1


def choose_chunk(budget: ChunkBudget) -> int:
    values = (budget.remaining_input, budget.token_budget, budget.memory_budget)
    if any(value < 0 for value in values):
        raise ValueError("budgets cannot be negative")
    if budget.policy_cap is not None and budget.policy_cap < 0:
        raise ValueError("policy_cap cannot be negative")
    if budget.alignment < 1:
        raise ValueError("alignment must be positive")
    limit = min(*values)
    if budget.policy_cap is not None:
        limit = min(limit, budget.policy_cap)
    # A non-final cut may be rounded down. If rounding would make no progress,
    # this toy model waits rather than inventing an unaligned chunk.
    if limit < budget.remaining_input and budget.alignment > 1:
        limit = limit // budget.alignment * budget.alignment
    return limit


def demo() -> None:
    rounds = [
        ChunkBudget(remaining_input=12, token_budget=6, memory_budget=4, policy_cap=8),
        ChunkBudget(remaining_input=8, token_budget=5, memory_budget=3, policy_cap=8),
        ChunkBudget(remaining_input=5, token_budget=8, memory_budget=6, policy_cap=8),
    ]
    sizes = [choose_chunk(item) for item in rounds]
    assert sizes == [4, 3, 5]
    print("toy chunk sizes:", sizes)
    aligned = choose_chunk(ChunkBudget(10, 7, 7, alignment=4))
    assert aligned == 4
    print("aligned non-final chunk:", aligned)


if __name__ == "__main__":
    demo()
