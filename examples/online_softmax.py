"""Numerical CPU demonstration of tiled online softmax accumulation."""
from math import exp, isclose


def online_weighted_sum(scores: list[float], values: list[float]) -> float:
    if not scores or len(scores) != len(values):
        raise ValueError("scores and values must be non-empty and equally sized")
    running_max = float("-inf")
    running_sum = 0.0
    running_output = 0.0
    for score, value in zip(scores, values, strict=True):
        new_max = max(running_max, score)
        old_scale = 0.0 if running_max == float("-inf") else exp(running_max - new_max)
        tile_scale = exp(score - new_max)
        running_output = old_scale * running_output + tile_scale * value
        running_sum = old_scale * running_sum + tile_scale
        running_max = new_max
    return running_output / running_sum


def full_weighted_sum(scores: list[float], values: list[float]) -> float:
    if not scores or len(scores) != len(values):
        raise ValueError("scores and values must be non-empty and equally sized")
    maximum = max(scores)
    weights = [exp(score - maximum) for score in scores]
    return sum(w * v for w, v in zip(weights, values, strict=True)) / sum(weights)


def demo() -> None:
    scores = [2.0, -1.0, 3.5, 0.25]
    values = [10.0, 20.0, -4.0, 7.0]
    tiled = online_weighted_sum(scores, values)
    full = full_weighted_sum(scores, values)
    assert isclose(tiled, full, rel_tol=1e-12, abs_tol=1e-12)
    print(f"online={tiled:.12f}; full={full:.12f}")


if __name__ == "__main__":
    demo()
