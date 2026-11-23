"""Exact Fraction toy for token-wise speculative rejection sampling.

This models a single draft-chain candidate. It is not EAGLE tree verification
or SGLang's full block-verification implementation.
"""
from fractions import Fraction
from typing import Sequence


def _validate_distribution(values: Sequence[Fraction]) -> None:
    if not values or any(v < 0 for v in values) or sum(values) != 1:
        raise ValueError("probabilities must be non-negative and sum exactly to 1")


def acceptance_probability(p_x: Fraction, q_x: Fraction) -> Fraction:
    if p_x < 0 or q_x <= 0:
        raise ValueError("candidate needs p(x) >= 0 and q(x) > 0")
    return min(Fraction(1), p_x / q_x)


def residual_distribution(
    target: Sequence[Fraction], draft: Sequence[Fraction]
) -> tuple[Fraction, ...]:
    _validate_distribution(target)
    _validate_distribution(draft)
    if len(target) != len(draft):
        raise ValueError("target and draft vocabularies must match")
    positive = [max(p - q, Fraction(0)) for p, q in zip(target, draft, strict=True)]
    total = sum(positive)
    if total == 0:
        raise ValueError("no residual distribution is needed when target equals draft")
    return tuple(value / total for value in positive)


def verify_candidate(
    target: Sequence[Fraction],
    draft: Sequence[Fraction],
    candidate_index: int,
    uniform: Fraction,
) -> bool:
    _validate_distribution(target)
    _validate_distribution(draft)
    if len(target) != len(draft):
        raise ValueError("target and draft vocabularies must match")
    if not 0 <= candidate_index < len(target):
        raise IndexError("candidate index out of range")
    if not 0 <= uniform < 1:
        raise ValueError("uniform variate must be in [0, 1)")
    alpha = acceptance_probability(target[candidate_index], draft[candidate_index])
    return uniform < alpha


def demo() -> None:
    p = (Fraction(3, 5), Fraction(2, 5))
    q = (Fraction(2, 5), Fraction(3, 5))
    candidate_b = 1
    alpha = acceptance_probability(p[candidate_b], q[candidate_b])
    accepted = verify_candidate(p, q, candidate_b, Fraction(4, 5))
    residual = residual_distribution(p, q)
    assert alpha == Fraction(2, 3)
    assert not accepted
    assert residual == (Fraction(1), Fraction(0))
    print(f"acceptance probability={alpha}; accepted={accepted}")
    print(f"residual distribution={residual}")


if __name__ == "__main__":
    demo()
