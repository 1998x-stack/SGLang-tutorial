"""CPU teaching model for per-batch adaptive speculative steps; not SGLang runtime code."""
from dataclasses import dataclass, field


@dataclass
class AdaptiveSlot:
    candidate_steps: tuple = (1, 3, 5, 7)
    current_steps: int = 3
    alpha: float = 0.5
    warmup_batches: int = 0
    update_interval: int = 1
    up_hysteresis: float = 0.0
    down_hysteresis: float = -0.25
    ema_accept_len: float = field(init=False)
    batches: int = 0

    def __post_init__(self):
        self.candidate_steps = tuple(sorted(set(self.candidate_steps)))
        if not self.candidate_steps or self.current_steps not in self.candidate_steps:
            raise ValueError("current_steps must be one of candidate_steps")
        if not 0 < self.alpha <= 1 or self.update_interval < 1:
            raise ValueError("invalid EMA or update interval")
        self.ema_accept_len = float(self.current_steps - 1)

    def update(self, accepted_per_request):
        if not accepted_per_request:
            return False
        if any(x < 0 for x in accepted_per_request):
            raise ValueError("accepted lengths cannot be negative")
        average = sum(accepted_per_request) / len(accepted_per_request)
        self.ema_accept_len = (1 - self.alpha) * self.ema_accept_len + self.alpha * average
        self.batches += 1
        if self.batches <= self.warmup_batches:
            return False
        if (self.batches - self.warmup_batches) % self.update_interval:
            return False
        old = self.current_steps
        idx = self.candidate_steps.index(old)
        while idx > 0:
            previous = self.candidate_steps[idx - 1]
            threshold = (0.5 if previous == 0 else previous - 0.5) + self.down_hysteresis
            if self.ema_accept_len <= threshold:
                idx -= 1
            else:
                break
        if idx == self.candidate_steps.index(old):
            while idx < len(self.candidate_steps) - 1:
                current = self.candidate_steps[idx]
                if self.ema_accept_len > current - 0.5 + self.up_hysteresis:
                    idx += 1
                else:
                    break
        self.current_steps = self.candidate_steps[idx]
        return self.current_steps != old


class AdaptivePolicy:
    def __init__(self, slots):
        self.slots = dict(sorted(slots.items()))
        if not self.slots or min(self.slots) < 1:
            raise ValueError("batch-size slot keys must be positive")

    def slot_for(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        eligible = [key for key in self.slots if key <= batch_size]
        return self.slots[max(eligible or [min(self.slots)])]


class RuntimeStatePool:
    def __init__(self, steps):
        self.states = {step: {"steps": step, "graph": f"graph-{step}"} for step in set(steps)}
        self.active = None
        self.swaps = 0

    def activate(self, step):
        if step not in self.states:
            raise KeyError("runtime state must be prebuilt")
        if self.active != step:
            self.active = step
            self.swaps += 1


if __name__ == "__main__":
    policy = AdaptivePolicy({
        1: AdaptiveSlot(candidate_steps=(1, 3, 5, 7), current_steps=3, alpha=1),
        8: AdaptiveSlot(candidate_steps=(1, 3), current_steps=1, alpha=1),
    })
    small = policy.slot_for(2)
    small.update([4, 4, 4])
    pool = RuntimeStatePool([1, 3, 5, 7])
    pool.activate(small.current_steps)
    print("small batch selected steps:", small.current_steps, "state swaps:", pool.swaps)
