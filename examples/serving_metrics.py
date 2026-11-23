"""Deterministic request-level serving metrics; no server or GPU is involved."""
from math import ceil


def request_metrics(arrival_s, first_token_s, finish_s, output_tokens):
    if not (arrival_s <= first_token_s <= finish_s) or output_tokens < 1:
        raise ValueError("timestamps must be ordered and output_tokens >= 1")
    ttft = first_token_s - arrival_s
    e2e = finish_s - arrival_s
    tpot = (finish_s - first_token_s) / (output_tokens - 1) if output_tokens > 1 else None
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e}


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("values must be non-empty and q in [0,1]")
    ordered = sorted(values)
    return ordered[max(0, ceil(q * len(ordered)) - 1)]


def throughput(total_tokens, elapsed_s):
    if total_tokens < 0 or elapsed_s <= 0:
        raise ValueError("invalid token count or elapsed time")
    return total_tokens / elapsed_s


def benchmark_layer(tool):
    layers = {
        "bench_serving": ("http", "server+scheduler", "online TTFT/TPOT/ITL"),
        "bench_one_batch_server": ("http", "server+scheduler", "single-batch E2E"),
        "bench_offline_throughput": ("none", "Engine", "offline throughput"),
        "bench_one_batch": ("none", "ModelRunner", "kernel/static batch"),
    }
    if tool not in layers:
        raise ValueError("unknown benchmark tool")
    return layers[tool]


if __name__ == "__main__":
    samples = [
        request_metrics(0.0, 0.25, 1.25, 6),
        request_metrics(0.1, 0.40, 1.10, 4),
        request_metrics(0.2, 0.70, 1.50, 5),
    ]
    print("TTFT p95:", percentile([x["ttft"] for x in samples], 0.95))
    print("request-level TPOT p95:", percentile([x["tpot"] for x in samples if x["tpot"] is not None], 0.95))
    print("decode throughput:", round(15 / 1.5, 2), "tokens/s")
    print("bench_serving:", benchmark_layer("bench_serving"))
