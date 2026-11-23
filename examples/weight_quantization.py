"""Toy symmetric per-row weight quantization; not AWQ/GPTQ or a SGLang kernel."""
from math import ceil


def quantize_row(values, bits=4):
    if bits not in (4, 8):
        raise ValueError("the teaching example supports 4 or 8 bits")
    if not values:
        raise ValueError("row must not be empty")
    qmax = (1 << (bits - 1)) - 1
    scale = max(abs(x) for x in values) / qmax or 1.0
    quantized = [max(-qmax, min(qmax, round(x / scale))) for x in values]
    restored = [q * scale for q in quantized]
    return quantized, scale, restored


def quantize_matrix_per_row(matrix, bits=4):
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("matrix must be non-empty and rectangular")
    rows = [quantize_row(row, bits) for row in matrix]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def estimate_packed_bytes(num_values, bits, group_size=128, scale_bytes=4):
    if num_values < 0 or bits not in (4, 8) or group_size <= 0:
        raise ValueError("invalid storage dimensions")
    return ceil(num_values * bits / 8) + ceil(num_values / group_size) * scale_bytes


def mse(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("vectors must have equal non-zero length")
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


if __name__ == "__main__":
    weights = [[0.1, -0.7, 1.2, 0.4], [3.0, -1.5, 0.25, -0.5]]
    q4, s4, r4 = quantize_matrix_per_row(weights, 4)
    q8, s8, r8 = quantize_matrix_per_row(weights, 8)
    print("W4 row scales:", [round(x, 5) for x in s4])
    print("W4 MSE:", round(mse(sum(weights, []), sum(r4, [])), 8))
    print("W8 MSE:", round(mse(sum(weights, []), sum(r8, [])), 10))
    print("packed bytes for 4096 values, W4/W8:", estimate_packed_bytes(4096, 4), estimate_packed_bytes(4096, 8))
