"""Toy block quantization and KV storage accounting; not an NVFP4 implementation."""
from math import ceil


def bf16_bytes(values):
    if values < 0:
        raise ValueError("values must be non-negative")
    return values * 2


def fp8_bytes(values, scale_bytes=4):
    if values < 0:
        raise ValueError("values must be non-negative")
    return values + scale_bytes


def nvfp4_bytes(values, block_size=16, global_scale_bytes=4):
    if values < 0 or block_size <= 0:
        raise ValueError("invalid values or block size")
    return ceil(values / 2) + ceil(values / block_size) + global_scale_bytes


def block_quantize(values, block_size=16, levels=7):
    if not values or block_size <= 0 or levels <= 0:
        raise ValueError("values, block size and levels must be positive")
    restored = []
    scales = []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        scale = max(abs(x) for x in block) / levels or 1.0
        scales.append(scale)
        restored.extend(max(-levels, min(levels, round(x / scale))) * scale for x in block)
    return restored, scales


def capacity_multiplier(reference_bytes, compressed_bytes):
    if reference_bytes <= 0 or compressed_bytes <= 0:
        raise ValueError("byte costs must be positive")
    return reference_bytes / compressed_bytes


def attention_access(backend_supports_fused, workspace_available):
    if backend_supports_fused:
        return "fused"
    if workspace_available:
        return "dequant-workspace"
    raise RuntimeError("backend cannot consume quantized KV and no workspace is available")


if __name__ == "__main__":
    values = [0.1, -0.4, 0.7, 1.2, -0.2, 0.3, -0.8, 0.5] * 4
    restored, scales = block_quantize(values)
    print("blocks:", len(scales), "max abs error:", round(max(abs(a-b) for a,b in zip(values, restored)), 5))
    print("BF16 / FP8 / NVFP4 bytes:", bf16_bytes(1024), fp8_bytes(1024), nvfp4_bytes(1024))
    print("theoretical BF16-to-NVFP4 multiplier:", round(capacity_multiplier(2, 0.5 + 1/16), 3))
    print("attention path:", attention_access(False, True))
