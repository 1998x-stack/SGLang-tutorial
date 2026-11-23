"""Float32 reduction-order and per-request RNG teaching model; no GPU involved."""
import random
import struct


def f32(value):
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def reduce_float32(values, order):
    acc = f32(0.0)
    for index in order:
        acc = f32(acc + f32(values[index]))
    return acc


def batch_invariant_sum(values):
    return reduce_float32(values, range(len(values)))


def seeded_choice(candidates, seed):
    return random.Random(seed).choice(tuple(candidates))


if __name__ == "__main__":
    values = [1e20, -1e20, 3.14]
    left = reduce_float32(values, [0, 1, 2])
    right = reduce_float32(values, [1, 2, 0])
    print("different reduction orders:", left, right)
    print("fixed order:", batch_invariant_sum(values))
    print("same request seed:", seeded_choice("ABC", 42), seeded_choice("ABC", 42))
