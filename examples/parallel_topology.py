"""Toy rank-group calculator; it does not create torch.distributed groups."""
def model_parallel_groups(world_size, tp_size, pp_size):
    if min(world_size, tp_size, pp_size) < 1 or tp_size * pp_size != world_size:
        raise ValueError("require positive widths and world_size == tp_size * pp_size")
    tp = tuple(tuple(stage * tp_size + r for r in range(tp_size)) for stage in range(pp_size))
    pp = tuple(tuple(stage * tp_size + r for stage in range(pp_size)) for r in range(tp_size))
    return {"tp": tp, "pp": pp}

def attention_tp_width(tp_size, attn_dp_size, attn_cp_size):
    divisor = attn_dp_size * attn_cp_size
    if min(tp_size, attn_dp_size, attn_cp_size) < 1 or tp_size % divisor:
        raise ValueError("attention DP × CP must divide TP size")
    return tp_size // divisor

def moe_tp_width(tp_size, moe_ep_size, moe_dp_size):
    divisor = moe_ep_size * moe_dp_size
    if min(tp_size, moe_ep_size, moe_dp_size) < 1 or tp_size % divisor:
        raise ValueError("MoE EP × DP must divide TP size in this toy layout")
    return tp_size // divisor

if __name__ == "__main__":
    g = model_parallel_groups(8, 2, 4)
    print("TP:", g["tp"]); print("PP:", g["pp"])
    print("Attention TP width:", attention_tp_width(8, 1, 2))
    print("MoE TP width:", moe_tp_width(8, 4, 2))
