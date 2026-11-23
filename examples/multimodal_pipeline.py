"""Toy token-space multimodal expansion and feature-transport contract."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MediaItem:
    modality: str
    placeholder_id: int
    expansion: tuple[int, ...]


def expand_placeholders(input_ids, expansions, start_len=0):
    if not isinstance(start_len, int) or not 0 <= start_len <= len(input_ids):
        raise ValueError("start_len must be in [0, len(input_ids)]")
    ids = list(input_ids)
    positions = []
    for token_id, token_runs in expansions.items():
        hits = [i for i in range(start_len, len(ids)) if ids[i] == token_id]
        if len(hits) > len(token_runs) or (start_len == 0 and len(hits) != len(token_runs)):
            raise ValueError("placeholder count does not match media expansions")
        positions.extend(zip(hits, token_runs[len(token_runs)-len(hits):]))
    positions.sort(key=lambda item: item[0])
    out, cursor = ids[:start_len], start_len
    for pos, expansion in positions:
        out.extend(ids[cursor:pos])
        out.extend(expansion)
        cursor = pos + 1
    out.extend(ids[cursor:])
    return out


def transport_target(mode):
    if mode in ("cuda_ipc", "cuda_vmm"):
        return "gpu"
    if mode == "cpu":
        return "cpu"
    raise ValueError("transport must be cpu, cuda_ipc or cuda_vmm")


def token_count(items):
    return sum(len(item.expansion) for item in items)


if __name__ == "__main__":
    media = [
        MediaItem("image", 900, (101, 102, 103, 104)),
        MediaItem("video-frame", 901, (201, 202)),
        MediaItem("audio", 902, (301, 302, 303)),
    ]
    ids = [1, 900, 2, 901, 3, 902, 4]
    expanded = expand_placeholders(ids, {x.placeholder_id: [x.expansion] for x in media})
    print("placeholder IDs:", len(ids), "expanded IDs:", len(expanded))
    print("media feature tokens:", token_count(media))
    print("CUDA IPC feature target:", transport_target("cuda_ipc"))
