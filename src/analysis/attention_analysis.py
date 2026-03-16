from collections import defaultdict
from typing import Dict, List


def mean_attn_by_position(
    results: List[dict],
    letters: List[str],
    layer: int = -1,
) -> Dict[str, Dict[str, float]]:
    by_pos = defaultdict(lambda: defaultdict(list))
    for r in results:
        pos = r.get("correct_label")
        attn = r.get("attn_to_options")
        if not pos or pos not in letters or not attn:
            continue
        if isinstance(attn, list) and attn:
            a = attn[layer] if layer >= 0 else attn[-1]
        elif isinstance(attn, dict):
            a = attn
        else:
            continue
        if isinstance(a, dict):
            for L in letters:
                if L in a:
                    by_pos[pos][L].append(float(a[L]))
    out = {}
    for pos in letters:
        out[pos] = {}
        for L in letters:
            lst = by_pos[pos][L]
            out[pos][L] = sum(lst) / len(lst) if lst else 0.0
    return out


def attn_to_correct_vs_others(
    results: List[dict],
    letters: List[str],
    layer: int = -1,
) -> Dict[str, float]:
    by_pos = defaultdict(lambda: {"correct": [], "others": []})
    for r in results:
        pos = r.get("correct_label")
        attn = r.get("attn_to_options")
        if not pos or pos not in letters or not attn:
            continue
        if isinstance(attn, list) and attn:
            a = attn[layer] if layer >= 0 else attn[-1]
        elif isinstance(attn, dict):
            a = attn
        else:
            continue
        if isinstance(a, dict):
            correct_val = a.get(pos)
            others_val = sum(a.get(L, 0) for L in letters if L != pos) / max(1, len(letters) - 1)
            if correct_val is not None:
                by_pos[pos]["correct"].append(float(correct_val))
                by_pos[pos]["others"].append(float(others_val))
    out = {}
    for L in letters:
        c = sum(by_pos[L]["correct"]) / len(by_pos[L]["correct"]) if by_pos[L]["correct"] else 0.0
        o = sum(by_pos[L]["others"]) / len(by_pos[L]["others"]) if by_pos[L]["others"] else 0.0
        out[L] = c - o
    return out
