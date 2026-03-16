from collections import defaultdict
from typing import Dict, List
import math


def accuracy_by_layer(results: List[dict], letters: List[str]) -> List[Dict[str, float]]:
    if not results or not any(r.get("probs_per_layer") for r in results):
        return []
    n_layers = 0
    for r in results:
        ply = r.get("probs_per_layer") or []
        if isinstance(ply, list):
            n_layers = max(n_layers, len(ply))
    if n_layers == 0:
        return []
    by_layer = [defaultdict(lambda: {"correct": 0, "total": 0}) for _ in range(n_layers)]
    for r in results:
        ply = r.get("probs_per_layer") or []
        if not isinstance(ply, list):
            continue
        pos = r.get("correct_label")
        if not pos or pos not in letters:
            continue
        for li, layer_probs in enumerate(ply):
            if li >= n_layers:
                break
            if isinstance(layer_probs, dict):
                pred = max(letters, key=lambda k: layer_probs.get(k, 0))
                by_layer[li][pos]["total"] += 1
                if pred == pos:
                    by_layer[li][pos]["correct"] += 1
    out = []
    for li in range(n_layers):
        d = {}
        for L in letters:
            t = by_layer[li][L]["total"]
            d[L] = by_layer[li][L]["correct"] / t if t else 0.0
        out.append(d)
    return out


def prob_correct_by_layer(results: List[dict], letters: List[str]) -> List[Dict[str, float]]:
    if not results or not any(r.get("probs_per_layer") for r in results):
        return []
    n_layers = 0
    for r in results:
        ply = r.get("probs_per_layer") or []
        if isinstance(ply, list):
            n_layers = max(n_layers, len(ply))
    if n_layers == 0:
        return []
    by_layer = [defaultdict(list) for _ in range(n_layers)]
    for r in results:
        ply = r.get("probs_per_layer") or []
        if not isinstance(ply, list):
            continue
        pos = r.get("correct_label")
        for li, layer_probs in enumerate(ply):
            if li >= n_layers:
                break
            if isinstance(layer_probs, dict) and pos and pos in layer_probs:
                by_layer[li][pos].append(float(layer_probs[pos]))
    out = []
    for li in range(n_layers):
        d = {}
        for L in letters:
            lst = by_layer[li][L]
            d[L] = sum(lst) / len(lst) if lst else 0.0
        out.append(d)
    return out


def logit_difference_by_layer(
    results: List[dict],
    letters: List[str],
    anchor_pos: str = "A",
) -> List[Dict[str, float]]:
    """
    Logit difference by layer (ACL 2025 style): logit[anchor_pos] - logit[correct] per layer.
    Computed only for samples where correct != anchor_pos (failing cases).
    Positive values indicate bias toward anchor_pos at that layer.
    
    Uses probs_per_layer: converts probabilities to logits, then computes difference.
    Returns: [{layer_idx: diff_value}, ...] where diff_value is averaged across samples.
    """
    if not results or not any(r.get("probs_per_layer") for r in results):
        return []
    if anchor_pos not in letters:
        anchor_pos = letters[0] if letters else "A"
    
    # Find max layers
    n_layers = 0
    for r in results:
        ply = r.get("probs_per_layer") or []
        if isinstance(ply, list):
            n_layers = max(n_layers, len(ply))
    if n_layers == 0:
        return []
    
    # Collect logit differences per layer (only for samples where correct != anchor_pos)
    by_layer = [defaultdict(list) for _ in range(n_layers)]
    
    for r in results:
        correct = r.get("correct_label")
        if not correct or correct == anchor_pos or correct not in letters:
            continue  # Skip if correct is anchor_pos (not a failing case)
        
        ply = r.get("probs_per_layer") or []
        if not isinstance(ply, list):
            continue
        
        for li, layer_probs in enumerate(ply):
            if li >= n_layers or not isinstance(layer_probs, dict):
                continue
            
            prob_anchor = layer_probs.get(anchor_pos, 0.0)
            prob_correct = layer_probs.get(correct, 0.0)
            
            # Convert probs to logits (avoid log(0))
            eps = 1e-10
            logit_anchor = math.log(max(prob_anchor, eps))
            logit_correct = math.log(max(prob_correct, eps))
            
            diff = logit_anchor - logit_correct
            by_layer[li][correct].append(diff)
    
    # Average differences per layer
    out = []
    for li in range(n_layers):
        all_diffs = []
        for pos in letters:
            if pos != anchor_pos:
                all_diffs.extend(by_layer[li][pos])
        avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0.0
        out.append({"layer": li, "logit_diff_anchor_minus_correct": avg_diff})
    
    return out
