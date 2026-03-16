from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math


def overall_accuracy(results: List[dict]) -> float:
    """Fraction of results where pred == correct_label."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("correct") == 1)
    return correct / len(results)


def option_proportion(results: List[dict], letters: List[str]) -> Dict[str, float]:
    """Distribution of predicted labels (A/B/C/D) as fraction of total. For comparison with ground truth."""
    if not results:
        return {L: 0.0 for L in letters}
    by_pred = defaultdict(int)
    for r in results:
        p = r.get("pred")
        if p and p in letters:
            by_pred[p] += 1
    n = len(results)
    return {L: by_pred[L] / n if n else 0.0 for L in letters}


def ground_truth_position_proportion(results: List[dict], letters: List[str]) -> Dict[str, float]:
    """Distribution of correct answer position (A/B/C/D) in the data (should be ~equal if balanced)."""
    if not results:
        return {L: 0.0 for L in letters}
    by_pos = defaultdict(int)
    for r in results:
        pos = r.get("correct_label")
        if pos and pos in letters:
            by_pos[pos] += 1
    n = len(results)
    return {L: by_pos[L] / n if n else 0.0 for L in letters}


def accuracy_by_position(results: List[dict], letters: List[str]) -> Dict[str, float]:
    by_pos = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        pos = r.get("correct_label")
        if pos and pos in letters:
            by_pos[pos]["total"] += 1
            if r.get("correct") == 1:
                by_pos[pos]["correct"] += 1
    out = {}
    for L in letters:
        t = by_pos[L]["total"]
        out[L] = by_pos[L]["correct"] / t if t else 0.0
    return out


def sensitivity_gap(acc_by_pos: Dict[str, float]) -> float:
    if not acc_by_pos:
        return 0.0
    vals = list(acc_by_pos.values())
    return max(vals) - min(vals) if vals else 0.0


def prob_correct_by_position(results: List[dict], letters: List[str]) -> Dict[str, float]:
    by_pos = defaultdict(list)
    for r in results:
        pos = r.get("correct_label")
        p = r.get("prob_correct")
        if pos and pos in letters and p is not None:
            by_pos[pos].append(float(p))
    out = {}
    for L in letters:
        lst = by_pos[L]
        out[L] = sum(lst) / len(lst) if lst else 0.0
    return out


def error_prediction_proportion(results: List[dict], letters: List[str]) -> Dict[str, float]:
    """Among errors only: proportion of predictions that were A/B/C/D (anchored bias in errors)."""
    errors = [r for r in results if r.get("correct") != 1]
    if not errors:
        return {L: 0.0 for L in letters}
    by_pred = defaultdict(int)
    for r in errors:
        p = r.get("pred")
        if p and p in letters:
            by_pred[p] += 1
    n = len(errors)
    return {L: by_pred[L] / n if n else 0.0 for L in letters}


def error_correct_position_proportion(results: List[dict], letters: List[str]) -> Dict[str, float]:
    """Among errors only: proportion where the correct answer was at A/B/C/D (where the model fails most)."""
    errors = [r for r in results if r.get("correct") != 1]
    if not errors:
        return {L: 0.0 for L in letters}
    by_pos = defaultdict(int)
    for r in errors:
        pos = r.get("correct_label")
        if pos and pos in letters:
            by_pos[pos] += 1
    n = len(errors)
    return {L: by_pos[L] / n if n else 0.0 for L in letters}


def chi_square_option_proportion(results: List[dict], letters: List[str]) -> Dict[str, float]:
    """
    Chi-square test: Is predicted option proportion significantly different from uniform (25% each)?
    Returns: {'chi2': value, 'p_value': p, 'df': degrees_of_freedom, 'is_significant': bool}
    """
    prop = option_proportion(results, letters)
    n = len(results)
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "df": len(letters) - 1, "is_significant": False}
    expected = n / len(letters)
    observed = [prop.get(L, 0) * n for L in letters]
    chi2 = sum((obs - expected) ** 2 / expected for obs in observed)
    df = len(letters) - 1
    # Approximate p-value using chi-square critical values
    # For df=3: 7.815 (0.05), 11.345 (0.01), 16.27 (0.001)
    # For df=2: 5.991 (0.05), 9.210 (0.01)
    # For df=4: 9.488 (0.05), 13.277 (0.01)
    if df == 3:
        if chi2 > 16.27:
            p_value = 0.001
        elif chi2 > 11.345:
            p_value = 0.01
        elif chi2 > 7.815:
            p_value = 0.05
        elif chi2 > 4.108:  # ~0.25
            p_value = 0.25
        elif chi2 > 2.366:  # ~0.5
            p_value = 0.5
        else:
            p_value = 0.9  # High p-value for small chi2
    elif df == 2:
        if chi2 > 9.210:
            p_value = 0.01
        elif chi2 > 5.991:
            p_value = 0.05
        elif chi2 > 2.773:  # ~0.25
            p_value = 0.25
        elif chi2 > 1.386:  # ~0.5
            p_value = 0.5
        else:
            p_value = 0.9
    else:
        # Generic approximation
        if chi2 > 11.345:
            p_value = 0.01
        elif chi2 > 7.815:
            p_value = 0.05
        elif chi2 > 4.108:
            p_value = 0.25
        elif chi2 > 2.366:
            p_value = 0.5
        else:
            p_value = 0.9
    return {"chi2": chi2, "p_value": p_value, "df": df, "is_significant": chi2 > (7.815 if df == 3 else 5.991 if df == 2 else 9.488)}


def cohens_d_sensitivity_gap(acc_by_pos: Dict[str, float]) -> float:
    """
    Effect size (Cohen's d) for sensitivity gap.
    Compares accuracy at best vs worst position.
    Returns: Cohen's d (small: <0.2, medium: 0.2-0.8, large: >0.8)
    """
    if not acc_by_pos or len(acc_by_pos) < 2:
        return 0.0
    vals = list(acc_by_pos.values())
    max_val = max(vals)
    min_val = min(vals)
    if max_val == min_val:
        return 0.0
    # Cohen's d for proportions: use pooled standard deviation
    # For proportions, we approximate the pooled std as sqrt(pooled_p * (1 - pooled_p))
    # where pooled_p is the average of the two proportions being compared
    mean_acc = sum(vals) / len(vals)
    # Use the mean proportion as an approximation for pooled proportion
    # This is a simplified approach; ideally we'd have sample sizes per position
    pooled_std = math.sqrt(mean_acc * (1 - mean_acc)) if 0 < mean_acc < 1 else 0.1
    if pooled_std == 0 or pooled_std < 1e-10:
        return 0.0
    d = (max_val - min_val) / pooled_std
    return d


def accuracy_confidence_interval(results: List[dict], letters: List[str], confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Confidence intervals for accuracy by position (Wilson score interval for proportions).
    Returns: {position: {'lower': ..., 'upper': ..., 'mean': ...}}
    """
    # z-score for 95% confidence = 1.96, for 99% = 2.58
    z = 1.96 if confidence >= 0.95 else (2.58 if confidence >= 0.99 else 1.645)
    acc_by_pos = accuracy_by_position(results, letters)
    by_pos = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        pos = r.get("correct_label")
        if pos and pos in letters:
            by_pos[pos]["total"] += 1
            if r.get("correct") == 1:
                by_pos[pos]["correct"] += 1
    intervals = {}
    for L in letters:
        n = by_pos[L]["total"]
        if n == 0:
            intervals[L] = {"lower": 0.0, "upper": 0.0, "mean": 0.0}
            continue
        p = by_pos[L]["correct"] / n
        # Wilson score interval
        denominator = 1 + (z ** 2) / n
        centre = (p + (z ** 2) / (2 * n)) / denominator
        margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator
        intervals[L] = {
            "lower": max(0.0, centre - margin),
            "upper": min(1.0, centre + margin),
            "mean": p,
        }
    return intervals


def accuracy_position_significance_test(results: List[dict], letters: List[str], pos1: str, pos2: str) -> Dict[str, float]:
    """
    Test if accuracy at pos1 is significantly different from pos2 (two-proportion z-test).
    Returns: {'z_score': z, 'p_value': p, 'is_significant': bool}
    """
    by_pos = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        pos = r.get("correct_label")
        if pos and pos in letters:
            by_pos[pos]["total"] += 1
            if r.get("correct") == 1:
                by_pos[pos]["correct"] += 1
    n1, x1 = by_pos[pos1]["total"], by_pos[pos1]["correct"]
    n2, x2 = by_pos[pos2]["total"], by_pos[pos2]["correct"]
    if n1 == 0 or n2 == 0:
        return {"z_score": 0.0, "p_value": 1.0, "is_significant": False}
    p1 = x1 / n1
    p2 = x2 / n2
    p_pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z_score": 0.0, "p_value": 1.0, "is_significant": False}
    z = (p1 - p2) / se
    # Two-tailed p-value approximation using standard normal distribution
    abs_z = abs(z)
    if abs_z > 3.29:
        p_value = 0.001
    elif abs_z > 2.58:
        p_value = 0.01
    elif abs_z > 1.96:
        p_value = 0.05
    elif abs_z > 1.645:
        p_value = 0.10
    elif abs_z > 1.28:
        p_value = 0.20
    elif abs_z > 0.67:
        p_value = 0.50
    else:
        p_value = 0.90  # High p-value for small z-scores
    return {"z_score": z, "p_value": p_value, "is_significant": abs_z > 1.96}


def anchored_bias_frequency(results: List[dict], letters: List[str]) -> float:
    """
    Anchored bias frequency (ACL 2025): % of samples where model predicts 'A' (first position)
    but correct answer is NOT 'A'. This directly measures the "anchored bias" phenomenon.
    
    Returns: frequency (0.0 to 1.0), e.g., 0.974 means 97.4% of non-A samples are predicted as 'A'.
    """
    if not letters:
        return 0.0
    anchor_pos = letters[0]  # 'A'
    anchored_count = 0
    total_non_anchor = 0
    for r in results:
        correct = r.get("correct_label")
        pred = r.get("pred")
        if correct and correct != anchor_pos:  # correct is NOT 'A'
            total_non_anchor += 1
            if pred == anchor_pos:  # but model predicted 'A'
                anchored_count += 1
    return anchored_count / total_non_anchor if total_non_anchor > 0 else 0.0


def failing_cases_where_predicted_anchor(results: List[dict], letters: List[str]) -> List[dict]:
    """
    Filter to failing cases where model predicted anchor position ('A') but correct is not 'A'.
    This aligns with ACL 2025's focus on analyzing failing cases to understand bias.
    """
    if not letters:
        return []
    anchor_pos = letters[0]
    return [
        r for r in results
        if r.get("pred") == anchor_pos
        and r.get("correct_label") != anchor_pos
        and r.get("correct_label") in letters
    ]
