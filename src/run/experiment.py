import math
from typing import Any, Dict, List

from tqdm.auto import tqdm

from ..data.shuffle import PermutedItem
from ..models.hf_model import HFModel
from ..models.openai_model import OpenAIModel


def run_openai(
    model: OpenAIModel,
    perm: PermutedItem,
    template: str,
    letters: List[str],
) -> Dict[str, Any]:
    prompt = template.format(question=perm.question, **perm.options)
    pred, logprobs = model.predict(prompt, letters)
    correct = 1 if pred == perm.correct_label else 0
    lp_correct = logprobs.get(perm.correct_label)
    prob_correct = math.exp(lp_correct) if lp_correct is not None else None
    return {
        "item_id": perm.item_id,
        "correct_label": perm.correct_label,
        "correct_idx": perm.correct_idx,
        "pred": pred,
        "correct": correct,
        "prob_correct": prob_correct,
        "logprobs": logprobs,
        "attn_to_options": None,
        "probs_per_layer": None,
    }


def run_hf(
    model: HFModel,
    perm: PermutedItem,
    template: str,
    letters: List[str],
    with_internals: bool = True,
) -> Dict[str, Any]:
    prompt = template.format(question=perm.question, **perm.options)
    pred, probs, attn_to_opts, probs_ly, _ = model.predict(prompt, letters)
    correct = 1 if pred == perm.correct_label else 0
    prob_correct = probs.get(perm.correct_label, 0.0)
    logprobs = {k: (math.log(v) if v and v > 0 else -100.0) for k, v in probs.items()}
    return {
        "item_id": perm.item_id,
        "correct_label": perm.correct_label,
        "correct_idx": perm.correct_idx,
        "pred": pred,
        "correct": correct,
        "prob_correct": prob_correct,
        "logprobs": logprobs,
        "attn_to_options": attn_to_opts if with_internals else None,
        "probs_per_layer": probs_ly if with_internals else None,
    }


def run_batch(
    perms: List[PermutedItem],
    model_name: str,
    model: OpenAIModel | HFModel,
    template: str,
    letters: List[str],
    is_hf: bool,
    with_internals: bool = True,
) -> List[Dict[str, Any]]:
    results = []
    iterator = tqdm(perms, desc=f"{model_name}", unit="item")
    for p in iterator:
        if is_hf:
            r = run_hf(model, p, template, letters, with_internals=with_internals)
        else:
            r = run_openai(model, p, template, letters)
        r["model"] = model_name
        results.append(r)
    return results
