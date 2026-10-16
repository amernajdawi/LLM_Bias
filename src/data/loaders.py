from dataclasses import dataclass
from glob import glob
import json
from pathlib import Path
from typing import List, Optional


@dataclass
class MCQItem:
    id: str
    question: str
    choices: List[str]
    correct_idx: int


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

_JSON_CACHE = {
    "allenai/ai2_arc": _PROJECT_ROOT / "data_arc.json",
    "allenai/openbookqa": _PROJECT_ROOT / "data_obqa.json",
}


def _load_rows(dataset_repo: str, config: str, split: str) -> list:
    """Load dataset rows from pre-extracted JSON, parquet cache, or HF."""
    json_path = _JSON_CACHE.get(dataset_repo)
    if json_path and json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    pq_path = _find_parquet(dataset_repo, config, split)
    if pq_path:
        import pyarrow.parquet as pq
        return pq.read_table(pq_path).to_pylist()

    import sys
    _block_torch = "torch" not in sys.modules
    if _block_torch:
        sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        from datasets import load_dataset
        kw = {"path": dataset_repo, "split": split}
        if config:
            kw["name"] = config
        return list(load_dataset(**kw))
    finally:
        if _block_torch and sys.modules.get("torch") is None:
            del sys.modules["torch"]


def _find_parquet(dataset_repo: str, config: str, split: str) -> Optional[str]:
    repo_dir = _HF_CACHE / f"datasets--{dataset_repo.replace('/', '--')}"
    if not repo_dir.exists():
        return None
    pattern = f"snapshots/*/{config or ''}*/{split}-*.parquet" if config else f"snapshots/*/**/{split}-*.parquet"
    matches = glob(str(repo_dir / pattern), recursive=True)
    if not matches:
        matches = glob(str(repo_dir / f"snapshots/**/{split}-*.parquet"), recursive=True)
    return matches[0] if matches else None


def _norm_choices(raw: dict, letters: List[str]) -> List[str]:
    out = []
    choices = raw.get("choices", {})
    if isinstance(choices, dict):
        if "label" in choices and "text" in choices:
            order = [l for l in choices["label"] if l in letters]
            idx = {l: choices["label"].index(l) for l in order}
            for L in letters:
                if L in idx:
                    out.append(choices["text"][idx[L]].strip())
        elif all(k in choices for k in letters):
            out = [str(choices[k]).strip() for k in letters]
    return out


def _correct_idx(raw: dict, letters: List[str]) -> int:
    key = (raw.get("answerKey") or raw.get("answer_key") or "").strip().upper()
    if key and key[0] in letters:
        return letters.index(key[0])
    return 0


def _rows_to_items(rows: list, letters: List[str], max_n: Optional[int], question_key: str = "question") -> List[MCQItem]:
    items = []
    for row in rows:
        q = row.get(question_key, row.get("question_stem", row.get("question", "")))
        ch = _norm_choices(row, letters)
        if len(ch) < 4:
            continue
        ch = ch[:4]
        ci = _correct_idx(row, letters)
        if ci >= len(ch):
            ci = 0
        items.append(MCQItem(id=row.get("id", str(len(items))), question=q, choices=ch, correct_idx=ci))
        if max_n and len(items) >= max_n:
            break
    return items


def load_arc(path: str, config: str, split: str, letters: List[str], max_n: Optional[int]) -> List[MCQItem]:
    rows = _load_rows(path, config, split)
    return _rows_to_items(rows, letters, max_n, question_key="question")


def load_openbookqa(path: str, config: str, split: str, letters: List[str], max_n: Optional[int]) -> List[MCQItem]:
    rows = _load_rows(path, config, split)
    return _rows_to_items(rows, letters, max_n, question_key="question_stem")


def load_dataset_by_name(path: str, config: str, split: str, letters: List[str], max_n: Optional[int]) -> List[MCQItem]:
    if "arc" in path.lower() or "ai2_arc" in path:
        return load_arc(path, config, split, letters, max_n)
    if "openbookqa" in path.lower():
        return load_openbookqa(path, config, split, letters, max_n)
    return load_arc(path, config, split, letters, max_n)
