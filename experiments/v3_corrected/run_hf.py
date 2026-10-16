"""
Open-weight models (Hugging Face) with the corrected protocol, answers read from probabilities.

Same questions, prompt text and full cyclic rotation as run_cloud.py. The answer is the label with
the highest next-token probability (no text parsing). The four label probabilities are saved so
every method (prior correction, PriDe, voting, PAV, averaging) can be tested by analyze_hf.py.

    nice -n 10 .venv/bin/python3 -u experiments/v3_corrected/run_hf.py --model meta-llama/Llama-3.2-1B-Instruct
    options: --items 200  --threads 4  --device cpu|cuda  --formats letter number

Gated models (Llama) need HF_TOKEN in .env with access granted on huggingface.co.
Resume-safe: one JSONL per model/dataset/format under experiments/v3_corrected/raw_hf/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
load_dotenv(ROOT / ".env")
from src.data.loaders import load_dataset_by_name
from run_cloud import DATASETS, LABELS, PROMPT, block, tasks_for

OUT = HERE / "raw_hf"


def short_name(model_id):
    """'meta-llama/Llama-3.2-1B-Instruct' -> 'llama-3.2-1b' (file prefix)."""
    return model_id.split("/")[-1].lower().replace("-instruct", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--formats", nargs="+", default=["letter", "number"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    label_ids = {}
    for fmt, labs in LABELS.items():
        ids = [tok.encode(x, add_special_tokens=False) for x in labs]
        assert all(len(i) == 1 for i in ids), f"labels must be single tokens: {ids}"
        label_ids[fmt] = [i[0] for i in ids]
    name = args.model.split("/")[-1]

    for fmt in args.formats:
        for ds in args.datasets:
            items = load_dataset_by_name(DATASETS[ds][0], DATASETS[ds][1], "test", LABELS["letter"], args.items)
            path = out / f"{short_name(args.model)}__{ds}__{fmt}.jsonl"
            done = set()
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        r = json.loads(line); done.add((r["item_id"], r["rotation"]))
                    except json.JSONDecodeError:
                        pass                                   # a line cut by an interrupted run is redone
            t0 = time.time(); n = 0
            with path.open("a", encoding="utf-8") as f:
                for it, k, order in tasks_for(items, fmt):
                    if (it.id, k) in done:
                        continue
                    shown = [it.choices[c] for c in order]
                    prompt = PROMPT[fmt].format(question=it.question, block=block(fmt, shown))
                    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(args.device)
                    with torch.no_grad():
                        logits = model(**enc).logits[0, -1].float()
                    p = torch.softmax(logits, -1)[label_ids[fmt]]
                    mass = float(p.sum())
                    p = (p / p.sum()).tolist()
                    pick = max(range(4), key=lambda s: p[s])
                    gold_slot = order.index(it.correct_idx)
                    f.write(json.dumps({"model": name, "hf_id": args.model, "dataset": ds, "format": fmt, "item_id": it.id,
                                        "rotation": k, "order": order, "gold_idx": it.correct_idx, "gold_slot": gold_slot,
                                        "probs": p, "label_mass": mass, "pick_slot": pick, "pick_idx": order[pick],
                                        "correct": int(pick == gold_slot), "parse": "probability", "error": None}) + "\n")
                    f.flush()
                    n += 1
                    if n % 200 == 0:
                        print(f"{name} {fmt} {ds}: {n} prompts ({time.time() - t0:.0f}s)", flush=True)
            print(f"{name} {fmt} {ds}: done ({time.time() - t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
