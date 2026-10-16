"""
Local Ollama models with the corrected protocol, answers read from token probabilities.

Same questions, prompt and rotations as run_cloud.py. Ollama (local server, version 0.12+) returns the
log-probabilities of the most likely first tokens; the four label probabilities are taken from them and
saved in the raw_hf/ format, so analyze_hf.py tests every method (prior correction, PriDe, averaging, ...).
Ollama Cloud does not return log-probabilities, so this works only with a local server.

    .venv/bin/python3 -u experiments/v3_corrected/run_ollama_probs.py --models llama3:8b qwen2.5:7b-instruct
    options: --host http://localhost:11434  --items 200  --formats letter number

Resume-safe: one JSONL per model/dataset/format under experiments/v3_corrected/raw_hf/ (prefix "ollama-").
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.data.loaders import load_dataset_by_name
from run_cloud import DATASETS, LABELS, PROMPT, WAITS, block, tasks_for

OUT = HERE / "raw_hf"


def label_probs(client, model, prompt, labels):
    """Probabilities of the four labels as the first token (labels outside the top 20 get 0)."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False,
               "logprobs": True, "top_logprobs": 20, "options": {"temperature": 0, "num_predict": 1}}
    for attempt in range(len(WAITS) + 1):
        try:
            r = client.post("/api/chat", json=payload)
            if r.status_code == 400 and "think" in r.text.lower() and "think" in payload:
                payload.pop("think"); continue
            r.raise_for_status()
            lp = r.json().get("logprobs") or []
            if not lp:
                return None, "no logprobs returned"
            p = [0.0] * len(labels)
            for t in lp[0].get("top_logprobs", []):
                tok = t["token"].strip()
                if tok in labels:
                    p[labels.index(tok)] += math.exp(t["logprob"])
            return p, None
        except httpx.TransportError as e:
            if attempt < len(WAITS):
                time.sleep(WAITS[attempt]); continue
            return None, f"transport: {e}"
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504) and attempt < len(WAITS):
                time.sleep(WAITS[attempt]); continue
            return None, f"http {e.response.status_code}: {e.response.text[:200]}"
    return None, "retries exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--formats", nargs="+", default=["letter", "number"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.host, timeout=httpx.Timeout(900.0, connect=30.0))
    data = {ds: load_dataset_by_name(DATASETS[ds][0], DATASETS[ds][1], "test", LABELS["letter"], args.items)
            for ds in args.datasets}

    for model in args.models:
        for fmt in args.formats:
            for ds in args.datasets:
                path = out / f"ollama-{model.replace(':', '_')}__{ds}__{fmt}.jsonl"
                done = set()
                if path.exists():
                    for line in path.read_text().splitlines():
                        try:
                            r = json.loads(line); done.add((r["item_id"], r["rotation"]))
                        except json.JSONDecodeError:
                            pass
                t0 = time.time(); n = skipped = 0
                with path.open("a", encoding="utf-8") as f:
                    for it, k, order in tasks_for(data[ds], fmt):
                        if (it.id, k) in done:
                            continue
                        shown = [it.choices[c] for c in order]
                        prompt = PROMPT[fmt].format(question=it.question, block=block(fmt, shown))
                        p, err = label_probs(client, model, prompt, LABELS[fmt])
                        if err:                                   # not saved: asked again on the next run
                            skipped += 1
                            print(f"  {model} {ds} {fmt}: {err}", flush=True)
                            if skipped >= 20 and n == 0:
                                raise SystemExit(f"{model}: no usable answers ({err})")
                            continue
                        mass = sum(p)
                        if mass <= 0:
                            p, mass = [0.25] * 4, 0.0                # no label in the top 20: no information
                        else:
                            p = [x / mass for x in p]
                        pick = max(range(4), key=lambda s: p[s])
                        gold_slot = order.index(it.correct_idx)
                        f.write(json.dumps({"model": f"{model} (Ollama)", "hf_id": f"ollama:{model}", "dataset": ds,
                                            "format": fmt, "item_id": it.id, "rotation": k, "order": order,
                                            "gold_idx": it.correct_idx, "gold_slot": gold_slot, "probs": p,
                                            "label_mass": mass, "pick_slot": pick, "pick_idx": order[pick],
                                            "correct": int(pick == gold_slot), "parse": "probability", "error": None}) + "\n")
                        f.flush()
                        n += 1
                        if n % 200 == 0:
                            print(f"{model} {fmt} {ds}: {n} prompts ({time.time() - t0:.0f}s)", flush=True)
                print(f"{model} {fmt} {ds}: done ({time.time() - t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
