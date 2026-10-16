"""
Corrected position-bias experiment on Ollama Cloud (v3).

Fixes the measurement problems found in the earlier runs:
  * explicit instruction to reply with only the label (the old config.yaml prompt ended in "Answer:")
  * strict answer extraction; unreadable replies are INVALID, never silently "A"
  * one retry for invalid replies with Ollama structured output (JSON schema enum), recorded as such
  * every raw reply (and thinking text) is saved
  * full cyclic rotation of ALL options (gold-independent), so label-free aggregation
    (majority vote, PAV) is evaluated fairly; the correct answer still visits every slot once
  * letters (A-D) and numbers (1-4) use the same prompt, rotations and extraction rules

    .venv/bin/python3 -u experiments/v3_corrected/run_cloud.py --models gemma4:31b-cloud ... [--items 200] [--formats letter number]

Providers (--provider):
    cloud   Ollama Cloud, key OLLAMA_API_KEY in .env (default)
    local   your own Ollama server, e.g. --host http://localhost:11434 (no key)
    openai  OpenAI API, key OPENAI_API_KEY in .env (paid)
--check sends one tiny request per model and reports whether it can be used, then stops.

Resume-safe: one JSONL per model/dataset/format under experiments/v3_corrected/raw/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.data.loaders import load_dataset_by_name

RAW = Path(__file__).resolve().parent / "raw"
K = 4
DATASETS = {
    "allenai_ai2_arc": ("allenai/ai2_arc", "ARC-Challenge"),
    "allenai_openbookqa": ("allenai/openbookqa", "main"),
}
LABELS = {"letter": ["A", "B", "C", "D"], "number": ["1", "2", "3", "4"]}
PROMPT = {
    "letter": "Question: {question}\n{block}\nReply with only the letter of the correct option (A, B, C, or D).",
    "number": "Question: {question}\n{block}\nReply with only the number of the correct option (1, 2, 3, or 4).",
}
THINK = {"gpt-oss:120b": "low", "gpt-oss:20b": "low"}   # gpt-oss cannot switch thinking off; everything else: False


def block(fmt, shown):
    if fmt == "letter":
        return "\n".join(f"{lab}) {txt}" for lab, txt in zip(LABELS[fmt], shown))
    return "\n".join(f"({lab}) {txt}" for lab, txt in zip(LABELS[fmt], shown))


def strict_pick(fmt, text, shown):
    """Return (slot, how) or (None, 'invalid'). Never guesses a default."""
    t = (text or "").strip()
    lab = "[A-D]" if fmt == "letter" else "[1-4]"
    nxt = "[A-Za-z]" if fmt == "letter" else "[0-9]"
    pats = [
        rf"^\W*\(?({lab})\)?(?=$|\s*$|[\)\]\.\:,;\n\*]|\s+-)",
        rf"(?i:\banswer\b)(?i:\s+is)?\s*:?\s*\**\(?({lab})\)?(?!{nxt})",
        rf"(?i:\boption\b)\s*\**\(?({lab})\)?(?!{nxt})",
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            return LABELS[fmt].index(m.group(1)), "strict"
    low = t.lower()
    hits = [s for s, txt in enumerate(shown) if txt and txt.lower().strip(" .") in low]
    if len(hits) == 1:
        return hits[0], "text"
    return None, "invalid"


WAITS = [15, 30, 60, 120, 240]          # back-off for rate limits and server errors


class Cloud:
    """Ollama chat API: Ollama Cloud (default) or a local Ollama server (host given, no key)."""

    def __init__(self, host=None):
        if host:
            self.c = httpx.Client(base_url=host, timeout=httpx.Timeout(900.0, connect=30.0))
            return
        key = os.environ.get("OLLAMA_API_KEY")
        if not key:
            raise SystemExit("OLLAMA_API_KEY is not set in .env")
        self.c = httpx.Client(base_url="https://ollama.com", headers={"Authorization": f"Bearer {key}"},
                              timeout=httpx.Timeout(300.0, connect=30.0))

    def chat(self, model, prompt, fmt_schema=None):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                   "think": THINK.get(model, False), "options": {"temperature": 0}}
        if fmt_schema:
            payload["format"] = fmt_schema
        for attempt in range(len(WAITS) + 1):
            try:
                r = self.c.post("/api/chat", json=payload)
                if r.status_code == 400 and "think" in r.text.lower() and "think" in payload:
                    payload.pop("think")                      # model does not accept the think flag
                    continue
                if r.status_code in (429, 500, 502, 503, 504) and attempt < len(WAITS):
                    time.sleep(WAITS[attempt]); continue
                r.raise_for_status()
                msg = r.json().get("message") or {}
                return (msg.get("content") or "").strip(), (msg.get("thinking") or "").strip(), None
            except httpx.TransportError as e:
                if attempt < len(WAITS):
                    time.sleep(WAITS[attempt]); continue
                return "", "", f"transport: {e}"
            except httpx.HTTPStatusError as e:
                return "", "", f"http {e.response.status_code}: {e.response.text[:200]}"
        return "", "", "retries exhausted"


class OpenAI:
    """OpenAI chat completions with the same interface as Cloud.chat (no thinking text is returned)."""

    def __init__(self, transport=None):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set in .env")
        self.c = httpx.Client(base_url="https://api.openai.com/v1", headers={"Authorization": f"Bearer {key}"},
                              timeout=httpx.Timeout(300.0, connect=30.0), transport=transport)

    def chat(self, model, prompt, fmt_schema=None):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                   "max_completion_tokens": 256}
        if fmt_schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "answer", "strict": True, "schema": {**fmt_schema, "additionalProperties": False}}}
        for attempt in range(len(WAITS) + 1):
            try:
                r = self.c.post("/chat/completions", json=payload)
                if r.status_code == 400 and "temperature" in r.text and "temperature" in payload:
                    payload.pop("temperature")                # reasoning models accept only the default
                    continue
                if r.status_code in (429, 500, 502, 503, 504) and attempt < len(WAITS):
                    time.sleep(WAITS[attempt]); continue
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                return (msg.get("content") or "").strip(), "", None
            except httpx.TransportError as e:
                if attempt < len(WAITS):
                    time.sleep(WAITS[attempt]); continue
                return "", "", f"transport: {e}"
            except httpx.HTTPStatusError as e:
                return "", "", f"http {e.response.status_code}: {e.response.text[:200]}"
        return "", "", "retries exhausted"


def check(client, models):
    """One tiny request per model: can it be used with this key and provider?"""
    for m in models:
        content, _, err = client.chat(m, "Reply with only the letter A.")
        print(f"  {m:<26} " + (f"OK (replied {content[:20]!r})" if err is None else f"NOT USABLE: {err[:110]}"), flush=True)


def tasks_for(items, fmt):
    for it in items:
        for k in range(K):
            order = [(k + s) % K for s in range(K)]   # slot s shows original option order[s]
            yield it, k, order


def run_one(cloud, model, ds, fmt, it, k, order):
    shown = [it.choices[c] for c in order]
    prompt = PROMPT[fmt].format(question=it.question, block=block(fmt, shown))
    t0 = time.time()
    content, thinking, err = cloud.chat(model, prompt)
    if err is None and content and set(content) <= {"?"}:
        err = "broken reply (only '?'): server fault, asked again on the next run"   # seen with a local Ollama GPU fault
    slot, how = strict_pick(fmt, content, shown) if err is None else (None, "invalid")
    retry_raw = None
    if slot is None and err is None:
        schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": LABELS[fmt]}},
                  "required": ["answer"]}
        retry_raw, _, err2 = cloud.chat(model, prompt, fmt_schema=schema)
        try:
            ans = json.loads(retry_raw).get("answer")
            if ans in LABELS[fmt]:
                slot, how = LABELS[fmt].index(ans), "format_retry"
        except Exception:
            pass
        err = err or err2
    gold_slot = order.index(it.correct_idx)
    return {
        "model": model, "provider": getattr(cloud, "provider", "cloud"), "dataset": ds, "format": fmt,
        "item_id": it.id, "rotation": k,
        "order": order, "gold_idx": it.correct_idx, "gold_slot": gold_slot,
        "pick_slot": slot, "pick_idx": order[slot] if slot is not None else None,
        "correct": int(slot == gold_slot), "parse": how,
        "raw": content, "retry_raw": retry_raw, "thinking": thinking[:2000], "error": err,
        "seconds": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--formats", nargs="+", default=["letter", "number"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("OLLAMA_CLOUD_CONCURRENCY", "5")))
    ap.add_argument("--provider", choices=["cloud", "local", "openai"], default="cloud")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama server for --provider local")
    ap.add_argument("--check", action="store_true", help="only test whether each model can be used")
    args = ap.parse_args()

    cloud = OpenAI() if args.provider == "openai" else Cloud(args.host if args.provider == "local" else None)
    cloud.provider = args.provider
    if args.check:
        check(cloud, args.models)
        return
    RAW.mkdir(parents=True, exist_ok=True)
    data = {ds: load_dataset_by_name(DATASETS[ds][0], DATASETS[ds][1], "test", LABELS["letter"], args.items)
            for ds in args.datasets}
    for fmt in args.formats:                       # all letters first: they carry the main results
        for model in args.models:
            for ds in args.datasets:
                path = RAW / f"{model.replace(':', '_')}__{ds}__{fmt}.jsonl"
                done = set()
                if path.exists():
                    for line in path.read_text().splitlines():
                        r = json.loads(line)
                        if r.get("error") is None:
                            done.add((r["item_id"], r["rotation"]))
                todo = [t for t in tasks_for(data[ds], fmt) if (t[0].id, t[1]) not in done]
                print(f"[{fmt}] {model} {ds}: {len(todo)} to run ({len(done)} done)", flush=True)
                lock = threading.Lock()
                stats = {"n": 0, "ok": 0, "invalid": 0, "err": 0}
                t0 = time.time()
                with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as ex:
                    for rec in ex.map(lambda t: run_one(cloud, model, ds, fmt, *t), todo):
                        with lock:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                            stats["n"] += 1; stats["ok"] += rec["correct"]
                            stats["invalid"] += rec["pick_slot"] is None; stats["err"] += rec["error"] is not None
                            if stats["n"] % 100 == 0 or stats["n"] == len(todo):
                                print(f"  {model} {ds} {fmt}: {stats['n']}/{len(todo)} acc={stats['ok']/stats['n']:.1%} "
                                      f"invalid={stats['invalid']} errors={stats['err']} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
