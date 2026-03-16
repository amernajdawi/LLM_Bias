import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Set, Tuple

import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_dataset_by_name
from src.data.shuffle import PermutedItem, build_permutations
from src.db.client import count_results_for_model_dataset_sync, run_with_db
from src.io import load_json, save_json
from src.models.ollama_cloud import OllamaCloudModel
from src.run.experiment import run_batch, run_openai

CHUNK_SIZE = 50
# Number of concurrent API requests (higher = faster but more risk of 503 rate limits)
CONCURRENCY = int(os.environ.get("OLLAMA_CLOUD_CONCURRENCY", "5"))


def _done_key(r: dict) -> Tuple[str, str, int]:
    return (r["item_id"], r["correct_label"], r["correct_idx"])


def main(model_name: str) -> None:
    load_dotenv(ROOT / ".env", override=True)

    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    letters: List[str] = cfg.get("option_letters", ["A", "B", "C", "D"])
    template: str = cfg.get(
        "prompt_template",
        "Question: {question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:\n",
    )
    max_n = cfg.get("max_samples_per_dataset")
    datasets = cfg.get("datasets", [])
    configs = cfg.get("dataset_configs", {})
    splits = cfg.get("splits", {})

    out_dir = Path(cfg.get("results_dir", "outputs/results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = OllamaCloudModel(model_name)
    use_db = bool(os.environ.get("DATABASE_URL"))

    for ds_path in datasets:
        dcfg = configs.get(ds_path, "default")
        split = splits.get(ds_path, "test")
        items = load_dataset_by_name(ds_path, dcfg, split, letters, max_n)
        perms = build_permutations(items, letters)
        ds_name = ds_path.replace("/", "_")
        expected = len(perms)

        if use_db and count_results_for_model_dataset_sync(model_name, ds_name) >= expected:
            print(f"Skipping {model_name} on {ds_name} (already have {expected} results in DB).", flush=True)
            continue

        partial_path = out_dir / f"partial_{model_name}_{ds_name}.json"
        done_results: List[dict] = []
        if partial_path.exists():
            try:
                data = load_json(partial_path)
                done_results = data.get("results") or []
            except Exception:
                pass
        done_keys: Set[Tuple[str, str, int]] = {_done_key(r) for r in done_results}
        remaining_perms: List[PermutedItem] = [
            p for p in perms
            if (p.item_id, p.correct_label, p.correct_idx) not in done_keys
        ]

        if not remaining_perms:
            results = done_results
            for r in results:
                r["dataset"] = ds_name
            save_json(
                {"model": model_name, "dataset": ds_name, "results": results},
                out_dir / f"{model_name}_{ds_name}.json",
            )
            if use_db:
                run_with_db(model_name, ds_name, results)
            partial_path.unlink(missing_ok=True)
            print(f"Completed {model_name} on {ds_name} (saved and wrote to DB).", flush=True)
            continue

        print(
            f"Running {model_name} on {ds_name} ({len(remaining_perms)} remaining of {expected}, concurrency={CONCURRENCY})...",
            flush=True,
        )

        pbar = tqdm(total=len(remaining_perms), desc=model_name, unit="item") if CONCURRENCY > 1 else None

        def _run_one(p: PermutedItem):
            r = run_openai(model, p, template, letters)
            r["model"] = model_name
            r["dataset"] = ds_name
            if pbar is not None:
                pbar.update(1)
            return r

        for i in range(0, len(remaining_perms), CHUNK_SIZE):
            chunk = remaining_perms[i : i + CHUNK_SIZE]
            if CONCURRENCY <= 1:
                chunk_results = run_batch(
                    chunk,
                    model_name,
                    model,
                    template,
                    letters,
                    is_hf=False,
                    with_internals=False,
                )
                for r in chunk_results:
                    r["dataset"] = ds_name
            else:
                with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
                    chunk_results = list(ex.map(_run_one, chunk))
            done_results.extend(chunk_results)
            save_json(
                {"model": model_name, "dataset": ds_name, "results": done_results},
                partial_path,
            )
            if len(done_results) >= expected:
                break
        if pbar is not None:
            pbar.close()

        results = done_results
        if len(results) >= expected:
            save_json(
                {"model": model_name, "dataset": ds_name, "results": results},
                out_dir / f"{model_name}_{ds_name}.json",
            )
            if use_db:
                run_with_db(model_name, ds_name, results)
            partial_path.unlink(missing_ok=True)
            print(f"Completed {model_name} on {ds_name}.", flush=True)
        else:
            print(
                f"Stopped with {len(results)}/{expected} for {model_name} on {ds_name}. "
                "Re-run the same command to resume.",
                flush=True,
            )

    print("Done.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m scripts.run_ollama_cloud <model_name>")
        raise SystemExit(1)
    main(sys.argv[1])

