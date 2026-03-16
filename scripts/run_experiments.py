import gc
import os
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.data.loaders import load_dataset_by_name
from src.data.shuffle import build_permutations
from src.db.client import count_results_for_model_dataset_sync, run_with_db
from src.io import load_json, save_json
from src.models.hf_model import HFModel
from src.models.openai_model import OpenAIModel
from src.run.experiment import run_batch


def main():
    print("Starting run_experiments.py ...", flush=True)
    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    letters = cfg.get("option_letters", ["A", "B", "C", "D"])
    template = cfg.get("prompt_template", "Question: {question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:\n")
    max_n = cfg.get("max_samples_per_dataset")
    datasets = cfg.get("datasets", [])
    configs = cfg.get("dataset_configs", {})
    splits = cfg.get("splits", {})
    out_dir = Path(cfg.get("results_dir", "outputs/results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = cfg.get("models", {})
    openai_id = models_cfg.get("openai", {}).get("id")
    llama_id = models_cfg.get("llama", {}).get("id")
    qwen_id = models_cfg.get("qwen", {}).get("id")
    mistral_id = models_cfg.get("mistral", {}).get("id")

    # On Mac Docker / low RAM, 3B OOMs. Use 0.5B when LOW_MEMORY=1 so runs complete.
    if os.environ.get("LOW_MEMORY", "").strip().lower() in ("1", "true", "yes") and qwen_id and "0.5B" not in qwen_id:
        qwen_id = "Qwen/Qwen2.5-0.5B-Instruct"
        print("LOW_MEMORY=1: using Qwen2.5-0.5B-Instruct so run fits in RAM.", flush=True)

    all_data = []
    for ds_path in datasets:
        print(f"Loading dataset: {ds_path} ...", flush=True)
        dcfg = configs.get(ds_path, "default")
        sp = splits.get(ds_path, "test")
        items = load_dataset_by_name(ds_path, dcfg, sp, letters, max_n)
        perms = build_permutations(items, letters)
        name = ds_path.replace("/", "_")
        all_data.append((name, perms))
        print(f"  {name}: {len(perms)} items", flush=True)

    use_db = bool(os.environ.get("DATABASE_URL"))
    print(f"DB enabled: {use_db}", flush=True)
    print("Checking which runs to skip / run ...", flush=True)

    def already_done(model_key: str, ds_name: str, expected: int) -> bool:
        json_path = out_dir / f"{model_key}_{ds_name}.json"
        if json_path.exists():
            try:
                data = load_json(json_path)
                n = len(data.get("results") or [])
                if n == expected:
                    return True
            except Exception:
                pass
        if use_db:
            n = count_results_for_model_dataset_sync(model_key, ds_name)
            if n == expected:
                return True
        return False

    def run_model(model_key: str, model, is_hf: bool):
        for ds_name, perms in all_data:
            expected = len(perms)
            if already_done(model_key, ds_name, expected):
                print(f"Skipping {model_key} + {ds_name} (already have {expected} results)", flush=True)
                continue
            print(f"Running {model_key} + {ds_name} ({expected} items) ...", flush=True)
            results = run_batch(perms, model_key, model, template, letters, is_hf=is_hf, with_internals=is_hf)
            for r in results:
                r["dataset"] = ds_name
            save_json({"model": model_key, "dataset": ds_name, "results": results}, out_dir / f"{model_key}_{ds_name}.json")
            if use_db:
                run_with_db(model_key, ds_name, results)

    if openai_id:
        run_model("openai", OpenAIModel(openai_id), is_hf=False)

    def _cleanup_after_hf():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _any_run_needed(model_key: str) -> bool:
        """True if at least one (model_key, dataset) run is not already done."""
        for ds_name, perms in all_data:
            if not already_done(model_key, ds_name, len(perms)):
                return True
        return False

    # Run Qwen before LLaMA so we never hold both in memory when only Qwen has work
    if qwen_id:
        if not _any_run_needed("qwen"):
            print("Skipping Qwen (all runs already done, not loading model)", flush=True)
        else:
            print("Loading Qwen model (download + load can take 5–30 min, may show no output) ...", flush=True)
            model = HFModel(qwen_id)
            run_model("qwen", model, is_hf=True)
            del model
            _cleanup_after_hf()

    if llama_id:
        if not _any_run_needed("llama"):
            print("Skipping LLaMA (all runs already done, not loading model)", flush=True)
        else:
            print("Loading LLaMA model (download + load can take 5–30 min, may show no output) ...", flush=True)
            model = HFModel(llama_id)
            run_model("llama", model, is_hf=True)
            del model
            _cleanup_after_hf()

    if mistral_id:
        if not _any_run_needed("mistral"):
            print("Skipping Mistral (all runs already done, not loading model)", flush=True)
        else:
            print("Loading Mistral model (download + load can take 5–30 min, may show no output) ...", flush=True)
            model = HFModel(mistral_id)
            run_model("mistral", model, is_hf=True)
            del model
            _cleanup_after_hf()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
