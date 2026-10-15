"""Experimentos com diferentes parametros (treino e inferencia).

Uso:
    python -m src.experiments --mode train --quick           # teste rapido (~10 min)
    python -m src.experiments --mode train                   # varredura completa
    python -m src.experiments --mode inference --adapter outputs/baseline/adapter
    python -m src.experiments --mode all
Os resultados ficam em results/experiments.csv e results/inference_sweep.csv.
"""
import argparse
import json
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd
import torch

from .config import DEFAULT_MODEL, OUTPUTS_DIR, RESULTS_DIR, GenConfig, TrainConfig
from .data import load_eval_examples, load_pt_examples, load_sql_dataset
from .evaluate import evaluate_generator, save_eval
from .inference import SQLGenerator
from .train import run_training

RUNS_DIR = RESULTS_DIR / "runs"

# Variacoes em relacao ao baseline (r=16, lr=2e-4, 1 epoca): um parametro por vez.
SWEEPS: Dict[str, dict] = {
    "baseline": {},
    "r4": dict(lora_r=4, lora_alpha=8),
    "r64": dict(lora_r=64, lora_alpha=128),
    "lr5e-5": dict(learning_rate=5e-5),
    "lr5e-4": dict(learning_rate=5e-4),
    "ep2": dict(epochs=2.0),
    "ep3": dict(epochs=3.0),
}
ORDER = ["base_model"] + list(SWEEPS)


def _tag(name: str, quick: bool) -> str:
    return f"quick_{name}" if quick else name


def run_base_eval(model_name: str, eval_samples: int, use_4bit: bool, quick: bool) -> dict:
    """Avalia o modelo base (zero-shot), ponto de comparacao para o fine-tuning."""
    name = _tag("base_model", quick)
    gen = SQLGenerator.load(model_name, None, use_4bit=use_4bit)
    m_en, rows = evaluate_generator(gen, load_eval_examples(eval_samples), GenConfig(), desc=f"{name} EN")
    save_eval(name, m_en, rows)
    pt = load_pt_examples("eval")
    m_pt = {}
    if pt:
        m_pt, rows_pt = evaluate_generator(gen, pt, GenConfig(), desc=f"{name} PT")
        save_eval(name, m_pt, rows_pt, suffix="_pt")
    gen.free()
    summary = {"run_name": name, "config": {"model_name": model_name}, "train": {},
               "eval_en": m_en, "eval_pt": m_pt}
    _save_run(summary)
    return summary


def run_one(name: str, overrides: dict, model_name: str, train_samples: int, eval_samples: int,
            use_4bit: bool, skip_existing: bool, quick: bool) -> dict:
    tag = _tag(name, quick)
    path = RUNS_DIR / f"{tag}.json"
    if skip_existing and path.exists():
        print(f"[{tag}] ja existe, pulando")
        return json.loads(path.read_text(encoding="utf-8"))

    cfg = TrainConfig(run_name=tag, model_name=model_name, train_samples=train_samples,
                      use_4bit=use_4bit, **overrides)
    data = load_sql_dataset(cfg.train_samples, eval_samples, cfg.seed, cfg.pt_repeat)
    info = run_training(cfg, data=data)

    gen = SQLGenerator.load(cfg.model_name, str(cfg.adapter_dir), use_4bit=cfg.use_4bit)
    m_en, rows = evaluate_generator(gen, data["eval"], GenConfig(), desc=f"{tag} EN")
    save_eval(tag, m_en, rows)
    pt = load_pt_examples("eval")
    m_pt = {}
    if pt:
        m_pt, rows_pt = evaluate_generator(gen, pt, GenConfig(), desc=f"{tag} PT")
        save_eval(tag, m_pt, rows_pt, suffix="_pt")
    gen.free()

    train_info = {k: v for k, v in info.items() if k != "log_history"}
    summary = {"run_name": tag, "config": asdict(cfg), "train": train_info,
               "eval_en": m_en, "eval_pt": m_pt}
    _save_run(summary)
    return summary


def _save_run(summary: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{summary['run_name']}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    collect_results()


def collect_results() -> pd.DataFrame:
    """Junta results/runs/*.json em results/experiments.csv."""
    rows = []
    for p in RUNS_DIR.glob("*.json"):
        s = json.loads(p.read_text(encoding="utf-8"))
        c, t = s.get("config") or {}, s.get("train") or {}
        en, pt = s.get("eval_en") or {}, s.get("eval_pt") or {}
        rows.append({
            "run": s["run_name"],
            "lora_r": c.get("lora_r"), "lora_alpha": c.get("lora_alpha"),
            "learning_rate": c.get("learning_rate"), "epochs": c.get("epochs"),
            "train_samples": c.get("train_samples"),
            "train_time_s": t.get("train_runtime_s"), "final_loss": t.get("final_loss"),
            "peak_gpu_gb": t.get("peak_gpu_gb"),
            "exact_match": en.get("exact_match"), "valid_sql": en.get("valid_sql"),
            "exec_acc": en.get("exec_accuracy"), "pt_exec_acc": pt.get("exec_accuracy"),
            "sec_per_example": en.get("sec_per_example"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ("lora_r", "lora_alpha", "train_samples"):
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in ("exact_match", "valid_sql", "exec_acc", "pt_exec_acc", "final_loss",
                "sec_per_example", "peak_gpu_gb", "train_time_s"):
        df[col] = pd.to_numeric(df[col]).round(4)

    def key(run: str):
        base = run.removeprefix("quick_")
        return (ORDER.index(base) if base in ORDER else 99, run)

    df = df.iloc[sorted(range(len(df)), key=lambda i: key(df.iloc[i]["run"]))]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_quick = df["run"].str.startswith("quick_")
    # Testes rapidos (--quick) ficam em arquivo separado, fora do relatorio
    if (~is_quick).any():
        df[~is_quick].to_csv(RESULTS_DIR / "experiments.csv", index=False, encoding="utf-8-sig")
    if is_quick.any():
        df[is_quick].to_csv(RESULTS_DIR / "experiments_quick.csv", index=False, encoding="utf-8-sig")
    return df


def run_train_sweep(model_name: str = DEFAULT_MODEL, train_samples: int = 2000, eval_samples: int = 200,
                    use_4bit: bool = True, only: Optional[List[str]] = None,
                    skip_existing: bool = True, quick: bool = False) -> pd.DataFrame:
    if quick:
        train_samples, eval_samples = 300, 60
    run_base_eval_needed = not (skip_existing and (RUNS_DIR / f"{_tag('base_model', quick)}.json").exists())
    if run_base_eval_needed and (only is None or "base_model" in only):
        run_base_eval(model_name, eval_samples, use_4bit, quick)
    for name, overrides in SWEEPS.items():
        if only is not None and name not in only:
            continue
        run_one(name, overrides, model_name, train_samples, eval_samples, use_4bit, skip_existing, quick)
    df = collect_results()
    print(df.to_string(index=False))
    return df


def run_inference_sweep(adapter_path: str, model_name: str = DEFAULT_MODEL, eval_samples: int = 200,
                        use_4bit: bool = True) -> pd.DataFrame:
    """Varia temperature/top_p na geracao, mantendo o modelo treinado fixo."""
    gen = SQLGenerator.load(model_name, adapter_path, use_4bit=use_4bit)
    examples = load_eval_examples(eval_samples)
    grid = [(0.0, 1.0)] + [(t, p) for t in (0.3, 0.7, 1.0) for p in (0.9, 1.0)]
    out = []
    for temperature, top_p in grid:
        torch.manual_seed(0)
        label = "greedy" if temperature == 0 else f"T={temperature}, top_p={top_p}"
        metrics, _ = evaluate_generator(gen, examples, GenConfig(128, temperature, top_p), desc=label)
        out.append({"config": label, "temperature": temperature, "top_p": top_p, **metrics})
    gen.free()
    df = pd.DataFrame(out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "inference_sweep.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Experimentos de parametros (treino e inferencia)")
    p.add_argument("--mode", choices=["train", "inference", "all"], default="all")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--quick", action="store_true", help="versao reduzida para testar o pipeline")
    p.add_argument("--only", default=None, help="lista separada por virgula, ex.: r4,lr5e-4")
    p.add_argument("--no-skip-existing", action="store_true", help="refaz runs ja concluidos")
    p.add_argument("--adapter", default=str(OUTPUTS_DIR / "baseline" / "adapter"),
                   help="adaptador usado no sweep de inferencia")
    p.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    only = a.only.split(",") if a.only else None

    if a.mode in ("train", "all"):
        run_train_sweep(a.model, a.train_samples, a.eval_samples, a.use_4bit, only,
                        not a.no_skip_existing, a.quick)
    if a.mode in ("inference", "all"):
        adapter = a.adapter
        if a.quick and a.adapter == str(OUTPUTS_DIR / "baseline" / "adapter"):
            adapter = str(OUTPUTS_DIR / "quick_baseline" / "adapter")
        run_inference_sweep(adapter, a.model, 60 if a.quick else a.eval_samples, a.use_4bit)


if __name__ == "__main__":
    main()
