"""Avaliacao do modelo: exact match, SQL valido e acuracia de execucao.

Uso:
    python -m src.evaluate --name baseline --adapter outputs/baseline/adapter --n 200 --pt
    python -m src.evaluate --name base_model --n 200 --pt          # modelo base, sem fine-tuning
"""
import argparse
import json
import time
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from .config import DEFAULT_MODEL, RESULTS_DIR, GenConfig
from .data import load_eval_examples, load_pt_examples
from .inference import SQLGenerator
from .sqlutils import execution_match, normalize_sql


def evaluate_generator(gen: SQLGenerator, examples: List[Dict[str, str]], gen_cfg: GenConfig = None,
                       batch_size: int = 8, use_adapter: bool = True,
                       desc: str = "avaliando") -> Tuple[Dict, List[Dict]]:
    gen_cfg = gen_cfg or GenConfig()
    rows: List[Dict] = []
    t0 = time.time()
    for i in tqdm(range(0, len(examples), batch_size), desc=desc):
        batch = examples[i:i + batch_size]
        preds = gen.generate_batch([e["context"] for e in batch], [e["question"] for e in batch],
                                   gen_cfg, use_adapter=use_adapter)
        for ex, pred in zip(batch, preds):
            valid, match = execution_match(ex["context"], ex["answer"], pred)
            rows.append({
                "question": ex["question"],
                "gold": ex["answer"],
                "pred": pred,
                "exact_match": normalize_sql(pred) == normalize_sql(ex["answer"]),
                "valid_sql": valid,
                "exec_match": match,
            })
    elapsed = time.time() - t0
    evaluable = [r for r in rows if r["exec_match"] is not None]
    n_ev = max(1, len(evaluable))
    metrics = {
        "n": len(rows),
        "n_evaluable": len(evaluable),
        "exact_match": round(sum(r["exact_match"] for r in rows) / max(1, len(rows)), 4),
        "valid_sql": round(sum(bool(r["valid_sql"]) for r in evaluable) / n_ev, 4),
        "exec_accuracy": round(sum(bool(r["exec_match"]) for r in evaluable) / n_ev, 4),
        "sec_per_example": round(elapsed / max(1, len(rows)), 3),
    }
    return metrics, rows


def save_eval(name: str, metrics: Dict, rows: List[Dict], suffix: str = "") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{name}{suffix}"
    (RESULTS_DIR / f"eval_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"preds_{tag}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    p = argparse.ArgumentParser(description="Avalia o modelo (base ou com adaptador LoRA)")
    p.add_argument("--name", default="eval", help="nome usado nos arquivos de resultado")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="pasta do adaptador LoRA (omitir = modelo base)")
    p.add_argument("--n", type=int, default=200, help="exemplos de avaliacao em ingles")
    p.add_argument("--pt", action="store_true", help="avalia tambem data/pt_br_eval.jsonl")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    gen_cfg = GenConfig(a.max_new_tokens, a.temperature, a.top_p)
    gen = SQLGenerator.load(a.model, a.adapter, use_4bit=a.use_4bit)

    metrics, rows = evaluate_generator(gen, load_eval_examples(a.n), gen_cfg, a.batch_size, desc="EN")
    save_eval(a.name, metrics, rows)
    print("\n=== Ingles (sql-create-context) ===")
    print(json.dumps(metrics, indent=2))

    if a.pt:
        pt = load_pt_examples("eval")
        if pt:
            m_pt, rows_pt = evaluate_generator(gen, pt, gen_cfg, a.batch_size, desc="PT")
            save_eval(a.name, m_pt, rows_pt, suffix="_pt")
            print("\n=== Portugues (pt_br_eval) ===")
            print(json.dumps(m_pt, indent=2))
    print(f"\nResultados salvos em {RESULTS_DIR}")


if __name__ == "__main__":
    main()
