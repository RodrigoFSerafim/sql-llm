"""Gera graficos e tabela em Markdown a partir de results/ (rode depois dos experimentos).

Uso:  python scripts/plot_results.py
Saidas: docs/img/loss_curves.png, docs/img/experiments.png, docs/img/inference_sweep.png,
        results/experiments.md
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]

    def fmt(v) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            if v == int(v):
                return str(int(v))
            return f"{v:.0e}" if abs(v) < 0.001 else f"{v:.3f}".rstrip("0").rstrip(".")
        return str(v)

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def plot_losses() -> None:
    plt.figure(figsize=(8, 4.5))
    n = 0
    for p in sorted((RESULTS / "logs").glob("*_log.json")):
        name = p.name[: -len("_log.json")]
        log = [e for e in json.loads(p.read_text(encoding="utf-8")) if "loss" in e]
        if log:
            label = name.removeprefix("quick_")
            if name.startswith("quick_"):
                label += " (rápido)"
            plt.plot([e["step"] for e in log], [e["loss"] for e in log], label=label)
            n += 1
    if n:
        plt.xlabel("step")
        plt.ylabel("loss de treino")
        plt.title("Curvas de loss por experimento")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMG / "loss_curves.png", dpi=150)
        print("salvo: docs/img/loss_curves.png")
    plt.close()


def plot_experiments() -> None:
    path = RESULTS / "experiments.csv"
    quick = False
    if not path.exists():
        path = RESULTS / "experiments_quick.csv"
        quick = True
        if not path.exists():
            return
    df = pd.read_csv(path)
    if not quick:
        df = df[~df["run"].str.startswith("quick_")]
    if df.empty:
        return
    (RESULTS / "experiments.md").write_text(to_markdown(df), encoding="utf-8")
    print("salvo: results/experiments.md")
    metrics = [("exact_match", "Exact match"), ("valid_sql", "SQL válido"), ("exec_acc", "Acurácia de execução"),
               ("pt_exec_acc", "Exec. (português)")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.2
    xs = range(len(df))
    for i, (col, label) in enumerate(metrics):
        ax.bar([x + (i - 1.5) * width for x in xs], df[col].fillna(0), width, label=label)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(df["run"], rotation=30, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    title = "Comparação entre experimentos (modelo base vs. variações de LoRA)"
    if quick:
        title = "Teste rápido (amostra reduzida; resultados preliminares)"
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "experiments.png", dpi=150)
    plt.close(fig)
    print("salvo: docs/img/experiments.png")


def plot_inference() -> None:
    path = RESULTS / "inference_sweep.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["config"], df["exec_accuracy"], color="tab:blue", label="Acurácia de execução")
    ax.plot(df["config"], df["valid_sql"], "o--", color="tab:orange", label="SQL válido")
    ax.set_ylim(0, 1.05)
    title = "Efeito de temperature / top_p na geração"
    if "n" in df.columns and df["n"].max() <= 60:
        title += " (teste rápido; n=60)"
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "inference_sweep.png", dpi=150)
    plt.close(fig)
    print("salvo: docs/img/inference_sweep.png")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    IMG.mkdir(parents=True, exist_ok=True)
    plot_losses()
    plot_experiments()
    plot_inference()


if __name__ == "__main__":
    main()
