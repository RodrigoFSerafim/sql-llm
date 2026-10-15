"""Configuracoes centrais do projeto: caminhos, modelo padrao e hiperparametros."""
import argparse
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "b-mc2/sql-create-context"

# As primeiras EVAL_POOL linhas (apos um shuffle com seed fixa) ficam reservadas
# para avaliacao; o treino sempre usa linhas a partir dai. Assim o conjunto de
# avaliacao e o mesmo em todos os experimentos e nunca vaza para o treino.
EVAL_POOL = 1000


@dataclass
class TrainConfig:
    run_name: str = "baseline"
    model_name: str = DEFAULT_MODEL
    epochs: float = 1.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 512
    train_samples: int = 3000
    pt_repeat: int = 3  # quantas vezes repetir os exemplos em portugues (data/pt_br_train.jsonl)
    use_4bit: bool = True  # QLoRA (4 bits). Use --no-use-4bit para LoRA em fp16/bf16
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    seed: int = 42

    @property
    def output_dir(self) -> Path:
        return OUTPUTS_DIR / self.run_name

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "adapter"


@dataclass
class GenConfig:
    """Parametros de geracao (inferencia)."""
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 = decodificacao gulosa (deterministica)
    top_p: float = 1.0


def add_args_from_dataclass(parser: argparse.ArgumentParser, cls) -> None:
    """Cria um argumento de linha de comando para cada campo do dataclass."""
    for f in fields(cls):
        names = ["--" + f.name.replace("_", "-")]
        if f.name == "model_name":
            names.append("--model")  # atalho, igual ao usado em evaluate/experiments/app
        if f.type is bool:
            parser.add_argument(*names, dest=f.name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(*names, dest=f.name, type=f.type, default=f.default)


def from_args(cls, args: argparse.Namespace):
    return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})
