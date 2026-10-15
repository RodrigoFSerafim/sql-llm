"""Carregamento do dataset, exemplos em portugues e tokenizacao para o treino."""
import json
import random
from dataclasses import dataclass
from typing import Dict, List

from .config import DATA_DIR, DATASET_NAME, EVAL_POOL
from .prompts import render_prompt

KEYS = ("question", "context", "answer")


def _load_hf_shuffled(seed: int):
    from datasets import load_dataset

    return load_dataset(DATASET_NAME, split="train").shuffle(seed=seed)


def _rows(ds, start: int, stop: int) -> List[Dict[str, str]]:
    if stop <= start:
        return []
    return [{k: r[k] for k in KEYS} for r in ds.select(range(start, stop))]


def load_pt_examples(split: str = "train") -> List[Dict[str, str]]:
    """Exemplos em portugues escritos a mao (data/pt_br_train.jsonl e pt_br_eval.jsonl)."""
    path = DATA_DIR / f"pt_br_{split}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_examples(eval_samples: int = 200, seed: int = 42) -> List[Dict[str, str]]:
    """Conjunto de avaliacao fixo (mesmos exemplos em qualquer experimento)."""
    if eval_samples > EVAL_POOL:
        raise ValueError(f"eval_samples deve ser <= {EVAL_POOL}")
    return _rows(_load_hf_shuffled(seed), 0, eval_samples)


def load_sql_dataset(train_samples: int = 3000, eval_samples: int = 200, seed: int = 42,
                     pt_repeat: int = 0) -> Dict[str, List[Dict[str, str]]]:
    """Retorna {'train': [...], 'eval': [...]} com dicts question/context/answer."""
    if eval_samples > EVAL_POOL:
        raise ValueError(f"eval_samples deve ser <= {EVAL_POOL}")
    ds = _load_hf_shuffled(seed)
    if EVAL_POOL + train_samples > len(ds):
        raise ValueError("train_samples maior que o dataset disponivel")
    train = _rows(ds, EVAL_POOL, EVAL_POOL + train_samples)
    evaluation = _rows(ds, 0, eval_samples)
    if pt_repeat > 0:
        train = train + load_pt_examples("train") * pt_repeat
        random.Random(seed).shuffle(train)
    return {"train": train, "eval": evaluation}


def build_train_features(examples: List[Dict[str, str]], tokenizer, max_length: int):
    """Tokeniza (prompt + resposta). A loss e calculada so nos tokens da resposta."""
    feats, dropped = [], 0
    for ex in examples:
        prompt_ids = tokenizer(render_prompt(tokenizer, ex["context"], ex["question"]),
                               add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(ex["answer"].strip() + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]
        ids = prompt_ids + answer_ids
        if len(ids) > max_length:
            dropped += 1
            continue
        feats.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        })
    return feats, dropped


@dataclass
class DataCollator:
    """Faz padding a direita; posicoes de padding recebem label -100 (ignoradas na loss)."""
    pad_token_id: int

    def __call__(self, features):
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            pad = longest - len(f["input_ids"])
            ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            mask.append(list(f["attention_mask"]) + [0] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
