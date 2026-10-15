"""Fine-tuning (QLoRA/LoRA) do modelo para texto -> SQL.

Uso:
    python -m src.train --run-name baseline --epochs 1 --lora-r 16 --learning-rate 2e-4
"""
import argparse
import gc
import json
import math
import time
from dataclasses import asdict
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, set_seed

from .config import RESULTS_DIR, TrainConfig, add_args_from_dataclass, from_args
from .data import DataCollator, build_train_features, load_sql_dataset
from .modeling import bf16_supported, load_base_model, load_tokenizer, require_gpu

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def run_training(cfg: TrainConfig, data: Optional[dict] = None) -> dict:
    """Treina o adaptador LoRA e salva em outputs/<run_name>/adapter. Retorna metricas do treino."""
    require_gpu()
    on_gpu = torch.cuda.is_available()
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = load_sql_dataset(cfg.train_samples, 0, cfg.seed, cfg.pt_repeat)

    tokenizer = load_tokenizer(cfg.model_name)
    feats, dropped = build_train_features(data["train"], tokenizer, cfg.max_length)
    train_ds = Dataset.from_list(feats)
    print(f"[{cfg.run_name}] exemplos de treino: {len(train_ds)} (descartados por tamanho: {dropped})")

    use_4bit = cfg.use_4bit and on_gpu
    model = load_base_model(cfg.model_name, use_4bit=use_4bit)
    model.config.use_cache = False
    gc_kwargs = {"use_reentrant": False}
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs=gc_kwargs)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()

    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[{cfg.run_name}] parametros treinaveis: {trainable:,} ({100 * trainable / total:.2f}%)")

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    bf16 = on_gpu and bf16_supported()
    args = TrainingArguments(
        output_dir=str(cfg.output_dir / "trainer"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        bf16=bf16,
        fp16=on_gpu and not bf16,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=cfg.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollator(tokenizer.pad_token_id))

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0.0

    model.save_pretrained(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))
    log_history = trainer.state.log_history
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    info = {
        "run_name": cfg.run_name,
        "adapter_dir": str(cfg.adapter_dir),
        "n_train_examples": len(train_ds),
        "dropped_long": dropped,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_runtime_s": round(elapsed, 1),
        "mean_train_loss": round(float(result.training_loss), 4),
        "final_loss": round(sum(losses[-5:]) / max(1, len(losses[-5:])), 4) if losses else None,
        "peak_gpu_gb": round(peak_gb, 2),
        "log_history": log_history,
    }
    (cfg.output_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{cfg.run_name}_log.json").write_text(
        json.dumps(log_history, indent=2), encoding="utf-8")

    del trainer, model
    gc.collect()
    if on_gpu:
        torch.cuda.empty_cache()
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA para texto -> SQL")
    add_args_from_dataclass(parser, TrainConfig)
    cfg = from_args(TrainConfig, parser.parse_args())
    info = run_training(cfg)
    print("\n=== Treino concluido ===")
    for k in ("run_name", "n_train_examples", "trainable_params", "train_runtime_s",
              "mean_train_loss", "final_loss", "peak_gpu_gb", "adapter_dir"):
        print(f"{k:>18}: {info[k]}")
    print(f"\nProximo passo: python -m src.evaluate --name {cfg.run_name} "
          f"--adapter {info['adapter_dir']} --pt")


if __name__ == "__main__":
    main()
