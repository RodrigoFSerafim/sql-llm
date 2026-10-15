"""Geracao de SQL com o modelo base e/ou com o adaptador LoRA treinado."""
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel

from .config import DEFAULT_MODEL, GenConfig
from .modeling import load_base_model, load_tokenizer
from .prompts import clean_sql, render_prompt


class SQLGenerator:
    def __init__(self, model, tokenizer, has_adapter: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.has_adapter = has_adapter

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL, adapter_path: Optional[str] = None,
             use_4bit: bool = True) -> "SQLGenerator":
        tokenizer = load_tokenizer(model_name)
        tokenizer.padding_side = "left"  # necessario para geracao em lote
        model = load_base_model(model_name, use_4bit=use_4bit)
        has_adapter = False
        if adapter_path:
            if not Path(adapter_path).exists():
                raise FileNotFoundError(f"Adaptador nao encontrado: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            has_adapter = True
        model.eval()
        return cls(model, tokenizer, has_adapter)

    @torch.inference_mode()
    def generate_batch(self, schemas: List[str], questions: List[str], gen: Optional[GenConfig] = None,
                       use_adapter: bool = True) -> List[str]:
        gen = gen or GenConfig()
        prompts = [render_prompt(self.tokenizer, s, q) for s, q in zip(schemas, questions)]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(self.model.device)
        kwargs = dict(max_new_tokens=gen.max_new_tokens,
                      pad_token_id=self.tokenizer.pad_token_id,
                      eos_token_id=self.tokenizer.eos_token_id)
        if gen.temperature and gen.temperature > 0:
            kwargs.update(do_sample=True, temperature=gen.temperature, top_p=gen.top_p)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        # Sem adaptador = modelo base (zero-shot), util para comparar antes/depois do fine-tuning
        ctx = self.model.disable_adapter() if (self.has_adapter and not use_adapter) else nullcontext()
        with ctx:
            out = self.model.generate(**enc, **kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [clean_sql(t) for t in texts]

    def generate_one(self, schema: str, question: str, gen: Optional[GenConfig] = None,
                     use_adapter: bool = True) -> str:
        return self.generate_batch([schema], [question], gen, use_adapter)[0]

    def free(self) -> None:
        """Libera a memoria da GPU."""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
