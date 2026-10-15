"""Carregamento do tokenizer e do modelo base (opcionalmente quantizado em 4 bits)."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GPU_REQUIRED_MSG = (
    "CUDA nao esta disponivel. Este projeto exige GPU NVIDIA: instale o PyTorch com CUDA "
    "(veja docs/GPU_CUDA.md) e confira com `python scripts/check_gpu.py`. "
    "(Apenas para testes sem GPU: defina a variavel de ambiente T2S_ALLOW_CPU=1.)"
)


def allow_cpu() -> bool:
    return os.environ.get("T2S_ALLOW_CPU") == "1"


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def require_gpu() -> None:
    if not torch.cuda.is_available() and not allow_cpu():
        raise RuntimeError(GPU_REQUIRED_MSG)


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name: str, use_4bit: bool = True):
    require_gpu()
    on_gpu = torch.cuda.is_available()
    kwargs = {}
    if on_gpu:
        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        kwargs["device_map"] = {"": 0}
    else:
        dtype, use_4bit = torch.float32, False
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    try:  # transformers recentes usam `dtype`; versoes antigas usam `torch_dtype`
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
