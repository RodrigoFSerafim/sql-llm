"""Informacoes sobre a GPU (usadas no app, no notebook e em scripts/check_gpu.py)."""
import platform

import torch


def gpu_summary() -> str:
    """Resumo de uma linha: nome da GPU e uso de VRAM."""
    if not torch.cuda.is_available():
        return "GPU nao detectada (CUDA indisponivel) - rodando na CPU"
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    used_gb = (total - free) / 1e9
    return f"{props.name} | VRAM {used_gb:.1f} / {total / 1e9:.1f} GB em uso | CUDA {torch.version.cuda}"


def print_gpu_report() -> None:
    print(f"Sistema        : {platform.system()} {platform.release()}")
    print(f"Python         : {platform.python_version()}")
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA (PyTorch) : {torch.version.cuda}")
    print(f"CUDA disponivel: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}          : {p.name}")
            print(f"  VRAM total   : {p.total_memory / 1e9:.1f} GB")
            print(f"  Compute cap. : {p.major}.{p.minor}")
            print(f"  bf16 suportado: {torch.cuda.is_bf16_supported()}")
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes   : {bnb.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"bitsandbytes   : NAO disponivel ({type(e).__name__}) - use --no-use-4bit")
