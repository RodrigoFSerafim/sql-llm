"""Verifica a GPU/CUDA e roda um pequeno benchmark.

Rode este script no terminal do VSCode (com o painel do GPU Monitor/GPU Environment visivel)
e tire o print exigido no trabalho:  python scripts/check_gpu.py
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import torch  # noqa: E402

from src.gpu import print_gpu_report  # noqa: E402


def bench(device: str, size: int = 4096, iters: int = 10) -> float:
    dtype = torch.float16 if device == "cuda" else torch.float32
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main() -> None:
    print("=" * 60)
    print_gpu_report()
    print("=" * 60)
    if torch.cuda.is_available():
        t_gpu = bench("cuda")
        t_cpu = bench("cpu", size=2048, iters=3) * 8  # 2048 -> 4096 = 8x mais operacoes
        print(f"Multiplicacao 4096x4096  GPU: {t_gpu * 1000:.1f} ms | CPU (estimado): {t_cpu * 1000:.0f} ms"
              f" | speedup ~{t_cpu / t_gpu:.0f}x")
        print("\n>>> GPU funcionando com PyTorch. <<<")
    else:
        print("\n>>> CUDA NAO disponivel. Veja docs/GPU_CUDA.md <<<")
    print("\n--- nvidia-smi ---")
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("nvidia-smi nao encontrado (instale o driver NVIDIA).")


if __name__ == "__main__":
    main()
