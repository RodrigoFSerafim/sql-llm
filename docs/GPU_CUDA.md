# GPU e CUDA: passo a passo

O professor pediu: **instalar a versão correta do CUDA para a sua placa**, usar os plugins do VSCode e mostrar a GPU
funcionando. Siga na ordem.

## 1. Descobrir a GPU e o driver

No terminal:

```bash
nvidia-smi
```

Anote **o nome da placa**, a **VRAM** e o campo **"CUDA Version"** no canto superior direito. Esse número é a versão
**máxima** de CUDA que o seu driver suporta. Se o comando não existir, instale/atualize o driver em
<https://www.nvidia.com/drivers>.

## 2. Instalar o PyTorch com a versão de CUDA compatível

As rodas (wheels) do PyTorch já trazem o runtime do CUDA, então normalmente **não é preciso instalar o CUDA Toolkit
inteiro** — basta o driver ser compatível. O que importa é escolher a roda cuja versão de CUDA seja **menor ou igual**
à mostrada no `nvidia-smi`.

1. Abra <https://pytorch.org/get-started/locally/>, escolha *Stable → seu SO → Pip → CUDA x.y*.
2. Rode o comando gerado, por exemplo:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Observações:
- Placas recentes (série RTX 50) exigem rodas de CUDA mais novas (12.8 ou superior). Se o `check_gpu.py` disser
  que a GPU não é suportada, use a versão mais recente da lista do site.
- Se quiser instalar o **CUDA Toolkit** completo (opcional, necessário só para compilar extensões), baixe a versão
  igual à do PyTorch em <https://developer.nvidia.com/cuda-downloads>.

## 3. Instalar o restante

```bash
pip install -r requirements.txt
python scripts/check_gpu.py
```

O resultado esperado inclui `CUDA disponivel: True`, o nome da placa, a VRAM e um benchmark de multiplicação de matrizes.

## 4. Plugins do VSCode

Em *Extensions* (`Ctrl+Shift+X`) instale os plugins indicados pelo professor:

- **GPU Environment**
- **GPU Monitor**

Deixe o painel de monitoramento visível (uso de GPU/VRAM) enquanto roda o treino.

## 5. Print exigido

Tire **um print único do VSCode** mostrando:

1. o terminal com a saída de `python scripts/check_gpu.py` (ou a célula de GPU do notebook);
2. o painel do GPU Monitor / GPU Environment com a placa listada;
3. se possível, o uso de GPU subindo durante `python -m src.train`.

Salve em `docs/img/gpu_vscode.png`.

## 6. Problemas comuns

| Sintoma | Solução |
|---|---|
| `CUDA disponivel: False` | Você instalou o PyTorch sem CUDA (roda `cpu`). Desinstale (`pip uninstall torch`) e reinstale com o `--index-url` da sua versão de CUDA |
| `CUDA out of memory` | `--batch-size 2 --grad-accum 8`, `--max-length 384` ou `--model Qwen/Qwen2.5-0.5B-Instruct`. Feche outros programas que usam a GPU |
| Erro no `bitsandbytes` (Windows) | `pip install -U bitsandbytes`; se persistir, use `--no-use-4bit` |
| Download do modelo lento/falhando | Rode de novo (o Hugging Face retoma). Confira a internet/proxy da rede |
| `bf16 suportado: False` | Normal em placas mais antigas (ex.: GTX 16xx/RTX 20xx); o código usa fp16 automaticamente |
| App e treino ao mesmo tempo estouram a VRAM | Pare o treino/notebook antes de abrir o `app/app.py` |
