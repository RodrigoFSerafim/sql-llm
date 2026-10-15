"""Frontend (Gradio) para gerar SQL a partir de perguntas em linguagem natural.

Uso:
    python app/app.py                                   # usa outputs/baseline/adapter se existir
    python app/app.py --adapter outputs/r64/adapter     # outro adaptador
    python app/app.py --share                           # link publico temporario
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DEFAULT_MODEL, OUTPUTS_DIR, GenConfig  # noqa: E402
from src.gpu import gpu_summary  # noqa: E402
from src.inference import SQLGenerator  # noqa: E402
from src.sample_dbs import (CUSTOM_KEY, db_choices, get_examples, get_schema,  # noqa: E402
                            make_connection)
from src.sqlutils import build_db_from_schema, is_safe_select, run_query  # noqa: E402

GEN: SQLGenerator = None


def on_db_change(db_key):
    examples = get_examples(db_key)
    schema = get_schema(db_key)
    return (
        gr.Textbox(value=schema, interactive=(db_key == CUSTOM_KEY)),
        gr.Dropdown(choices=examples, value=None),
    )


def on_example(example):
    return example or ""


def run_sql(db_key, schema, sql):
    """Executa o SQL gerado (somente SELECT) no banco de exemplo ou em dados ficticios."""
    if not sql.strip():
        return None, "Nenhum SQL gerado."
    if not is_safe_select(sql):
        return None, "Execução bloqueada: só consultas SELECT (uma instrução) são executadas."
    try:
        if db_key == CUSTOM_KEY:
            conn = build_db_from_schema(schema, sql, n_rows=20)
            if conn is None:
                return None, "Não foi possível criar o esquema informado no SQLite."
            note = "executado em dados fictícios gerados a partir do seu esquema"
        else:
            conn = make_connection(db_key)
            note = "executado no banco de exemplo"
        cols, rows = run_query(conn, sql)
        conn.close()
        return pd.DataFrame(rows, columns=cols), f"{len(rows)} linha(s) - {note}"
    except Exception as e:  # noqa: BLE001
        return None, f"Erro ao executar o SQL: {e}"


def on_generate(db_key, schema, question, compare, temperature, top_p, max_new_tokens):
    if not question.strip():
        return "", "", None, "Digite uma pergunta."
    if not schema.strip():
        return "", "", None, "Informe o esquema (CREATE TABLE ...) do banco."
    gen_cfg = GenConfig(int(max_new_tokens), float(temperature), float(top_p))
    t0 = time.time()
    sql = GEN.generate_one(schema, question, gen_cfg, use_adapter=True)
    elapsed = time.time() - t0
    sql_base = ""
    if compare and GEN.has_adapter:
        sql_base = GEN.generate_one(schema, question, gen_cfg, use_adapter=False)
    df, info = run_sql(db_key, schema, sql)
    return sql, sql_base, df, f"Geração: {elapsed:.2f}s | {info}"


def build_ui(model_label: str) -> gr.Blocks:
    first = db_choices()[0]
    with gr.Blocks(title="Texto para SQL") as demo:
        gr.Markdown("# Texto → SQL com LLM local\n"
                    "Escreva uma pergunta em português (ou inglês) e o modelo gera a consulta SQL.")
        gr.Markdown(f"**Modelo:** {model_label}  \n**GPU:** {gpu_summary()}")
        with gr.Row():
            with gr.Column(scale=1):
                db = gr.Dropdown(db_choices(), value=first, label="Banco de exemplo")
                schema = gr.Textbox(value=get_schema(first), label="Esquema (CREATE TABLE)", lines=7,
                                    interactive=False)
                example = gr.Dropdown(get_examples(first), value=None, label="Perguntas de exemplo")
                question = gr.Textbox(label="Sua pergunta", lines=2,
                                      placeholder="Ex.: Quais produtos custam mais de 500?")
                with gr.Accordion("Parâmetros de geração", open=False):
                    temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05,
                                            label="temperature (0 = determinístico)")
                    top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                    max_new = gr.Slider(32, 256, value=128, step=16, label="max_new_tokens")
                    compare = gr.Checkbox(value=False, label="Comparar com o modelo base (sem fine-tuning)")
                btn = gr.Button("Gerar SQL", variant="primary")
            with gr.Column(scale=1):
                sql_out = gr.Code(label="SQL gerado (modelo fine-tuned)" if GEN.has_adapter
                                  else "SQL gerado (modelo base - adaptador não encontrado)",
                                  language="sql")
                sql_base = gr.Code(label="SQL do modelo base (sem LoRA)", language="sql")
                table = gr.Dataframe(label="Resultado da consulta", interactive=False, wrap=True)
                status = gr.Textbox(label="Status", interactive=False)

        db.change(on_db_change, db, [schema, example])
        example.change(on_example, example, question)
        inputs = [db, schema, question, compare, temperature, top_p, max_new]
        outputs = [sql_out, sql_base, table, status]
        btn.click(on_generate, inputs, outputs)
        question.submit(on_generate, inputs, outputs)
    return demo


def main() -> None:
    global GEN
    p = argparse.ArgumentParser(description="App Gradio texto -> SQL")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=str(OUTPUTS_DIR / "baseline" / "adapter"))
    p.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    a = p.parse_args()

    adapter = a.adapter if Path(a.adapter).exists() else None
    if adapter is None:
        print(f"[aviso] adaptador '{a.adapter}' nao encontrado - usando o modelo base. "
              "Treine com: python -m src.train")
    print("Carregando o modelo (pode levar alguns minutos na primeira vez)...")
    GEN = SQLGenerator.load(a.model, adapter, use_4bit=a.use_4bit)
    label = f"{a.model} + LoRA ({adapter})" if adapter else f"{a.model} (base, sem fine-tuning)"
    demo = build_ui(label)
    demo.queue().launch(server_name="127.0.0.1", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
