"""Montagem do prompt (esquema + pergunta) e limpeza da resposta do modelo."""
import re

SYSTEM_PROMPT = (
    "Você é um especialista em SQL (dialeto SQLite). "
    "Dado o esquema do banco de dados e uma pergunta em linguagem natural, "
    "responda APENAS com a consulta SQL correspondente, sem explicações."
)


def build_user_prompt(schema: str, question: str) -> str:
    return f"### Esquema:\n{schema.strip()}\n\n### Pergunta:\n{question.strip()}"


def build_messages(schema: str, question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(schema, question)},
    ]


def render_prompt(tokenizer, schema: str, question: str) -> str:
    """Texto do prompt ja com o marcador de inicio da resposta do assistente."""
    messages = build_messages(schema, question)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Alguns modelos nao aceitam o papel 'system': funde no turno do usuario.
        merged = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + messages[1]["content"]}]
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def clean_sql(text: str) -> str:
    """Remove cercas de markdown e texto extra; devolve so a primeira instrucao SQL."""
    text = (text or "").strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        text = m.group(1).strip()
    if ";" in text:
        text = text.split(";")[0]
    return text.strip()
