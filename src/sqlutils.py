"""Utilitarios SQL: normalizacao, banco SQLite com dados ficticios e metricas de execucao."""
import random
import re
import sqlite3
from collections import Counter
from typing import List, Optional, Tuple

_QUOTED = re.compile(r"('(?:[^']|'')*'|\"[^\"]*\")")
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.I
)
_WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "red", "blue", "green",
          "north", "south", "east", "west", "john", "maria", "paulo", "ana"]


def normalize_sql(sql: str) -> str:
    """Normaliza espacos e caixa (fora de literais entre aspas) para o exact match."""
    sql = (sql or "").strip().rstrip(";").strip()
    out = []
    for i, part in enumerate(_QUOTED.split(sql)):
        if i % 2 == 1:  # literal entre aspas: mantem intacto
            out.append(part)
        else:
            part = re.sub(r"\s*(>=|<=|<>|!=|[=<>(),])\s*", r"\1", part.lower())
            out.append(re.sub(r"\s+", " ", part).strip())
    return "".join(out).strip()


def is_safe_select(sql: str) -> bool:
    """So aceita UMA instrucao SELECT/WITH somente leitura."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return False
    stripped = _QUOTED.sub("''", s)
    if ";" in stripped:
        return False
    if not re.match(r"^(select|with)\b", s, re.I):
        return False
    return not _FORBIDDEN.search(stripped)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literals(sql: str) -> Tuple[List[str], List[float]]:
    texts = [t.replace("''", "'") for t in re.findall(r"'((?:[^']|'')*)'", sql or "")]
    no_str = _QUOTED.sub(" ", sql or "")
    nums = [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", no_str)]
    return texts, nums


def _rand_value(typ: str, rng: random.Random, text_pool: list, num_pool: list):
    use_pool = bool(num_pool) and rng.random() < 0.5
    if "INT" in typ:
        return int(rng.choice(num_pool)) if use_pool else rng.randint(0, 20)
    if any(k in typ for k in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return float(rng.choice(num_pool)) if use_pool else round(rng.uniform(0, 100), 1)
    if rng.random() < 0.7:
        return rng.choice(text_pool)
    return rng.choice(_WORDS) + str(rng.randint(0, 9))


def build_db_from_schema(schema: str, ref_sql: str = "", n_rows: int = 40,
                         seed: int = 0) -> Optional[sqlite3.Connection]:
    """Cria um SQLite em memoria a partir dos CREATE TABLE e o preenche com dados ficticios.

    Os literais da consulta de referencia (textos e numeros) entram no conjunto de
    valores sorteados, para que filtros como WHERE nome = 'Ana' retornem linhas.
    Retorna None se o esquema nao puder ser criado no SQLite.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        if not tables:
            raise sqlite3.Error("sem tabelas")
        rng = random.Random(seed)
        texts, nums = _literals(ref_sql)
        text_pool = texts + _WORDS
        num_pool: List[float] = []
        for n in nums:
            num_pool += [n - 1, n, n + 1]
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({_q(t)})").fetchall()
            rows = [[_rand_value((c[2] or "").upper(), rng, text_pool, num_pool) for c in cols]
                    for _ in range(n_rows)]
            conn.executemany(
                f"INSERT OR IGNORE INTO {_q(t)} VALUES ({','.join('?' * len(cols))})", rows)
        conn.commit()
        return conn
    except sqlite3.Error:
        conn.close()
        return None


def run_query(conn: sqlite3.Connection, sql: str, max_rows: int = 1000):
    cur = conn.execute(sql)
    rows = cur.fetchmany(max_rows)
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, rows


def _norm_row(row):
    return tuple(round(v, 4) if isinstance(v, float) else v for v in row)


def execution_match(schema: str, gold_sql: str, pred_sql: str,
                    seeds=(0, 1)) -> Tuple[Optional[bool], Optional[bool]]:
    """Executa a consulta de referencia e a prevista em bancos ficticios.

    Retorna (sql_valido, resultados_iguais); (None, None) quando a referencia nao
    pode ser avaliada (esquema nao cria no SQLite ou a referencia falha).
    """
    all_match = True
    for seed in seeds:
        conn = build_db_from_schema(schema, gold_sql, seed=seed)
        if conn is None:
            return None, None
        try:
            try:
                _, gold_rows = run_query(conn, gold_sql)
            except sqlite3.Error:
                return None, None
            if not is_safe_select(pred_sql):
                return False, False
            try:
                _, pred_rows = run_query(conn, pred_sql)
            except sqlite3.Error:
                return False, False
            g = [_norm_row(r) for r in gold_rows]
            p = [_norm_row(r) for r in pred_rows]
            if re.search(r"\border\s+by\b", gold_sql, re.I):
                same = g == p
            else:
                same = Counter(g) == Counter(p)
            all_match = all_match and same
        finally:
            conn.close()
    return True, all_match
