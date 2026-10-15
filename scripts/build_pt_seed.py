"""Gera data/pt_br_train.jsonl e data/pt_br_eval.jsonl a partir dos exemplos abaixo.

Cada SQL de referencia e VALIDADO: executado em um SQLite com dados ficticios antes de ser salvo.
Para ampliar o dataset em portugues, acrescente tuplas (pergunta, SQL) e rode:
    python scripts/build_pt_seed.py
Os esquemas de EVAL nao aparecem no treino (avaliacao em esquemas nunca vistos).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.config import DATA_DIR  # noqa: E402
from src.sqlutils import build_db_from_schema, run_query  # noqa: E402

TRAIN = [
    ("CREATE TABLE produtos (id INTEGER, nome VARCHAR, preco FLOAT, categoria VARCHAR, estoque INTEGER)", [
        ("Liste o nome de todos os produtos.", "SELECT nome FROM produtos"),
        ("Quais produtos custam mais de 100 reais?", "SELECT nome FROM produtos WHERE preco > 100"),
        ("Quantos produtos existem na categoria 'Eletrônicos'?", "SELECT COUNT(*) FROM produtos WHERE categoria = 'Eletrônicos'"),
        ("Qual é o preço médio dos produtos?", "SELECT AVG(preco) FROM produtos"),
        ("Mostre o nome e o preço do produto mais caro.", "SELECT nome, preco FROM produtos ORDER BY preco DESC LIMIT 1"),
        ("Qual a quantidade total em estoque por categoria?", "SELECT categoria, SUM(estoque) FROM produtos GROUP BY categoria"),
        ("Liste os produtos sem estoque.", "SELECT nome FROM produtos WHERE estoque = 0"),
    ]),
    ("CREATE TABLE students (id INTEGER, name VARCHAR, class_name VARCHAR, grade FLOAT, age INTEGER)", [
        ("Quais alunos da turma 4ESPG tiveram nota acima de 8?", "SELECT name FROM students WHERE class_name = '4ESPG' AND grade > 8"),
        ("Qual é a maior nota registrada?", "SELECT MAX(grade) FROM students"),
        ("Quantos alunos existem em cada turma?", "SELECT class_name, COUNT(*) FROM students GROUP BY class_name"),
        ("Qual a idade média dos alunos?", "SELECT AVG(age) FROM students"),
        ("Liste os 5 alunos com as melhores notas.", "SELECT name, grade FROM students ORDER BY grade DESC LIMIT 5"),
        ("Quais alunos têm menos de 18 anos?", "SELECT name FROM students WHERE age < 18"),
        ("Mostre a nota média da turma 3ESPG.", "SELECT AVG(grade) FROM students WHERE class_name = '3ESPG'"),
    ]),
    ("CREATE TABLE employees (id INTEGER, name VARCHAR, department VARCHAR, salary FLOAT, hire_year INTEGER)", [
        ("Quem ganha mais de 5000?", "SELECT name FROM employees WHERE salary > 5000"),
        ("Qual é o salário médio por departamento?", "SELECT department, AVG(salary) FROM employees GROUP BY department"),
        ("Quantos funcionários foram contratados em 2020?", "SELECT COUNT(*) FROM employees WHERE hire_year = 2020"),
        ("Quem é o funcionário com o maior salário?", "SELECT name FROM employees ORDER BY salary DESC LIMIT 1"),
        ("Liste os funcionários do departamento de Vendas em ordem alfabética.", "SELECT name FROM employees WHERE department = 'Vendas' ORDER BY name ASC"),
        ("Qual o total gasto com salários?", "SELECT SUM(salary) FROM employees"),
        ("Quais departamentos têm mais de 3 funcionários?", "SELECT department FROM employees GROUP BY department HAVING COUNT(*) > 3"),
    ]),
    ("CREATE TABLE clientes (id INTEGER, nome VARCHAR, cidade VARCHAR);\n"
     "CREATE TABLE pedidos (id INTEGER, cliente_id INTEGER, valor FLOAT, data_pedido VARCHAR)", [
        ("Liste o nome dos clientes de São Paulo.", "SELECT nome FROM clientes WHERE cidade = 'São Paulo'"),
        ("Qual o valor total de todos os pedidos?", "SELECT SUM(valor) FROM pedidos"),
        ("Mostre o nome de cada cliente e o valor dos seus pedidos.", "SELECT clientes.nome, pedidos.valor FROM clientes JOIN pedidos ON clientes.id = pedidos.cliente_id"),
        ("Quantos pedidos cada cliente fez?", "SELECT clientes.nome, COUNT(*) FROM clientes JOIN pedidos ON clientes.id = pedidos.cliente_id GROUP BY clientes.nome"),
        ("Quais clientes fizeram pedidos acima de 1000?", "SELECT DISTINCT clientes.nome FROM clientes JOIN pedidos ON clientes.id = pedidos.cliente_id WHERE pedidos.valor > 1000"),
        ("Quantos clientes existem em cada cidade?", "SELECT cidade, COUNT(*) FROM clientes GROUP BY cidade"),
        ("Qual foi o maior pedido?", "SELECT MAX(valor) FROM pedidos"),
    ]),
    ("CREATE TABLE movies (id INTEGER, title VARCHAR, genre VARCHAR, release_year INTEGER, rating FLOAT)", [
        ("Quais filmes foram lançados depois de 2015?", "SELECT title FROM movies WHERE release_year > 2015"),
        ("Qual a nota média dos filmes do gênero Comedy?", "SELECT AVG(rating) FROM movies WHERE genre = 'Comedy'"),
        ("Quantos filmes existem por gênero?", "SELECT genre, COUNT(*) FROM movies GROUP BY genre"),
        ("Qual é o filme mais bem avaliado?", "SELECT title FROM movies ORDER BY rating DESC LIMIT 1"),
        ("Liste os títulos dos filmes com nota menor que 5.", "SELECT title FROM movies WHERE rating < 5"),
        ("Qual o ano de lançamento do filme mais antigo?", "SELECT MIN(release_year) FROM movies"),
        ("Mostre títulos e notas dos filmes de Drama, da maior para a menor nota.", "SELECT title, rating FROM movies WHERE genre = 'Drama' ORDER BY rating DESC"),
    ]),
    ("CREATE TABLE veiculos (placa VARCHAR, modelo VARCHAR, marca VARCHAR, ano INTEGER, preco FLOAT)", [
        ("Quais veículos são da marca Toyota?", "SELECT modelo FROM veiculos WHERE marca = 'Toyota'"),
        ("Quantos veículos são anteriores a 2010?", "SELECT COUNT(*) FROM veiculos WHERE ano < 2010"),
        ("Qual o preço médio dos veículos por marca?", "SELECT marca, AVG(preco) FROM veiculos GROUP BY marca"),
        ("Qual é o veículo mais barato?", "SELECT modelo FROM veiculos ORDER BY preco ASC LIMIT 1"),
        ("Liste a placa e o modelo dos veículos de 2020 em diante.", "SELECT placa, modelo FROM veiculos WHERE ano >= 2020"),
        ("Qual o ano do veículo mais novo?", "SELECT MAX(ano) FROM veiculos"),
        ("Quais marcas têm mais de 2 veículos cadastrados?", "SELECT marca FROM veiculos GROUP BY marca HAVING COUNT(*) > 2"),
    ]),
]

EVAL = [
    ("CREATE TABLE books (id INTEGER, title VARCHAR, author VARCHAR, pages INTEGER, price FLOAT)", [
        ("Quantos livros têm mais de 300 páginas?", "SELECT COUNT(*) FROM books WHERE pages > 300"),
        ("Liste os títulos dos livros escritos por 'Machado de Assis'.", "SELECT title FROM books WHERE author = 'Machado de Assis'"),
        ("Qual é o preço médio dos livros?", "SELECT AVG(price) FROM books"),
        ("Qual autor tem mais livros?", "SELECT author FROM books GROUP BY author ORDER BY COUNT(*) DESC LIMIT 1"),
        ("Mostre o título do livro mais caro.", "SELECT title FROM books ORDER BY price DESC LIMIT 1"),
        ("Qual o total de páginas por autor?", "SELECT author, SUM(pages) FROM books GROUP BY author"),
    ]),
    ("CREATE TABLE jogadores (id INTEGER, nome VARCHAR, clube VARCHAR, gols INTEGER, idade INTEGER)", [
        ("Quem marcou mais de 10 gols?", "SELECT nome FROM jogadores WHERE gols > 10"),
        ("Quantos jogadores tem o clube Flamengo?", "SELECT COUNT(*) FROM jogadores WHERE clube = 'Flamengo'"),
        ("Qual a média de gols por clube?", "SELECT clube, AVG(gols) FROM jogadores GROUP BY clube"),
        ("Qual é o jogador mais jovem?", "SELECT nome FROM jogadores ORDER BY idade ASC LIMIT 1"),
        ("Qual o total de gols marcados?", "SELECT SUM(gols) FROM jogadores"),
        ("Liste os nomes dos jogadores com idade entre 20 e 30 anos.", "SELECT nome FROM jogadores WHERE idade BETWEEN 20 AND 30"),
    ]),
]


def build(groups):
    out = []
    for schema, items in groups:
        for question, sql in items:
            conn = build_db_from_schema(schema, sql)
            if conn is None:
                raise SystemExit(f"Esquema invalido no SQLite: {schema}")
            try:
                run_query(conn, sql)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"SQL invalido ({e}): {sql}")
            finally:
                conn.close()
            out.append({"question": question, "context": schema, "answer": sql})
    return out


def write(name, rows):
    path = DATA_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(rows)} exemplos")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    write("pt_br_train.jsonl", build(TRAIN))
    write("pt_br_eval.jsonl", build(EVAL))
