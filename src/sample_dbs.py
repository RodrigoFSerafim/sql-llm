"""Bancos SQLite de exemplo (em memoria) usados pelo frontend para executar o SQL gerado."""
import random
import sqlite3
from typing import Dict, List, Tuple

CUSTOM_KEY = "Personalizado (cole seu esquema)"

_rng = random.Random(7)


def _loja_rows():
    customers = [(1, "Ana Souza", "São Paulo"), (2, "Bruno Lima", "Rio de Janeiro"),
                 (3, "Carla Dias", "Belo Horizonte"), (4, "Diego Alves", "Curitiba"),
                 (5, "Elisa Rocha", "São Paulo"), (6, "Fábio Costa", "Porto Alegre"),
                 (7, "Gabriela Nunes", "Curitiba"), (8, "Hugo Martins", "São Paulo"),
                 (9, "Isabela Reis", "Rio de Janeiro"), (10, "João Pereira", "Belo Horizonte"),
                 (11, "Karina Melo", "São Paulo"), (12, "Lucas Ferreira", "Porto Alegre")]
    products = [(1, "Notebook", 4500.0, "Eletrônicos", 12), (2, "Mouse", 80.0, "Acessórios", 150),
                (3, "Teclado", 250.0, "Acessórios", 70), (4, "Monitor", 1200.0, "Eletrônicos", 25),
                (5, "Cadeira Gamer", 900.0, "Móveis", 10), (6, "Mesa", 600.0, "Móveis", 8),
                (7, "Headset", 320.0, "Acessórios", 40), (8, "Smartphone", 2800.0, "Eletrônicos", 30),
                (9, "Webcam", 350.0, "Acessórios", 0), (10, "Impressora", 750.0, "Eletrônicos", 5)]
    orders = [(i, _rng.randint(1, 12), _rng.randint(1, 10), _rng.randint(1, 5),
               f"2025-{_rng.randint(1, 12):02d}-{_rng.randint(1, 28):02d}") for i in range(1, 41)]
    return {"customers": customers, "products": products, "orders": orders}


def _escola_rows():
    names = ["Alice", "Bernardo", "Camila", "Daniel", "Eduarda", "Felipe", "Giovana", "Henrique",
             "Íris", "Joaquim", "Larissa", "Mateus", "Natália", "Otávio", "Paula", "Rafael",
             "Sofia", "Thiago", "Valentina", "Wesley"]
    courses = ["Engenharia de Software", "Ciência de Dados", "Redes"]
    students = [(i + 1, n, _rng.choice(courses), round(_rng.uniform(4, 10), 1), _rng.choice([2022, 2023, 2024, 2025]))
                for i, n in enumerate(names)]
    return {"students": students}


def _rh_rows():
    names = ["Amanda", "Breno", "Cláudia", "Davi", "Eliane", "Fernando", "Gisele", "Heitor",
             "Ingrid", "Jorge", "Kelly", "Leonardo", "Marina", "Nelson", "Olga", "Pedro"]
    roles = ["Analista", "Desenvolvedor", "Gerente", "Designer"]
    cities = ["São Paulo", "Curitiba", "Recife"]
    salaries = {"Analista": (4000, 7000), "Desenvolvedor": (6000, 12000),
                "Gerente": (12000, 20000), "Designer": (4500, 8000)}
    employees = []
    for i, n in enumerate(names):
        role = _rng.choice(roles)
        lo, hi = salaries[role]
        employees.append((i + 1, n, role, float(_rng.randrange(lo, hi, 100)), _rng.choice(cities)))
    return {"employees": employees}


def _biblioteca_rows():
    livros = [(1, "Dom Casmurro", "Machado de Assis", 1899, 256),
              (2, "Memórias Póstumas de Brás Cubas", "Machado de Assis", 1881, 240),
              (3, "Quincas Borba", "Machado de Assis", 1891, 300),
              (4, "O Cortiço", "Aluísio Azevedo", 1890, 280),
              (5, "Iracema", "José de Alencar", 1865, 150),
              (6, "O Guarani", "José de Alencar", 1857, 400),
              (7, "Senhora", "José de Alencar", 1875, 260),
              (8, "Vidas Secas", "Graciliano Ramos", 1938, 176),
              (9, "São Bernardo", "Graciliano Ramos", 1934, 220),
              (10, "Grande Sertão: Veredas", "Guimarães Rosa", 1956, 608),
              (11, "A Hora da Estrela", "Clarice Lispector", 1977, 96),
              (12, "Capitães da Areia", "Jorge Amado", 1937, 280),
              (13, "O Ateneu", "Raul Pompeia", 1888, 220),
              (14, "Macunaíma", "Mário de Andrade", 1928, 200)]
    return {"livros": livros}


# chave -> (esquema, linhas por tabela, [(pergunta de exemplo, SQL de referencia)])
SAMPLE_DBS: Dict[str, dict] = {
    "Loja (customers, products, orders)": {
        "schema": (
            "CREATE TABLE customers (id INTEGER, name VARCHAR, city VARCHAR);\n"
            "CREATE TABLE products (id INTEGER, name VARCHAR, price FLOAT, category VARCHAR, stock INTEGER);\n"
            "CREATE TABLE orders (id INTEGER, customer_id INTEGER, product_id INTEGER, quantity INTEGER, order_date VARCHAR);"
        ),
        "rows": _loja_rows(),
        "examples": [
            ("Quais produtos custam mais de 500?", "SELECT name FROM products WHERE price > 500"),
            ("Quantos clientes existem em cada cidade?", "SELECT city, COUNT(*) FROM customers GROUP BY city"),
            ("Qual é o produto mais caro?", "SELECT name FROM products ORDER BY price DESC LIMIT 1"),
            ("Quais clientes moram em São Paulo?", "SELECT name FROM customers WHERE city = 'São Paulo'"),
            ("Quantos pedidos foram feitos?", "SELECT COUNT(*) FROM orders"),
            ("Qual a quantidade total vendida de cada produto?",
             "SELECT products.name, SUM(orders.quantity) FROM products JOIN orders ON products.id = orders.product_id GROUP BY products.name"),
        ],
    },
    "Escola (students)": {
        "schema": "CREATE TABLE students (id INTEGER, name VARCHAR, course VARCHAR, score FLOAT, enrollment_year INTEGER);",
        "rows": _escola_rows(),
        "examples": [
            ("Qual é a nota média de cada curso?", "SELECT course, AVG(score) FROM students GROUP BY course"),
            ("Quais alunos têm nota maior que 9?", "SELECT name FROM students WHERE score > 9"),
            ("Quantos alunos entraram em 2024?", "SELECT COUNT(*) FROM students WHERE enrollment_year = 2024"),
            ("Quem tem a maior nota?", "SELECT name FROM students ORDER BY score DESC LIMIT 1"),
        ],
    },
    "RH (employees)": {
        "schema": "CREATE TABLE employees (id INTEGER, name VARCHAR, role VARCHAR, salary FLOAT, city VARCHAR);",
        "rows": _rh_rows(),
        "examples": [
            ("Qual o salário médio por cargo?", "SELECT role, AVG(salary) FROM employees GROUP BY role"),
            ("Quem trabalha em Curitiba?", "SELECT name FROM employees WHERE city = 'Curitiba'"),
            ("Qual o maior salário?", "SELECT MAX(salary) FROM employees"),
            ("Quantos desenvolvedores existem?", "SELECT COUNT(*) FROM employees WHERE role = 'Desenvolvedor'"),
        ],
    },
    "Biblioteca (livros, em português)": {
        "schema": "CREATE TABLE livros (id INTEGER, titulo VARCHAR, autor VARCHAR, ano INTEGER, paginas INTEGER);",
        "rows": _biblioteca_rows(),
        "examples": [
            ("Quais livros foram publicados antes de 1900?", "SELECT titulo FROM livros WHERE ano < 1900"),
            ("Quantos livros cada autor tem?", "SELECT autor, COUNT(*) FROM livros GROUP BY autor"),
            ("Qual é o livro com mais páginas?", "SELECT titulo FROM livros ORDER BY paginas DESC LIMIT 1"),
            ("Liste os livros de Machado de Assis.", "SELECT titulo FROM livros WHERE autor = 'Machado de Assis'"),
        ],
    },
}


def db_choices() -> List[str]:
    return list(SAMPLE_DBS) + [CUSTOM_KEY]


def get_schema(key: str) -> str:
    return SAMPLE_DBS[key]["schema"] if key in SAMPLE_DBS else ""


def get_examples(key: str) -> List[str]:
    return [q for q, _ in SAMPLE_DBS[key]["examples"]] if key in SAMPLE_DBS else []


def get_reference_examples(key: str) -> List[Tuple[str, str]]:
    return SAMPLE_DBS[key]["examples"]


def make_connection(key: str) -> sqlite3.Connection:
    """Cria um banco novo em memoria (isolado a cada consulta) com os dados de exemplo."""
    db = SAMPLE_DBS[key]
    conn = sqlite3.connect(":memory:")
    conn.executescript(db["schema"])
    for table, rows in db["rows"].items():
        conn.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)
    conn.commit()
    return conn
