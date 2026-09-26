"""
Synthetic e-commerce database used as the ground truth for the text-to-SQL task.

Design goals:
- Small enough to reason about in a README screenshot.
- Rich enough to need real joins/aggregations (not just SELECT * WHERE).
- Deterministic seed data so eval scores are reproducible run-to-run.
"""

import sqlite3
from contextlib import contextmanager

SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    price      REAL NOT NULL
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date  TEXT NOT NULL,
    status      TEXT NOT NULL   -- 'completed' | 'cancelled' | 'refunded'
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL
);
"""

CUSTOMERS = [
    (1, "Amara Chen", "USA", "2023-01-15"),
    (2, "Rahul Mehta", "India", "2023-02-20"),
    (3, "Sofia Rossi", "Italy", "2023-03-05"),
    (4, "Liam O'Connor", "Ireland", "2023-04-12"),
    (5, "Yuki Tanaka", "Japan", "2023-05-30"),
    (6, "Priya Nair", "India", "2023-06-18"),
    (7, "Carlos Diaz", "Mexico", "2023-07-22"),
    (8, "Emma Wilson", "USA", "2023-08-09"),
]

PRODUCTS = [
    (1, "Wireless Mouse", "Electronics", 25.99),
    (2, "Mechanical Keyboard", "Electronics", 89.99),
    (3, "Standing Desk", "Furniture", 349.00),
    (4, "Office Chair", "Furniture", 219.50),
    (5, "Notebook Set", "Stationery", 12.00),
    (6, "Fountain Pen", "Stationery", 45.00),
    (7, "USB-C Hub", "Electronics", 39.99),
    (8, "Desk Lamp", "Furniture", 32.00),
    (9, "Desk Organizer", "Furniture", 18.50),  # deliberately never ordered
]

ORDERS = [
    (1, 1, "2023-09-01", "completed"),
    (2, 1, "2023-10-14", "completed"),
    (3, 2, "2023-09-20", "completed"),
    (4, 3, "2023-09-25", "cancelled"),
    (5, 4, "2023-10-02", "completed"),
    (6, 5, "2023-10-05", "completed"),
    (7, 6, "2023-10-08", "refunded"),
    (8, 2, "2023-10-19", "completed"),
    (9, 7, "2023-10-22", "completed"),
    (10, 8, "2023-10-30", "completed"),
    (11, 1, "2023-11-02", "completed"),
    (12, 3, "2023-11-05", "completed"),
]

ORDER_ITEMS = [
    (1, 1, 1, 2),
    (2, 1, 2, 1),
    (3, 2, 3, 1),
    (4, 3, 5, 5),
    (5, 4, 4, 1),
    (6, 5, 6, 3),
    (7, 6, 7, 1),
    (8, 7, 8, 2),
    (9, 8, 2, 1),
    (10, 9, 1, 1),
    (11, 9, 7, 1),
    (12, 10, 3, 1),
    (13, 10, 4, 1),
    (14, 11, 5, 10),
    (15, 12, 6, 1),
]


def build_connection() -> sqlite3.Connection:
    """Create a fresh in-memory DB with schema + seed data loaded."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", ORDERS)
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", ORDER_ITEMS)
    conn.commit()
    return conn


@contextmanager
def fresh_db():
    conn = build_connection()
    try:
        yield conn
    finally:
        conn.close()


class SQLExecutionError(Exception):
    """Raised when a candidate-generated query fails to execute at all."""


def run_query(conn: sqlite3.Connection, sql: str):
    """Execute SQL and return rows as a list of tuples, or raise SQLExecutionError."""
    try:
        cur = conn.execute(sql)
        return cur.fetchall()
    except sqlite3.Error as e:
        raise SQLExecutionError(str(e)) from e


def results_match(predicted_rows, gold_rows) -> bool:
    """
    Order-insensitive, type-tolerant comparison of two result sets.
    Normalizes numeric types (int vs float) and sorts rows so a
    semantically-correct query isn't marked wrong just because it
    returned rows in a different order, or 25 vs 25.0.
    """
    def normalize(rows):
        norm = []
        for row in rows:
            norm_row = tuple(
                round(v, 4) if isinstance(v, float) else v for v in row
            )
            norm.append(norm_row)
        return sorted(norm)

    return normalize(predicted_rows) == normalize(gold_rows)
