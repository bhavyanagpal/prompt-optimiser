import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db


def test_schema_loads_and_seed_data_present():
    with db.fresh_db() as conn:
        rows = db.run_query(conn, "SELECT COUNT(*) FROM customers;")
        assert rows[0][0] == 8

        rows = db.run_query(conn, "SELECT COUNT(*) FROM products;")
        assert rows[0][0] == 9


def test_bad_sql_raises_execution_error():
    with db.fresh_db() as conn:
        try:
            db.run_query(conn, "SELECT * FROM not_a_real_table;")
            assert False, "expected SQLExecutionError"
        except db.SQLExecutionError:
            pass


def test_results_match_ignores_order():
    a = [(1, "x"), (2, "y")]
    b = [(2, "y"), (1, "x")]
    assert db.results_match(a, b)


def test_results_match_ignores_float_vs_int():
    a = [(25,)]
    b = [(25.0,)]
    assert db.results_match(a, b)


def test_results_match_detects_real_difference():
    a = [(1, "x")]
    b = [(1, "z")]
    assert not db.results_match(a, b)


def test_known_query_against_seed_data():
    with db.fresh_db() as conn:
        rows = db.run_query(conn, "SELECT name FROM customers WHERE country = 'India';")
        names = sorted(r[0] for r in rows)
        assert names == ["Priya Nair", "Rahul Mehta"]


if __name__ == "__main__":
    test_schema_loads_and_seed_data_present()
    test_bad_sql_raises_execution_error()
    test_results_match_ignores_order()
    test_results_match_ignores_float_vs_int()
    test_results_match_detects_real_difference()
    test_known_query_against_seed_data()
    print("All tests passed.")
