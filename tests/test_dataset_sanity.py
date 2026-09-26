import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db
from src.dataset import TRAIN_SET, TEST_SET


def test_all_gold_sql_executes():
    with db.fresh_db() as conn:
        for ex in TRAIN_SET + TEST_SET:
            rows = db.run_query(conn, ex.gold_sql)
            assert rows is not None, f"Example {ex.id} gold SQL failed silently"


def test_no_duplicate_ids():
    all_ids = [ex.id for ex in TRAIN_SET + TEST_SET]
    assert len(all_ids) == len(set(all_ids)), "duplicate example ids found"


def test_all_gold_queries_return_nonempty_where_expected():
    # A gold query returning zero rows usually means a typo (wrong filter
    # value, wrong join), not a real "no results" case in this dataset.
    with db.fresh_db() as conn:
        for ex in TRAIN_SET + TEST_SET:
            rows = db.run_query(conn, ex.gold_sql)
            assert len(rows) > 0, f"Example {ex.id} ('{ex.question}') gold SQL returned 0 rows"


if __name__ == "__main__":
    test_all_gold_sql_executes()
    test_no_duplicate_ids()
    test_all_gold_queries_return_nonempty_where_expected()
    print("All dataset sanity checks passed.")
