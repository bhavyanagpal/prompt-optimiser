"""
Runs a Candidate's (instruction, few-shot set) against a list of Examples
by actually executing both the predicted and gold SQL against the seeded
SQLite DB and comparing result sets. This is stricter than string-matching
SQL (which breaks on trivially equivalent queries written differently) and
is what makes the eval trustworthy enough to optimize against.
"""

from dataclasses import dataclass
from typing import Optional

from src import db
from src.candidate import Candidate
from src.llm_client import LLMClient


@dataclass
class ExampleResult:
    example_id: int
    question: str
    predicted_sql: str
    gold_sql: str
    correct: bool
    error: Optional[str] = None


@dataclass
class EvalReport:
    accuracy: float
    results: list  # list[ExampleResult]

    def failures(self):
        return [r for r in self.results if not r.correct]


def evaluate_candidate(candidate: Candidate, examples: list, client: LLMClient,
                        train_set: list) -> EvalReport:
    few_shot = candidate.few_shot_examples(train_set)
    results = []

    for ex in examples:
        # Never let a candidate's few-shot set leak the answer to itself.
        active_few_shot = [f for f in few_shot if f.id != ex.id]

        try:
            predicted_sql = client.generate_sql(candidate.instruction, active_few_shot, ex.question)
        except Exception as e:  # LLM/API failure counts as wrong, not a crash
            results.append(ExampleResult(ex.id, ex.question, "", ex.gold_sql, False, error=str(e)))
            continue

        with db.fresh_db() as conn:
            try:
                predicted_rows = db.run_query(conn, predicted_sql)
                gold_rows = db.run_query(conn, ex.gold_sql)
                correct = db.results_match(predicted_rows, gold_rows)
                results.append(ExampleResult(ex.id, ex.question, predicted_sql, ex.gold_sql, correct))
            except db.SQLExecutionError as e:
                results.append(ExampleResult(ex.id, ex.question, predicted_sql, ex.gold_sql, False, error=str(e)))

    accuracy = sum(r.correct for r in results) / len(results) if results else 0.0
    return EvalReport(accuracy=accuracy, results=results)
