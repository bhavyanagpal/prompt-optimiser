"""
Thin wrapper around the Anthropic API.

Two distinct roles use this client:
  1. The STUDENT model: given a candidate's instruction + few-shot examples,
     generate SQL for a new question. This is what's being evaluated.
  2. The MUTATOR model: given a candidate's current config + which train
     examples it got wrong, propose a revised config. This is the
     "optimizer" half of the system.

Keeping both behind one client makes it easy to point the student at a
cheaper/faster model than the mutator, which is a real production pattern
(the model doing the work doesn't need to be the model designing the work).
"""

import os
import re
from dataclasses import dataclass

import anthropic

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "claude-3-5-haiku-latest")
MUTATOR_MODEL = os.environ.get("MUTATOR_MODEL", "claude-sonnet-4-6")

SCHEMA_DESCRIPTION = """
Tables:
  customers(customer_id, name, country, signup_date)
  products(product_id, name, category, price)
  orders(order_id, customer_id, order_date, status)   -- status: 'completed' | 'cancelled' | 'refunded'
  order_items(order_item_id, order_id, product_id, quantity)
"""


@dataclass
class LLMResponse:
    text: str
    raw: dict


class LLMClient:
    def __init__(self):
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def _call(self, model: str, system: str, user: str, max_tokens: int = 512) -> str:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def generate_sql(self, instruction: str, few_shot: list, question: str) -> str:
        """Ask the student model to produce SQL for `question`, given the
        candidate's current instruction text and few-shot examples."""
        few_shot_block = "\n\n".join(
            f"Q: {ex.question}\nSQL: {ex.gold_sql}" for ex in few_shot
        )
        system = (
            f"{instruction}\n\n{SCHEMA_DESCRIPTION}\n\n"
            "Respond with ONLY the SQL query, no explanation, no markdown fences."
        )
        user = f"{few_shot_block}\n\nQ: {question}\nSQL:"
        raw = self._call(STUDENT_MODEL, system, user, max_tokens=256)
        return self._extract_sql(raw)

    def propose_mutation(self, instruction: str, failures: list) -> str:
        """Ask the mutator model to rewrite the instruction, given a sample
        of (question, predicted_sql, gold_sql) triples the current
        instruction got wrong. Returns a new instruction string."""
        failure_block = "\n\n".join(
            f"Question: {q}\nModel produced: {pred}\nCorrect answer: {gold}"
            for q, pred, gold in failures
        )
        system = (
            "You improve instructions given to a smaller text-to-SQL model. "
            "You will see the current instruction and examples of questions "
            "it got wrong. Rewrite the instruction to fix these failure "
            "patterns, without making it dramatically longer. "
            "Respond with ONLY the new instruction text, nothing else."
        )
        user = (
            f"Current instruction:\n{instruction}\n\n"
            f"Schema:\n{SCHEMA_DESCRIPTION}\n\n"
            f"Failures on training examples:\n{failure_block}\n\n"
            "New instruction:"
        )
        return self._call(MUTATOR_MODEL, system, user, max_tokens=400).strip()

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Strip markdown fences if the model added them anyway, and trim
        whitespace/trailing semicolons inconsistency."""
        text = text.strip()
        fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return text
