"""
A Candidate is a point in the search space: an instruction string plus a
choice of which training examples to use as few-shot demonstrations.

Keeping this as a small structured object (rather than a raw prompt blob)
is what makes mutation/crossover well-defined instead of ad hoc string
surgery.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Candidate:
    instruction: str
    few_shot_ids: tuple  # ids into dataset.TRAIN_SET
    generation: int = 0
    parent_id: Optional[str] = None
    score: Optional[float] = None  # filled in after evaluation

    def id(self) -> str:
        return f"gen{self.generation}-{hash((self.instruction, self.few_shot_ids)) & 0xffff:x}"

    def few_shot_examples(self, train_set):
        by_id = {ex.id: ex for ex in train_set}
        return [by_id[i] for i in self.few_shot_ids if i in by_id]
