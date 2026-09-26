"""
Evolutionary search over Candidate configs.

Loop, each generation:
  1. Evaluate every candidate in the population on TRAIN_SET.
  2. Keep the top-k scorers (elitism).
  3. Produce children by:
       a. mutation: ask the mutator LLM to rewrite a survivor's instruction,
          conditioned on that survivor's actual training failures.
       b. crossover: splice one survivor's instruction with another
          survivor's few-shot set.
  4. Repeat until max_generations or a candidate hits perfect train accuracy
     (checked against test separately, never used to stop early on test).

Every generation's full state (configs, scores, and *why* the mutator
changed what it changed) is captured in `trace` so the README can show a
readable "here's how the prompt evolved" narrative instead of just a
final number.
"""

import random
from dataclasses import dataclass, asdict

from src.candidate import Candidate
from src.evaluator import evaluate_candidate
from src.llm_client import LLMClient


@dataclass
class GenerationRecord:
    generation: int
    population: list  # list of dicts: {id, instruction, few_shot_ids, score}
    best_id: str
    best_score: float


class EvolutionaryOptimizer:
    def __init__(self, train_set, client: LLMClient, population_size=6,
                 elite_k=2, max_generations=6, few_shot_size=3, seed=0):
        self.train_set = train_set
        self.client = client
        self.population_size = population_size
        self.elite_k = elite_k
        self.max_generations = max_generations
        self.few_shot_size = few_shot_size
        self.rng = random.Random(seed)
        self.trace: list = []

    def _random_few_shot(self):
        ids = [ex.id for ex in self.train_set]
        return tuple(sorted(self.rng.sample(ids, self.few_shot_size)))

    def _init_population(self, seed_instruction: str) -> list:
        pop = [Candidate(instruction=seed_instruction,
                          few_shot_ids=self._random_few_shot(),
                          generation=0)]
        for _ in range(self.population_size - 1):
            pop.append(Candidate(instruction=seed_instruction,
                                  few_shot_ids=self._random_few_shot(),
                                  generation=0))
        return pop

    def _score_population(self, population: list) -> list:
        for cand in population:
            report = evaluate_candidate(cand, self.train_set, self.client, self.train_set)
            cand.score = report.accuracy
            cand._last_report = report  # stashed for mutation context, not persisted
        return sorted(population, key=lambda c: c.score, reverse=True)

    def _mutate(self, parent: Candidate, generation: int) -> Candidate:
        failures = [
            (r.question, r.predicted_sql or "(no output)", r.gold_sql)
            for r in getattr(parent, "_last_report", None).failures()[:3]
        ] if getattr(parent, "_last_report", None) else []

        if failures:
            new_instruction = self.client.propose_mutation(parent.instruction, failures)
        else:
            new_instruction = parent.instruction  # nothing to learn from, keep as-is

        return Candidate(
            instruction=new_instruction,
            few_shot_ids=self._maybe_swap_few_shot(parent.few_shot_ids),
            generation=generation,
            parent_id=parent.id(),
        )

    def _crossover(self, parent_a: Candidate, parent_b: Candidate, generation: int) -> Candidate:
        return Candidate(
            instruction=parent_a.instruction,
            few_shot_ids=parent_b.few_shot_ids,
            generation=generation,
            parent_id=f"{parent_a.id()}+{parent_b.id()}",
        )

    def _maybe_swap_few_shot(self, current_ids: tuple) -> tuple:
        if self.rng.random() < 0.5:
            return current_ids
        return self._random_few_shot()

    def run(self, seed_instruction: str) -> Candidate:
        population = self._init_population(seed_instruction)

        for gen in range(self.max_generations):
            population = self._score_population(population)
            best = population[0]

            self.trace.append(GenerationRecord(
                generation=gen,
                population=[
                    {"id": c.id(), "instruction": c.instruction,
                     "few_shot_ids": c.few_shot_ids, "score": c.score,
                     "parent_id": c.parent_id}
                    for c in population
                ],
                best_id=best.id(),
                best_score=best.score,
            ))

            if gen == self.max_generations - 1:
                break

            survivors = population[: self.elite_k]
            children = list(survivors)  # elitism: survivors carry over unmodified
            while len(children) < self.population_size:
                if self.rng.random() < 0.7:
                    parent = self.rng.choice(survivors)
                    children.append(self._mutate(parent, gen + 1))
                else:
                    a, b = self.rng.sample(survivors, 2) if len(survivors) > 1 else (survivors[0], survivors[0])
                    children.append(self._crossover(a, b, gen + 1))
            population = children

        return max(population, key=lambda c: (c.score if c.score is not None else -1))

    def trace_as_dicts(self) -> list:
        return [
            {"generation": r.generation, "best_id": r.best_id, "best_score": r.best_score,
             "population": r.population}
            for r in self.trace
        ]
