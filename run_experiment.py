"""
Runs the full experiment:
  1. Score the hand-written baseline on TEST_SET (held out, touched once).
  2. Run the evolutionary optimizer on TRAIN_SET only.
  3. Score the optimizer's best candidate on TEST_SET (same held-out set,
     touched once, after the search is already finished).
  4. Save results/trace.json (generation-by-generation history) and
     results/report.md (human-readable comparison).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_experiment.py [--generations 6] [--population 6]
"""

import argparse
import json
import os
from dataclasses import asdict

from src.baseline import BASELINE_INSTRUCTION, BASELINE_FEW_SHOT_IDS
from src.candidate import Candidate
from src.dataset import TRAIN_SET, TEST_SET
from src.evaluator import evaluate_candidate
from src.llm_client import LLMClient
from src.optimizer import EvolutionaryOptimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elite-k", type=int, default=2)
    parser.add_argument("--few-shot-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = LLMClient()

    print("== Baseline ==")
    baseline_candidate = Candidate(instruction=BASELINE_INSTRUCTION,
                                    few_shot_ids=BASELINE_FEW_SHOT_IDS,
                                    generation=-1)
    baseline_report = evaluate_candidate(baseline_candidate, TEST_SET, client, TRAIN_SET)
    print(f"Baseline test accuracy: {baseline_report.accuracy:.2%}")

    print("\n== Running evolutionary optimizer on TRAIN_SET ==")
    optimizer = EvolutionaryOptimizer(
        train_set=TRAIN_SET,
        client=client,
        population_size=args.population,
        elite_k=args.elite_k,
        max_generations=args.generations,
        few_shot_size=args.few_shot_size,
        seed=args.seed,
    )
    best_candidate = optimizer.run(seed_instruction=BASELINE_INSTRUCTION)
    for record in optimizer.trace:
        print(f"  gen {record.generation}: best train accuracy = {record.best_score:.2%}")

    print("\n== Evaluating optimizer's best candidate on held-out TEST_SET ==")
    optimized_report = evaluate_candidate(best_candidate, TEST_SET, client, TRAIN_SET)
    print(f"Optimized test accuracy: {optimized_report.accuracy:.2%}")

    trace_path = os.path.join(RESULTS_DIR, "trace.json")
    with open(trace_path, "w") as f:
        json.dump({
            "baseline": {
                "instruction": BASELINE_INSTRUCTION,
                "few_shot_ids": list(BASELINE_FEW_SHOT_IDS),
                "test_accuracy": baseline_report.accuracy,
            },
            "optimized": {
                "instruction": best_candidate.instruction,
                "few_shot_ids": list(best_candidate.few_shot_ids),
                "train_accuracy": best_candidate.score,
                "test_accuracy": optimized_report.accuracy,
            },
            "generations": optimizer.trace_as_dicts(),
        }, f, indent=2)

    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(_render_report(baseline_report, best_candidate, optimized_report, optimizer))

    print(f"\nSaved trace to {trace_path}")
    print(f"Saved report to {report_path}")


def _render_report(baseline_report, best_candidate, optimized_report, optimizer) -> str:
    lines = [
        "# Experiment Report\n",
        f"**Baseline test accuracy:** {baseline_report.accuracy:.2%}\n",
        f"**Optimized test accuracy:** {optimized_report.accuracy:.2%}\n",
        f"**Delta:** {optimized_report.accuracy - baseline_report.accuracy:+.2%}\n",
        "\n## Best evolved instruction\n",
        f"```\n{best_candidate.instruction}\n```\n",
        "\n## Generation-by-generation best train accuracy\n",
    ]
    for record in optimizer.trace:
        lines.append(f"- Generation {record.generation}: {record.best_score:.2%}")
    lines.append("\n## Held-out test failures (optimized candidate)\n")
    for r in optimized_report.failures():
        lines.append(f"- **Q:** {r.question}\n  - predicted: `{r.predicted_sql}`\n  - gold: `{r.gold_sql}`")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
