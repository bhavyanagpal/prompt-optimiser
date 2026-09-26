# Self-Improving Prompt Optimizer

A small, from-scratch implementation of the idea behind DSPy / APE / OPRO:
**treat a prompt as a set of parameters, and search for good values using a
real eval set — instead of hand-tuning by feel.**

This repo optimizes a text-to-SQL agent. The task is picked deliberately:
SQL correctness is *checkable* (execute the query, compare results), which
means the optimizer's score is trustworthy, not vibes-based.

## Why this project exists

Most "prompt engineering" is manual: write a prompt, eyeball a few outputs,
tweak, repeat. That doesn't scale, isn't reproducible, and overfits to
whatever examples the engineer happened to look at.

This project treats prompt design as an actual ML problem:

- **Parameters**: the instruction text + which few-shot examples to include.
- **Objective**: accuracy on a labeled training set (checked by *executing*
  the generated SQL and comparing real result sets, not string-matching).
- **Search**: an evolutionary loop — score a population, keep the best,
  generate children via LLM-driven mutation (rewrite the instruction based
  on what it got wrong) and crossover (mix instruction + few-shot set from
  two survivors).
- **Held-out evaluation**: a train/test split, where test is touched
  exactly once, after search is finished — the same discipline you'd use
  for any ML model, applied to a prompt.

## Architecture

```
src/
  db.py          Synthetic e-commerce SQLite DB (schema + seed data) used as ground truth
  dataset.py     Train/test examples: (question, gold SQL) pairs
  candidate.py   A "candidate" = instruction text + few-shot example ids
  llm_client.py  Anthropic API wrapper — student model (generates SQL) + mutator model (rewrites instructions)
  evaluator.py   Executes predicted vs gold SQL, compares result sets, returns accuracy + failures
  optimizer.py   Evolutionary search loop: score -> select -> mutate/crossover -> repeat
  baseline.py    Hand-written baseline instruction (the thing the optimizer has to beat)
run_experiment.py  Entry point: baseline vs optimizer, saves results/trace.json + results/report.md
tests/           Unit tests that don't require an API key (DB layer + dataset sanity checks)
```

### Why execute SQL instead of string-matching it?

`SELECT name FROM customers WHERE country='India'` and
`SELECT c.name FROM customers c WHERE c.country = 'India'` are the same
query with different surface forms. String-matching would mark the second
one wrong. Instead, both are run against a real seeded SQLite DB and their
**result sets** are compared (order-insensitive, numeric-type-tolerant).
This is what makes the accuracy number something you can actually
optimize against without gaming it.

### Why an evolutionary loop instead of just... asking the model to write a better prompt once?

A single "please improve this prompt" call has no signal about *which*
direction is actually better — it's just another guess. The evolutionary
loop keeps what's measurably working (elitism), explores variations
(mutation conditioned on real failures + crossover of few-shot sets), and
the trace lets you see *why* each change happened, not just the final
number.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python run_experiment.py --generations 6 --population 6
```

This will:
1. Score the hand-written baseline prompt on the held-out test set.
2. Run the evolutionary optimizer against the training set only.
3. Score the optimizer's best candidate on the same held-out test set.
4. Write `results/trace.json` (full generation-by-generation history) and
   `results/report.md` (human-readable comparison + evolved instruction).

Tune `--population`, `--generations`, `--elite-k`, and `--few-shot-size` to
trade off search quality against API cost — a bigger population/more
generations explores more of the space but costs proportionally more calls.

### Running the tests (no API key needed)

```bash
python tests/test_db.py
python tests/test_dataset_sanity.py
```

These check the SQL-execution/comparison logic and verify every gold SQL
example in the dataset is actually correct — the parts of the system you
can (and should) verify without spending API calls.

## Design notes / honest limitations

- **Train/test split is small.** 10 train / 6 test examples is enough to
  demonstrate the method, not enough to make strong statistical claims.
  Scaling the dataset is the natural next step.
- **The mutator can overfit to train failures**, exactly like an ML model
  overfits to its training set — this is why test accuracy is reported
  separately and only once. If optimized-train accuracy is much higher
  than optimized-test accuracy, that's the overfitting signature, and it's
  worth showing rather than hiding.
- **Student/mutator use different models on purpose** (`STUDENT_MODEL` /
  `MUTATOR_MODEL` env vars) — a cheaper model does the actual SQL
  generation being evaluated, while a stronger model does the harder job
  of diagnosing failures and rewriting instructions. This mirrors a real
  production pattern where the "worker" model and the "prompt engineer"
  model don't need to be the same size.
- **This is not a DSPy reimplementation.** DSPy's actual optimizers
  (MIPRO, COPRO, etc.) are considerably more sophisticated. The point of
  building this from scratch is understanding *why* those systems work —
  treating prompts as searchable parameters — not competing with them.

## Possible extensions

- Swap the evolutionary search for a bandit-based one (UCB/Thompson
  sampling) and compare sample-efficiency at the same API budget.
- Grow the dataset and check whether the optimizer's gains hold up, or
  were an artifact of a small test set.
- Add a second task (e.g. a classification task) to check whether the
  optimizer's gains generalize across task types, not just this one.
