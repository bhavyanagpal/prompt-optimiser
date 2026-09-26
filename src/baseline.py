"""
The hand-written baseline prompt a reasonably careful engineer would write
without any automated tuning. This is the thing the optimizer needs to beat
-- if it can't, that's a real (and reportable) negative result, not a bug
to hide.
"""

BASELINE_INSTRUCTION = (
    "You are a SQL expert. Given a database schema and a question, "
    "write a single SQLite query that answers the question. "
    "Only use the tables and columns given in the schema."
)

# A fixed, reasonable-looking few-shot set (first 3 train examples),
# the way most people would just grab the first few examples they wrote.
BASELINE_FEW_SHOT_IDS = (1, 2, 3)
