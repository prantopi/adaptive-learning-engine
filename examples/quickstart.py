"""A tiny tour of the engine on non-language items (world capitals).

Run:  uv run python examples/quickstart.py
"""
from datetime import date, timedelta

from learning_engine import bkt, budget, personalize, srs

params = bkt.params_for("Beginner")
today = date(2026, 1, 1)

# 1. Knowledge tracing: how sure are we the learner knows "Capital of Peru -> Lima"?
p = params.p_init
for correct in (False, True, True, True):
    p = bkt.update(p, correct, params, p_guess=0.02)  # typed answers: guessing is unlikely
    print(f"after {'right' if correct else 'wrong'} answer: P(known) = {p:.2f}")

# 2. Spaced repetition: when should it come back?
stability = srs.INITIAL_STABILITY
for review in range(4):
    stability = srs.grow(stability, p, days_since_last=max(1, round(stability)))
    print(f"review {review + 1}: stability {stability:.1f} days, next review {srs.next_review(today, stability)}")

# 3. Personal memory: this learner keeps forgetting sooner than the default model expects.
history = [personalize.Observation(elapsed_days=d, stability=4.0, correct=d < 3, guess_rate=0.02)
           for d in (1, 2, 3, 4, 5, 6) for _ in range(4)]
fit = personalize.fit(history)
print(f"memory factor k = {fit.factor:.2f} (80% range {fit.low:.2f}-{fit.high:.2f}): "
      f"an item with 4 days' stability comes back after {srs.interval_days(fit.scale(0) * 4)} day(s), not 4")

# 4. Ten minutes to spare: which due reviews are worth most?
due = [budget.Candidate(item_id, stability=s, elapsed_days=e, p_learned=pl, seconds=8)
       for item_id, (s, e, pl) in enumerate([(2, 5, 0.4), (30, 2, 0.95), (6, 7, 0.7), (1, 3, 0.3)])]
print("review first:", [c.item_id for c in budget.choose(due, budget_seconds=20)])
