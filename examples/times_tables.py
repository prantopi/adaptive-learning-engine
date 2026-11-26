"""A student learning the 6-9 times tables for three weeks, driven by AdaptiveLearner.

Nothing here is about language: items are facts like "7 x 8", answers are right or wrong,
and the engine decides what to practise each day within a 5-minute session.

Run:  uv run python examples/times_tables.py
"""
import random
from datetime import date, timedelta

from adaptive_learning_engine import AdaptiveLearner, srs

rng = random.Random(1)
facts = [f"{a} x {b}" for a in range(6, 10) for b in range(2, 10)]
learner = AdaptiveLearner(daily_new=6)
learner.add_items(facts)

# The simulated student: a hidden memory strength per fact, harder for bigger numbers.
memory = {f: 1.0 for f in facts}
last_seen: dict[str, date] = {}

start = date(2026, 3, 2)
for d in range(21):
    today = start + timedelta(days=d)
    if d % 7 == 6:
        continue                                   # a day off each week
    plan = learner.recommend(available_time=5 * 60, when=today)
    session = plan.reviews + plan.new_items
    right = 0
    for action in session:
        fact = action.item_id
        a, b = map(int, fact.split(" x "))
        gap = (today - last_seen[fact]).days if fact in last_seen else 0
        recall = 0.9 ** (gap / memory[fact]) if fact in last_seen else 0.6
        correct = rng.random() < recall
        response_time = rng.uniform(1.5, 3.0) if correct and recall > 0.7 else rng.uniform(4, 9)
        learner.record_answer(fact, correct, response_time=response_time, choices=None, when=today)
        memory[fact] = memory[fact] * (2.2 - 0.1 * (a + b) / 4) if correct else max(0.8, memory[fact] * 0.5)
        last_seen[fact] = today
        right += correct
    print(f"{today:%a %d %b}: {len(plan.reviews):2} reviews + {len(plan.new_items)} new "
          f"({right}/{len(session)} right), {len(learner.due(today + timedelta(days=1)))} due tomorrow")

k = learner.memory
print(f"\nStarted {sum(1 for i in learner.items.values() if i.p_learned is not None)} of {len(facts)} facts.")
print(f"Memory model after {k.observations} spaced reviews: k = {k.factor:.2f} ({k.low:.2f}-{k.high:.2f}), "
      f"growth g = {k.growth:.2f}")
print(f"Usual time for a right answer: {learner.typical_response_time():.1f}s")
end = start + timedelta(days=21)


def recall_now(item):  # predicted chance of remembering it today, from this student's memory model
    return srs.retention(k.scale(item.successes) * item.stability, end.toordinal() - item.last_review)


weakest = sorted((i for i in learner.items.values() if i.last_review is not None), key=recall_now)[:3]
print("Most likely forgotten by now:", ", ".join(f"{i.item_id} ({recall_now(i):.0%} recall)" for i in weakest))
