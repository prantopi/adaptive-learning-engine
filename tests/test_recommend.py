from __future__ import annotations

from learning_engine import dropout
from learning_engine.personalize import Observation
from learning_engine.recommend import Item, Learner, recommend

TODAY = 1000


def studied(item_id, p=0.6, stability=2.0, days_ago=3, **kw):
    return Item(item_id, p_learned=p, stability=stability, last_review=TODAY - days_ago, **kw)


def steady_history():
    return [dropout.DayStats(TODAY - d, 20, 17) for d in range(1, 8)]


def test_nothing_to_do():
    plan = recommend([], Learner(), TODAY)
    assert plan.next is None and plan.reviews == [] and plan.new_items == []


def test_reviews_come_before_new_items():
    items = [studied(1), Item(2), Item(3)]
    plan = recommend(items, Learner(), TODAY)
    assert plan.next.kind == "review" and plan.next.item_id == 1
    assert [a.item_id for a in plan.new_items] == [2, 3]


def test_items_not_yet_due_are_left_alone():
    plan = recommend([studied(1, stability=10, days_ago=2)], Learner(), TODAY)
    assert plan.due == 0 and plan.next is None


def test_most_valuable_review_first():
    items = [studied(1, p=0.95, stability=1, days_ago=1), studied(2, p=0.4, stability=2, days_ago=6)]
    plan = recommend(items, Learner(), TODAY)
    assert plan.reviews[0].item_id == 2
    assert "overdue" in plan.reviews[0].reason and plan.reviews[0].predicted_recall < 0.9


def test_time_budget_limits_the_session():
    items = [studied(i, days_ago=2 + i % 4) for i in range(40)]
    full = recommend(items, Learner(), TODAY)
    short = recommend(items, Learner(), TODAY, minutes=2)
    assert full.due == 40 and len(full.reviews) == 40
    assert 0 < len(short.reviews) < 40 and short.estimated_minutes <= 2
    assert short.new_items == []   # no time left for new items


def test_forgetful_learner_gets_reviews_sooner():
    item = [studied(1, stability=4, days_ago=2)]
    assert recommend(item, Learner(), TODAY).due == 0
    forgetful = [Observation(d, 4.0, False, 0.02) for d in (1, 2, 3) for _ in range(10)]
    assert recommend(item, Learner(reviews=forgetful), TODAY).due == 1


def test_light_day_after_a_long_break():
    lapsed = [dropout.DayStats(TODAY - d, 20, 15) for d in (14, 15, 16)]
    items = [studied(i, days_ago=14) for i in range(40)] + [Item(100 + i) for i in range(10)]
    plan = recommend(items, Learner(history=lapsed), TODAY)
    assert plan.light_day and plan.quit_risk > 0.55
    assert plan.new_quota == 5 and plan.minutes == 5
    assert len(plan.reviews) == 30   # 40 due, but only 30 fit in the suggested 5 minutes


def test_steady_learner_gets_a_normal_day():
    plan = recommend([Item(i) for i in range(12)], Learner(history=steady_history()), TODAY)
    assert not plan.light_day and plan.new_quota == 10 and len(plan.new_items) == 10


def test_allowance_counts_what_was_learned_today():
    plan = recommend([Item(i) for i in range(12)], Learner(learned_today=7), TODAY)
    assert len(plan.new_items) == 3


def test_new_items_get_a_head_start_from_related_known_items():
    known = studied(1, p=0.95, days_ago=1, stability=5)
    plain, linked = Item(2), Item(3, related=((1, 0.6),))
    plan = recommend([known, plain, linked], Learner(), TODAY)
    starts = {a.item_id: a for a in plan.new_items}
    assert starts[3].starting_p > starts[2].starting_p
    assert "builds on 1 item" in starts[3].reason


def test_hard_items_start_lower_and_are_flagged():
    plan = recommend([Item(1, difficulty=1.5), Item(2)], Learner(), TODAY)
    starts = {a.item_id: a.starting_p for a in plan.new_items}
    assert starts[1] < starts[2]
    review = recommend([studied(5, difficulty=1.5)], Learner(), TODAY).reviews[0]
    assert "hard item" in review.reason
