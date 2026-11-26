from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from adaptive_learning_engine import AdaptiveLearner

D0 = date(2026, 3, 2)


def day(n: int) -> date:
    return D0 + timedelta(days=n)


def make(n_items: int = 5, **kw) -> AdaptiveLearner:
    learner = AdaptiveLearner(**kw)
    learner.add_items([f"q{i}" for i in range(n_items)])
    return learner


class TestBasics:
    def test_readme_example(self):
        learner = AdaptiveLearner()
        learner.add_items(["spanish_001", "spanish_002", "spanish_003"])
        result = learner.record_answer("spanish_001", correct=True, response_time=2.4)
        assert result.first_time and result.p_learned > 0.15
        plan = learner.recommend(available_time=900)
        assert plan.next.kind == "learn" and plan.next.item_id == "spanish_002"

    def test_new_items_follow_the_order_added(self):
        plan = make(5).recommend(when=D0)
        assert [a.item_id for a in plan.new_items] == ["q0", "q1", "q2", "q3", "q4"]

    def test_daily_allowance_counts_items_started_today(self):
        learner = make(15, daily_new=4)
        for i in range(3):
            learner.record_answer(f"q{i}", True, when=D0)
        assert len(learner.recommend(when=D0).new_items) == 1
        assert len(learner.recommend(when=day(1)).new_items) == 4

    def test_wrong_answer_comes_back_tomorrow_and_shows_as_due(self):
        learner = make(3)
        r = learner.record_answer("q0", False, when=D0)
        assert r.next_review == day(1)
        assert learner.due(day(1)) == ["q0"]
        plan = learner.recommend(when=day(1))
        assert plan.next.kind == "review" and plan.next.item_id == "q0"

    def test_right_answers_space_reviews_out(self):
        learner = make(1)
        r = learner.record_answer("q0", True, when=D0)
        gaps = []
        for _ in range(4):
            when = r.next_review
            r = learner.record_answer("q0", True, when=when)
            gaps.append((r.next_review - when).days)
        assert gaps == sorted(gaps) and gaps[-1] > gaps[0]

    def test_multiple_choice_hits_count_for_less_than_free_recall(self):
        a, b = make(1), make(1)
        free = a.record_answer("q0", True, when=D0)
        mc = b.record_answer("q0", True, choices=4, when=D0)
        assert free.p_learned > mc.p_learned

    def test_datetime_and_date_are_both_accepted(self):
        learner = make(1)
        r = learner.record_answer("q0", True, when=datetime(2026, 3, 2, 21, 30))
        assert r.next_review > D0


class TestSignals:
    def test_hesitant_right_answers_grow_memory_less(self):
        def run(last_time):
            learner = make(8)
            for i in range(7):
                learner.record_answer(f"q{i}", True, response_time=2.0, when=D0)
            for i in range(6):
                learner.record_answer(f"q{i}", True, response_time=2.0, when=day(1))
            return learner.record_answer("q6", True, response_time=last_time, when=day(1))
        fast, slow = run(2.1), run(12.0)
        assert fast.hesitation == 0 and slow.hesitation == 1
        assert slow.stability < fast.stability and slow.p_learned < fast.p_learned

    def test_typical_response_time_needs_history(self):
        learner = make(6)
        for i in range(4):
            learner.record_answer(f"q{i}", True, response_time=3.0, when=D0)
        assert learner.typical_response_time() is None
        learner.record_answer("q4", True, response_time=3.0, when=D0)
        assert learner.typical_response_time() == 3.0

    def test_forgetting_after_gaps_teaches_the_memory_model(self):
        learner = make(12)
        for i in range(12):
            learner.record_answer(f"q{i}", True, when=D0)
        for i in range(12):
            learner.record_answer(f"q{i}", i < 2, when=day(6))
        assert learner.memory.observations == 12 and learner.memory.factor < 0.8
        assert learner.ability < 0

    def test_confusions(self):
        learner = make(3)
        for d in range(2):
            learner.record_answer("q0", False, confused_with="q1", when=day(d))
        learner.record_answer("q2", False, confused_with="q1", when=D0)
        assert learner.confusions() == [("q0", "q1", 2)]
        assert len(learner.confusions(min_times=1)) == 2

    def test_related_items_give_a_head_start(self):
        learner = AdaptiveLearner()
        learner.add_items(["base", "plain"])
        learner.add_item("linked", related=[("base", 0.8)])
        for d in range(3):
            learner.record_answer("base", True, when=day(d * 3))
        plan = {a.item_id: a for a in learner.recommend(when=day(7)).new_items}
        assert plan["linked"].starting_p > plan["plain"].starting_p

    def test_link_related_from_text(self):
        learner = AdaptiveLearner()
        texts = {"a": "Buenos días", "b": "Muy buenos días", "c": "Es mi casa", "d": "Es grande", "e": "Tengo frío"}
        learner.add_items(texts)
        assert learner.link_related(texts) > 0
        assert "b" in {o for o, _ in learner.items["a"].related}

    def test_time_budget(self):
        learner = make(30)
        for i in range(30):
            learner.record_answer(f"q{i}", False, when=D0)
        assert len(learner.recommend(when=day(1)).reviews) == 30
        short = learner.recommend(available_time=60, when=day(1))
        assert 0 < len(short.reviews) < 30 and short.estimated_minutes <= 1


class TestSaving:
    def test_round_trip_through_json(self):
        learner = make(6)
        for i in range(6):
            learner.record_answer(f"q{i}", i % 2 == 0, response_time=2 + i, when=D0)
        learner.record_answer("q1", False, confused_with="q3", when=day(1))
        learner.record_answer("q1", False, confused_with="q3", when=day(2))
        learner.record_answer("q0", True, when=day(3))
        copy = AdaptiveLearner.from_dict(json.loads(json.dumps(learner.to_dict())))
        assert copy.to_dict() == learner.to_dict()
        a, b = learner.recommend(when=day(4)), copy.recommend(when=day(4))
        assert [x.item_id for x in a.reviews] == [x.item_id for x in b.reviews]
        assert copy.confusions() == learner.confusions()

    def test_unknown_schema_is_rejected(self):
        with pytest.raises(ValueError, match="schema"):
            AdaptiveLearner.from_dict({"schema": 99})


class TestValidation:
    def test_unknown_item(self):
        with pytest.raises(KeyError, match="add_item"):
            AdaptiveLearner().record_answer("nope", True)

    @pytest.mark.parametrize("kwargs, error", [
        ({"correct": "yes"}, TypeError),
        ({"correct": True, "response_time": -1}, ValueError),
        ({"correct": True, "response_time": float("nan")}, ValueError),
        ({"correct": True, "choices": 1}, ValueError),
        ({"correct": False, "confused_with": "missing"}, KeyError),
        ({"correct": True, "when": "2026-03-02"}, TypeError),
    ])
    def test_bad_answers(self, kwargs, error):
        with pytest.raises(error):
            make(1).record_answer("q0", **kwargs)

    def test_answers_must_be_in_date_order(self):
        learner = make(1)
        learner.record_answer("q0", True, when=day(3))
        with pytest.raises(ValueError, match="date order"):
            learner.record_answer("q0", True, when=day(1))

    def test_bad_items_and_settings(self):
        learner = AdaptiveLearner()
        with pytest.raises(ValueError):
            learner.add_item("x", level="Expert")
        with pytest.raises(ValueError):
            learner.add_item("x", related=[("y", 1.5)])
        with pytest.raises(TypeError):
            learner.add_item(True)
        with pytest.raises(ValueError):
            AdaptiveLearner(daily_new=-1)
        with pytest.raises(ValueError):
            learner.recommend(available_time=-5)
