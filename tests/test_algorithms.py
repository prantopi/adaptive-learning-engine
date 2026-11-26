from datetime import date

import pytest

from learning_engine import bkt, srs
from learning_engine.grading import accepted_answers, check_typed
from learning_engine.scheduler import daily_new_quota, exercise_for

BEGINNER = bkt.params_for("Beginner")


class TestBKT:
    def test_correct_answer_raises_belief(self):
        assert bkt.update(0.3, True, BEGINNER) > 0.3

    def test_wrong_answer_lowers_belief_for_known_phrase(self):
        assert bkt.update(0.9, False, BEGINNER) < 0.9

    def test_matches_hand_calculation(self):
        # P(L|correct) = 0.3*0.9 / (0.3*0.9 + 0.7*0.25) = 0.6067; then + (1-0.6067)*0.2
        assert bkt.update(0.3, True, BEGINNER) == pytest.approx(0.6854, abs=1e-4)

    def test_typed_answer_is_stronger_evidence_than_multiple_choice(self):
        mc = bkt.update(0.3, True, BEGINNER, p_guess=0.25)
        typed = bkt.update(0.3, True, BEGINNER, p_guess=0.02)
        assert typed > mc

    def test_stays_within_probability_bounds(self):
        p = 0.0
        for _ in range(50):
            p = bkt.update(p, True, BEGINNER)
        assert 0.0 <= p <= 1.0
        for _ in range(50):
            p = bkt.update(p, False, BEGINNER)
        assert 0.0 <= p <= 1.0

    def test_mastery_needs_both_probability_and_streak(self):
        assert not bkt.is_mastered(0.99, 2)
        assert not bkt.is_mastered(0.90, 5)
        assert bkt.is_mastered(0.95, 3)

    def test_unknown_level_falls_back_to_beginner(self):
        assert bkt.params_for("Expert") == BEGINNER


class TestSRS:
    def test_retention_is_target_after_one_stability_period(self):
        assert srs.retention(10, 10) == pytest.approx(srs.TARGET_RETENTION)
        assert srs.retention(10, 0) == 1.0
        assert srs.retention(10, 20) < srs.retention(10, 5)

    def test_success_after_gap_grows_stability_more_for_known_phrases(self):
        weak = srs.grow(4.0, 0.2, days_since_last=4)
        strong = srs.grow(4.0, 0.95, days_since_last=4)
        assert 4.0 < weak < strong

    def test_same_day_success_does_not_grow_stability(self):
        assert srs.grow(4.0, 0.95, days_since_last=0) == 4.0

    def test_lapse_shrinks_but_never_below_minimum(self):
        assert srs.lapse(20.0) == 8.0
        assert srs.lapse(1.2) == srs.MIN_STABILITY

    def test_stability_is_capped(self):
        assert srs.grow(300, 1.0, days_since_last=300) == srs.MAX_STABILITY

    def test_next_review_is_at_least_tomorrow(self):
        assert srs.next_review(date(2026, 1, 1), 0.2) == date(2026, 1, 2)
        assert srs.next_review(date(2026, 1, 1), 7.4) == date(2026, 1, 8)


class TestScheduler:
    @pytest.mark.parametrize("score,low,high", [(0.0, 1, 1), (0.2, 1, 3), (0.4, 4, 7), (0.6, 8, 12),
                                                (0.8, 13, 18), (0.95, 20, 20), (1.0, 20, 20)])
    def test_quota_bands(self, score, low, high):
        assert low <= daily_new_quota(score) <= high

    def test_quota_never_decreases_as_score_rises(self):
        quotas = [daily_new_quota(s / 100) for s in range(101)]
        assert quotas == sorted(quotas)

    def test_exercises_get_harder_as_phrase_is_learned(self):
        assert exercise_for(0.1, "review") == "mc_meaning"
        assert exercise_for(0.6, "review") == "mc_phrase"
        assert exercise_for(0.9, "review") == "typed_meaning"
        assert exercise_for(0.9, "learn") == "mc_meaning"


class TestGrading:
    @pytest.mark.parametrize("answer", ["Thank you", "thank you very much", "thank you!", "Thank yuo very much"])
    def test_accepts_variants_and_small_typos(self, answer):
        assert check_typed(answer, "Thank you (very much)")

    @pytest.mark.parametrize("answer,translation", [("", "Good morning"), ("   ", "Good morning"),
                                                    ("Good evening", "Good morning"),
                                                    ("thanks a lot", "Thank you (very much)")])
    def test_rejects_wrong_answers(self, answer, translation):
        assert not check_typed(answer, translation)

    def test_alternatives_separated_by_slash(self):
        assert check_typed("okay", "Yes / Okay")
        assert check_typed("yes", "Yes / Okay")

    def test_leading_to_and_articles_ignored(self):
        assert check_typed("eat", "To eat")
        assert check_typed("the station", "Station")

    def test_short_words_need_exact_match(self):
        assert not check_typed("yen", "Yes")

    def test_accepted_answers_normalised(self):
        assert "thank you" in accepted_answers("Thank you (very much)")
