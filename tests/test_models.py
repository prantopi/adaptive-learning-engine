"""Unit tests for the learning models in app/algorithms (no database)."""
from __future__ import annotations

import random

import numpy as np

from learning_engine import budget, difficulty, dropout, hesitation, related
from learning_engine.personalize import Observation, fit


# 2. Response time
class TestHesitation:
    def test_needs_history(self):
        assert hesitation.typical_ms([3000] * 4) is None
        assert hesitation.typical_ms([3000] * 5) == 3000

    def test_scale(self):
        assert hesitation.hesitation(4000, 3000) == 0.0       # 1.3x: normal
        assert hesitation.hesitation(12_000, 3000) == 1.0     # 4x: fully hesitant
        assert 0 < hesitation.hesitation(7500, 3000) < 1      # 2.5x: in between
        assert hesitation.hesitation(None, 3000) == 0.0
        assert hesitation.hesitation(9000, None) == 0.0

    def test_long_pauses_are_capped(self):
        # 10 minutes away from the screen isn't treated as 10 minutes of thinking.
        assert hesitation.hesitation(600_000, 30_000) == hesitation.hesitation(60_000, 30_000)

    def test_effects(self):
        assert hesitation.adjusted_guess(0.02, 0) == 0.02
        assert hesitation.adjusted_guess(0.02, 1) == 0.5
        assert hesitation.growth_boost(0) == 1 and hesitation.growth_boost(1) < 0.5


# 3. Phrase difficulty
class TestDifficulty:
    def test_wrong_answers_make_a_phrase_harder_and_the_learner_weaker(self):
        ability, diff = difficulty.update(0.0, 0.0, 0, 0, False, "mc_meaning")
        assert diff > 0 and ability < 0

    def test_surprising_results_move_ratings_more(self):
        _, after_expected_hit = difficulty.update(0.0, 0.0, 0, 0, True, "mc_meaning")
        _, after_surprising_miss = difficulty.update(0.0, 0.0, 0, 0, False, "mc_meaning")
        assert abs(after_surprising_miss) > abs(after_expected_hit)

    def test_steps_shrink_with_experience(self):
        _, new = difficulty.update(0.0, 0.0, 0, 0, False, "typed_meaning")
        _, seasoned = difficulty.update(0.0, 0.0, 0, 200, False, "typed_meaning")
        assert new > seasoned > 0

    def test_ratings_converge_to_a_known_difficulty(self):
        rng = random.Random(3)
        true_diff, diff, n = 1.0, 0.0, 0
        for _ in range(600):
            ability = rng.gauss(0, 0.5)  # many different learners, known ability
            correct = rng.random() < difficulty.expected(ability, true_diff, "typed_meaning")
            _, diff = difficulty.update(ability, diff, 0, n, correct, "typed_meaning")
            n += 1
        assert abs(diff - true_diff) < 0.35

    def test_uses(self):
        assert difficulty.initial_p(0.15, 1.5, "Beginner") < 0.15 < difficulty.initial_p(0.15, -1.5, "Beginner")
        assert difficulty.growth_boost(2.0, "Beginner") < 1 < difficulty.growth_boost(-2.0, "Beginner")
        assert difficulty.label(1.0, "Beginner", 3) is None           # not enough answers yet
        assert difficulty.label(1.0, "Beginner", 10) == "hard"
        assert difficulty.label(-0.5, "Beginner", 10) is None         # typical for its level


# 4. Personal memory growth
def reviews_with_growth(k: float, g: float, n: int, seed: int = 2) -> list[Observation]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        successes = rng.randint(0, 8)
        stability = 2.0 ** min(successes, 6)
        elapsed = max(1, round(stability * rng.uniform(0.5, 2.0)))
        recall = 0.9 ** (elapsed / (k * g ** successes * stability))
        out.append(Observation(elapsed, stability, rng.random() < recall, 0.02, successes))
    return out


def test_memory_fit_separates_scale_from_growth():
    weak_growth = fit(reviews_with_growth(k=1.0, g=0.8, n=400))
    assert weak_growth.growth < 0.9 and 0.6 < weak_growth.factor < 1.6
    assert weak_growth.scale(6) < weak_growth.scale(0)        # later reviews come relatively sooner
    strong_growth = fit(reviews_with_growth(k=1.0, g=1.2, n=400))
    assert strong_growth.growth > 1.1


# 5. Related phrases
class TestRelated:
    def test_tokens(self):
        assert related.tokens("¿Cómo estás?", "KOH-mo") == ["como", "esta"]
        assert related.tokens("Buenos días", "") == related.tokens("Buenas días", "")
        assert related.tokens("おはようございます", "Ohayou gozaimasu") == ["ohayo", "gozaimas"]  # romaji, lightly stemmed

    def test_links_prefer_distinctive_shared_words(self):
        phrases = [(1, "Buenos días", ""), (2, "Buenas noches", ""), (3, "Es mi casa", ""),
                   (4, "Es tu perro", ""), (5, "Muy buenos días", ""), (6, "Es grande", "")]
        links = {(a, b): s for a, b, s in related.links(phrases)}
        assert (1, 5) in links                       # share "buen" and "dias"
        assert links.get((1, 2), 0) < links[(1, 5)]  # share only "buen"
        assert (3, 4) not in links                   # share only the very common "es"

    def test_transfer(self):
        assert related.transfer_boost([]) == 0
        assert related.transfer_boost([(0.6, 0.9), (0.3, 0.2)]) == round(0.5 * 0.54, 4)
        assert related.transfer_boost([(1.0, 1.0)]) == related.MAX_BOOST
        assert related.boosted_p(0.15, 0.2) == round(0.15 + 0.85 * 0.2, 4)


# 6. Time budget
class TestBudget:
    def test_overdue_and_shaky_phrases_are_worth_more(self):
        fresh_known = budget.recall_gain(stability=20, elapsed_days=1, p_learned=0.95)
        overdue_shaky = budget.recall_gain(stability=2, elapsed_days=5, p_learned=0.3)
        assert overdue_shaky > fresh_known >= 0

    def test_fits_the_budget(self):
        items = [budget.Candidate(i, stability=2, elapsed_days=2 + i % 3, p_learned=0.5, seconds=10) for i in range(30)]
        chosen = budget.choose(items, budget_seconds=60)
        assert len(chosen) == 6 and sum(c.seconds for c in chosen) <= 60
        assert len(budget.choose(items, None)) == 30

    def test_always_offers_at_least_one(self):
        assert len(budget.choose([budget.Candidate(1, 2, 2, 0.5, seconds=90)], budget_seconds=30)) == 1

    def test_prefers_value_per_second(self):
        quick = budget.Candidate(1, stability=2, elapsed_days=3, p_learned=0.5, seconds=6)
        slow = budget.Candidate(2, stability=2, elapsed_days=3, p_learned=0.5, seconds=20)
        assert budget.choose([slow, quick], budget_seconds=10) == [quick]


# 7. Quit risk
class TestDropout:
    def history(self, days):
        return [dropout.DayStats(d, 10, 8) for d in days]

    def test_no_history_no_prediction(self):
        assert dropout.features([], 100) is None
        assert dropout.risk(dropout.PRIOR_WEIGHTS, None) is None

    def test_long_gap_is_riskier_than_a_steady_habit(self):
        steady = dropout.features(self.history(range(90, 100)), 100)
        lapsed = dropout.features(self.history(range(80, 87)), 100)
        w = dropout.PRIOR_WEIGHTS
        assert dropout.risk(w, lapsed) > 0.55 > 0.2 > dropout.risk(w, steady)

    def test_falling_accuracy_raises_risk(self):
        good = [dropout.DayStats(d, 10, 9) for d in range(86, 100)]
        falling = [dropout.DayStats(d, 10, 9 if d < 93 else 4) for d in range(86, 100)]
        w = dropout.PRIOR_WEIGHTS
        assert dropout.risk(w, dropout.features(falling, 100)) > dropout.risk(w, dropout.features(good, 100))

    def test_labels(self):
        rows = dropout.training_rows(self.history([0, 1, 2, 20]), last_labelled_day=12)
        labels = {i + 1: y for i, (_, y) in enumerate(rows)}
        assert labels[1] == 0 and labels[2] == 0      # studied again within a week
        assert labels[3] == 1 and labels[12] == 1     # nothing until day 20

    def test_fit_learns_from_data_and_keeps_the_prior_without_it(self):
        assert np.allclose(dropout.fit([]), dropout.PRIOR_WEIGHTS)
        # Synthetic users for whom the gap matters much more than the prior assumes.
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(3000):
            x = np.array([1.0, rng.uniform(0, 3), rng.uniform(0, 1), 0.0, 0.0])
            p = 1 / (1 + np.exp(-(-4.0 + 3.0 * x[1])))
            rows.append((x, int(rng.random() < p)))
        w = dropout.fit(rows)
        assert w[1] > 2.3   # gap weight moved from 1.3 towards the true 3.0
