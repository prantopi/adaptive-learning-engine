from __future__ import annotations

import random

from adaptive_learning_engine import personalize
from adaptive_learning_engine.personalize import Observation, fit


def simulated_reviews(true_k: float, n: int, guess: float = 0.02, seed: int = 1) -> list[Observation]:
    """Reviews from someone whose memory really lasts true_k times the default."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        stability = rng.choice([1.0, 2.0, 4.0, 8.0])
        elapsed = max(1, round(stability * rng.uniform(0.7, 2.5)))
        recall = 0.9 ** (elapsed / (true_k * stability))
        p = recall + (1 - recall) * guess
        out.append(Observation(elapsed, stability, rng.random() < p, guess))
    return out


def test_no_data_means_default_model():
    result = fit([])
    assert result.factor == 1.0 and result.observations == 0 and not result.personalised


def test_same_day_answers_are_ignored():
    assert fit([Observation(0, 1.0, False, 0.02)] * 20).observations == 0


def test_recovers_a_fast_forgetter():
    result = fit(simulated_reviews(true_k=0.4, n=300))
    assert 0.3 < result.factor < 0.55
    assert result.low < 0.4 < result.high


def test_recovers_a_strong_memory():
    result = fit(simulated_reviews(true_k=2.5, n=300))
    assert 1.8 < result.factor < 3.3


def test_few_reviews_stay_close_to_default():
    one_miss = fit([Observation(3, 2.0, False, 0.02)])
    assert 0.6 < one_miss.factor < 1.0
    assert one_miss.high - one_miss.low > 0.6  # still very unsure


def test_more_evidence_narrows_the_estimate():
    small = fit(simulated_reviews(true_k=0.6, n=20))
    large = fit(simulated_reviews(true_k=0.6, n=400))
    assert (large.high - large.low) < (small.high - small.low)


def test_multiple_choice_hits_count_for_less_than_typed():
    # The same right answers after a long gap: typed ones are strong evidence of memory,
    # multiple-choice ones might be guesses.
    typed = fit([Observation(10, 2.0, True, 0.02)] * 8)
    mc = fit([Observation(10, 2.0, True, 0.25)] * 8)
    assert typed.factor > mc.factor > 1.0


def test_factor_is_bounded():
    assert fit([Observation(60, 1.0, True, 0.02)] * 200).factor <= personalize.MAX_FACTOR
    assert fit([Observation(1, 50.0, False, 0.02)] * 200).factor >= personalize.MIN_FACTOR
