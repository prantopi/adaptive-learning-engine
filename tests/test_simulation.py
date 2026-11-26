"""The README's simulation numbers must be reproducible: same seed, same results."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("simulate", Path(__file__).parent.parent / "sim" / "simulate.py")
simulate = importlib.util.module_from_spec(spec)
sys.modules["simulate"] = simulate  # dataclasses look their module up here
spec.loader.exec_module(simulate)


def summary(policy, seed):
    learners = simulate.make_learners(3, random.Random(seed))
    recall, reviews, _ = simulate.run(policy, learners, seed)
    return round(recall, 10), round(reviews, 10)


def test_same_seed_same_results():
    for policy in (simulate.AdaptivePolicy, simulate.PersonalisedPolicy):
        assert summary(policy, 7) == summary(policy, 7)


def test_different_seed_different_results():
    assert summary(simulate.AdaptivePolicy, 7) != summary(simulate.AdaptivePolicy, 8)
