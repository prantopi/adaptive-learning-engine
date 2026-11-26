"""How hard is each phrase? Learned from everyone's answers with an Elo-style rating.

Every user has an ability rating and every phrase a difficulty rating (per language). For
each spaced answer the expected chance of a right answer is

    P(correct) = guess + (1 - guess) * sigmoid(ability - difficulty + ease[exercise])

and both ratings move towards what happened (online gradient step on the log-loss, the
same update as Elo or a one-parameter IRT model). Step sizes shrink as a rating sees more
answers, so it settles. Ratings start from the CSV level (Beginner easier).

Uses:
- a new phrase starts with lower confidence if it's harder than its level suggests;
- harder phrases grow stability more slowly after a right answer, so they come back sooner.
"""
from __future__ import annotations

import math

LEVEL_PRIOR = {"Beginner": -0.5, "Intermediate": 0.0, "Advanced": 0.5}
EASE = {"mc_meaning": 1.0, "mc_phrase": 0.5, "typed_meaning": -0.5}
GUESS = {"mc_meaning": 0.25, "mc_phrase": 0.25, "typed_meaning": 0.02}
HARD, EASY = 0.6, -0.6   # relative to the level prior, for labels
MIN_ANSWERS_FOR_LABEL = 5


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def expected(ability: float, difficulty: float, exercise: str) -> float:
    g = GUESS.get(exercise, 0.25)
    return g + (1 - g) * sigmoid(ability - difficulty + EASE.get(exercise, 0.0))


def _step(base: float, n: int) -> float:
    return max(0.05, base / (1 + 0.08 * n))


def update(ability: float, difficulty: float, user_n: int, item_n: int, correct: bool,
           exercise: str) -> tuple[float, float]:
    err = (1.0 if correct else 0.0) - expected(ability, difficulty, exercise)
    return (round(ability + _step(0.6, user_n) * err, 4),
            round(difficulty - _step(0.8, item_n) * err, 4))


def relative(difficulty: float, level: str) -> float:
    """How much harder (+) or easier (-) than a typical phrase of its level."""
    return difficulty - LEVEL_PRIOR.get(level, 0.0)


def initial_p(p_init: float, difficulty: float, level: str) -> float:
    return round(sigmoid(logit(p_init) - 0.5 * relative(difficulty, level)), 4)


def growth_boost(difficulty: float, level: str) -> float:
    return min(max(math.exp(-0.25 * relative(difficulty, level)), 0.7), 1.3)


def label(difficulty: float, level: str, answers: int) -> str | None:
    if answers < MIN_ANSWERS_FOR_LABEL:
        return None
    r = relative(difficulty, level)
    return "hard" if r >= HARD else "easy" if r <= EASY else None
