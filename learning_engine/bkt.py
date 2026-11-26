"""Bayesian Knowledge Tracing.

Each phrase has a hidden state: learned or not learned. P(L) is our belief that it is
learned. After every answer we apply Bayes' rule (allowing for lucky guesses and
careless slips), then add the chance that the learner picked it up during this attempt.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BKTParams:
    p_init: float     # P(L0): known before any practice
    p_transit: float  # P(T): learned during one attempt
    p_guess: float    # P(G): correct without knowing it
    p_slip: float     # P(S): wrong despite knowing it


LEVEL_PARAMS: dict[str, BKTParams] = {
    "Beginner": BKTParams(p_init=0.15, p_transit=0.20, p_guess=0.25, p_slip=0.10),
    "Intermediate": BKTParams(p_init=0.10, p_transit=0.15, p_guess=0.20, p_slip=0.10),
    "Advanced": BKTParams(p_init=0.05, p_transit=0.10, p_guess=0.15, p_slip=0.08),
}

MASTERY_THRESHOLD = 0.95
MASTERY_STREAK = 3


def params_for(level: str) -> BKTParams:
    return LEVEL_PARAMS.get(level, LEVEL_PARAMS["Beginner"])


def update(p_learned: float, correct: bool, params: BKTParams, p_guess: float | None = None) -> float:
    """One BKT step. `p_guess` overrides the level default (e.g. 0.25 for 4-option
    multiple choice, near 0 for a typed answer)."""
    guess = params.p_guess if p_guess is None else p_guess
    slip = params.p_slip
    if correct:
        evidence_learned = p_learned * (1 - slip)
        evidence_total = evidence_learned + (1 - p_learned) * guess
    else:
        evidence_learned = p_learned * slip
        evidence_total = evidence_learned + (1 - p_learned) * (1 - guess)
    posterior = evidence_learned / evidence_total if evidence_total > 0 else p_learned
    new_p = posterior + (1 - posterior) * params.p_transit
    return round(min(max(new_p, 0.0), 1.0), 4)


def is_mastered(p_learned: float, correct_streak: int) -> bool:
    return p_learned >= MASTERY_THRESHOLD and correct_streak >= MASTERY_STREAK
