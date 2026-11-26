"""Spaced repetition on a forgetting curve.

Stability S is the number of days until predicted recall falls to TARGET_RETENTION,
so retention after t days is TARGET_RETENTION ** (t / S) and the next review is due
after about S days. Successful recall after a gap grows S (more for well-learned
phrases); a lapse shrinks it.
"""
from __future__ import annotations

from datetime import date, timedelta

TARGET_RETENTION = 0.90
INITIAL_STABILITY = 1.0
MIN_STABILITY = 1.0
MAX_STABILITY = 365.0
LAPSE_FACTOR = 0.4


def retention(stability: float, days_elapsed: float) -> float:
    if days_elapsed <= 0:
        return 1.0
    return TARGET_RETENTION ** (days_elapsed / max(stability, 1e-6))


def interval_days(stability: float) -> int:
    return max(1, round(stability))


def next_review(today: date, stability: float) -> date:
    return today + timedelta(days=interval_days(stability))


def grow(stability: float, p_learned: float, days_since_last: int | None, boost: float = 1.0) -> float:
    """Stability after a correct answer. Same-day repeats add no spacing benefit.

    `boost` scales the growth part of the factor: below 1 for a hesitant answer or a
    hard phrase, above 1 for an easy one."""
    if days_since_last is not None and days_since_last < 1:
        return stability
    factor = 1.5 + 1.5 * p_learned  # ~1.6x for shaky phrases, ~2.9x for known ones
    factor = 1 + (factor - 1) * max(boost, 0.0)
    return round(min(max(stability * factor, stability), MAX_STABILITY), 2)


def lapse(stability: float) -> float:
    return round(max(MIN_STABILITY, stability * LAPSE_FACTOR), 2)
