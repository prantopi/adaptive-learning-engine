"""Daily planning rules: how much new material, and which exercise type to show."""
from __future__ import annotations

DEFAULT_QUOTA = 10
MAX_QUOTA = 20
EXAM_SIZE = 15
MIN_EXAM_PHRASES = 3

EXERCISE_TYPES = ("mc_meaning", "mc_phrase", "typed_meaning")


def daily_new_quota(exam_score: float) -> int:
    """New phrases for the next day, from the daily test score (0.0 to 1.0).

    under 30% -> 1-3, 30-50% -> 4-7, 50-70% -> 8-12, 70-90% -> 13-18, 90%+ -> 20
    """
    s = min(max(exam_score, 0.0), 1.0)
    if s < 0.30:
        return max(1, round(s * 10))
    if s < 0.50:
        return round(4 + (s - 0.30) * 15)
    if s < 0.70:
        return round(8 + (s - 0.50) * 20)
    if s < 0.90:
        return round(13 + (s - 0.70) * 25)
    return MAX_QUOTA


def exercise_for(p_learned: float, mode: str) -> str:
    """Harder exercise types as the phrase becomes better known."""
    if mode == "learn" or p_learned < 0.40:
        return "mc_meaning"   # see the phrase, pick its meaning
    if p_learned < 0.75:
        return "mc_phrase"    # see the meaning, pick the phrase
    return "typed_meaning"    # see the phrase, type its meaning


def guess_rate(exercise_type: str) -> float:
    return 0.02 if exercise_type == "typed_meaning" else 0.25
