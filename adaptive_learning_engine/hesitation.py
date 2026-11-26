"""Response time as evidence: a right answer after a long pause shows weaker memory.

Speed is judged against the user's own typical time for that exercise type, so a slow
typist isn't penalised. A hesitant correct answer:

- counts as weaker evidence in BKT: its guess rate rises towards 0.5, as if it were
  partly a lucky or effortful reconstruction rather than recall;
- grows stability less, like the "hard" grade in Anki/FSRS.

Wrong answers are unaffected, and so are answers with no usable timing.
"""
from __future__ import annotations

import statistics

FAST_RATIO = 1.5        # up to 1.5x your usual time: no penalty
SLOW_RATIO = 4.0        # 4x or slower: full penalty
MIN_HISTORY = 5         # timed correct answers needed before judging speed
CAP_MS = 60_000         # longer than a minute is probably a distraction, not thinking
MAX_GUESS = 0.5
GROWTH_DAMPING = 0.6    # a fully hesitant answer keeps 40% of the normal stability growth

DEFAULT_MS = {"mc_meaning": 5000, "mc_phrase": 6000, "typed_meaning": 10000}


def typical_ms(history_ms: list[int]) -> float | None:
    usable = [min(ms, CAP_MS) for ms in history_ms if ms and ms > 0]
    return statistics.median(usable) if len(usable) >= MIN_HISTORY else None


def hesitation(response_ms: int | None, typical: float | None) -> float:
    """0 = normal speed, 1 = very slow. 0 when there isn't enough information."""
    if not response_ms or response_ms <= 0 or not typical:
        return 0.0
    ratio = min(response_ms, CAP_MS) / typical
    return round(min(max((ratio - FAST_RATIO) / (SLOW_RATIO - FAST_RATIO), 0.0), 1.0), 3)


def adjusted_guess(guess: float, h: float) -> float:
    return guess + h * (max(MAX_GUESS, guess) - guess)


def growth_boost(h: float) -> float:
    return 1 - GROWTH_DAMPING * h
