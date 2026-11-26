"""Short on time? Pick the reviews that save the most memory per second.

For each due phrase we estimate how much reviewing it now raises its predicted recall a
week from now, compared with leaving it:

    leave it:   0.9 ** ((t + H) / S)
    review now: r * 0.9 ** (H / S_grown) + (1 - r) * 0.9 ** (H / S_lapsed),   r = 0.9 ** (t / S)

and divide by how long a review of that kind usually takes you. Then we take phrases
greedily, best value per second first, until the time budget is used up. This is the
standard greedy solution to a knapsack problem, close to optimal when items are small
compared with the budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

ItemId = Union[int, str]   # any hashable id works; int or str are the documented types

from . import srs

HORIZON_DAYS = 7
FEEDBACK_SECONDS = 4.0   # reading the feedback and pressing Next


@dataclass(frozen=True)
class Candidate:
    item_id: ItemId
    stability: float     # effective stability (base x personal scale)
    elapsed_days: float
    p_learned: float
    seconds: float       # expected time to answer, plus feedback


def recall_gain(stability: float, elapsed_days: float, p_learned: float, horizon: float = HORIZON_DAYS) -> float:
    s = max(stability, 0.1)
    r = srs.TARGET_RETENTION ** (max(elapsed_days, 0) / s)
    leave = srs.TARGET_RETENTION ** ((max(elapsed_days, 0) + horizon) / s)
    grown = s * (1.5 + 1.5 * p_learned)
    lapsed = max(srs.MIN_STABILITY, s * srs.LAPSE_FACTOR)
    review = r * srs.TARGET_RETENTION ** (horizon / grown) + (1 - r) * srs.TARGET_RETENTION ** (horizon / lapsed)
    return max(review - leave, 0.0)


def value_per_second(c: Candidate) -> float:
    return recall_gain(c.stability, c.elapsed_days, c.p_learned) / max(c.seconds, 1.0)


def choose(candidates: list[Candidate], budget_seconds: float | None) -> list[Candidate]:
    ranked = sorted(candidates, key=value_per_second, reverse=True)
    if budget_seconds is None:
        return ranked
    chosen, used = [], 0.0
    for c in ranked:
        if chosen and used + c.seconds > budget_seconds:
            continue
        chosen.append(c)
        used += c.seconds
    return chosen
