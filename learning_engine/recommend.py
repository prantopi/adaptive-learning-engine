"""Put the models together: given a learner's state, what should they study next?

    Learner answers -> knowledge state -> memory prediction -> difficulty + ability
                    -> daily load -> time constraints -> next best study action

`recommend()` is a pure function. You pass in what you've stored about the learner and
their items; it returns a plan with the reviews to do (most valuable first), the new
items to start, and the single next action, each with a short reason. Nothing is
written anywhere, so you decide how to store state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import budget, difficulty, dropout, personalize, related, srs
from .hesitation import DEFAULT_MS
from .scheduler import exercise_for

LIGHT_DAY = {"high": (0.5, 3, 5), "medium": (0.75, 5, 10)}   # quota share, minimum quota, suggested minutes
NEW_ITEM_SECONDS = 15.0     # reading a new card and answering its quick check
MIN_STUDY_DAYS = 2          # no quit-risk prediction before this


@dataclass(frozen=True)
class Item:
    """One item (flashcard, phrase, question) and what is known about it for this learner."""
    item_id: int
    level: str = "Beginner"
    difficulty: float | None = None           # Elo rating; None = the level's prior
    p_learned: float | None = None            # None = never studied
    stability: float = srs.INITIAL_STABILITY  # base stability in days
    successes: int = 0                        # successful spaced reviews
    last_review: int | None = None            # day number of the last answer, incl. learning (date.toordinal())
    related: tuple[tuple[int, float], ...] = ()   # (other item_id, similarity), e.g. from related.links

    @property
    def started(self) -> bool:
        return self.p_learned is not None and self.last_review is not None


@dataclass(frozen=True)
class Learner:
    reviews: list[personalize.Observation] = field(default_factory=list)  # for the memory model
    history: list[dropout.DayStats] = field(default_factory=list)         # for quit risk
    typical_ms: dict[str, float] = field(default_factory=dict)           # usual answer time per exercise type
    dropout_weights: object = None            # numpy array from dropout.fit; None = the prior weights
    daily_new: int = 10                       # normal new-item allowance
    learned_today: int = 0


@dataclass(frozen=True)
class Action:
    kind: str                # "review" or "learn"
    item_id: int
    exercise: str
    seconds: float
    reason: str
    value_per_second: float = 0.0     # reviews: recall gained a week out, per second
    predicted_recall: float | None = None
    starting_p: float | None = None   # new items: confidence before the first answer


@dataclass(frozen=True)
class Plan:
    next: Action | None
    reviews: list[Action]
    new_items: list[Action]
    due: int                       # reviews due in total (before any time budget)
    new_quota: int
    quit_risk: float | None
    light_day: bool
    minutes: float | None          # the time budget used, if any
    estimated_minutes: float
    memory: personalize.MemoryFit


def _seconds(learner: Learner, exercise: str) -> float:
    return learner.typical_ms.get(exercise, DEFAULT_MS[exercise]) / 1000 + budget.FEEDBACK_SECONDS


def _starting_p(item: Item, by_id: dict[int, Item]) -> tuple[float, int]:
    diff = item.difficulty if item.difficulty is not None else difficulty.LEVEL_PRIOR.get(item.level, 0.0)
    base = difficulty.initial_p(0.15, diff, item.level)
    known = [(sim, by_id[o].p_learned) for o, sim in item.related
             if o in by_id and by_id[o].p_learned is not None]
    boost = related.transfer_boost(known)
    return related.boosted_p(base, boost), sum(1 for _, p in known if p >= 0.5)


def recommend(items: list[Item], learner: Learner, today: int, minutes: float | None = None) -> Plan:
    """today: day number (date.toordinal()). minutes: time available, or None for no limit."""
    # Memory prediction: this learner's own forgetting curve.
    memory = personalize.fit(learner.reviews)

    # Daily load: lighten the day when quitting looks likely.
    risk, level = None, "unknown"
    if len([d for d in learner.history if d.day < today]) >= MIN_STUDY_DAYS:
        weights = learner.dropout_weights if learner.dropout_weights is not None else dropout.PRIOR_WEIGHTS
        risk = dropout.risk(weights, dropout.features(learner.history, today))
        level = dropout.level(risk)
    quota = learner.daily_new
    if level in LIGHT_DAY:
        share, minimum, suggested = LIGHT_DAY[level]
        quota = max(minimum, round(learner.daily_new * share))
        minutes = minutes if minutes is not None else suggested
    light_day = quota < learner.daily_new

    # Reviews that are due, valued by recall gained per second (knowledge state, memory, difficulty).
    by_id = {i.item_id: i for i in items}
    candidates, info = [], {}
    for item in items:
        if not item.started:
            continue
        effective = memory.scale(item.successes) * item.stability
        elapsed = today - item.last_review
        if elapsed < srs.interval_days(effective):
            continue
        exercise = exercise_for(item.p_learned, "review")
        candidates.append(budget.Candidate(item.item_id, effective, elapsed, item.p_learned, _seconds(learner, exercise)))
        info[item.item_id] = (exercise, srs.retention(effective, elapsed), elapsed - srs.interval_days(effective))
    due = len(candidates)
    budget_seconds = minutes * 60 if minutes is not None else None
    chosen = budget.choose(candidates, budget_seconds)

    reviews = []
    for c in chosen:
        exercise, recall, overdue = info[c.item_id]
        item = by_id[c.item_id]
        hard = item.difficulty is not None and difficulty.relative(item.difficulty, item.level) >= difficulty.HARD
        reason = f"predicted recall {recall:.0%}"
        if overdue:
            reason += f", {overdue} day{'s' if overdue != 1 else ''} overdue"
        if hard:
            reason += ", a hard item"
        gain = budget.recall_gain(c.stability, c.elapsed_days, c.p_learned)
        reason += f"; reviewing now adds {gain * 100:.0f} points of recall a week from now"
        reviews.append(Action("review", c.item_id, exercise, round(c.seconds, 1), reason,
                              round(budget.value_per_second(c), 5), round(recall, 3)))

    # New items: up to today's allowance, if time is left.
    used = sum(r.seconds for r in reviews)
    new_items = []
    remaining = max(0, quota - learner.learned_today)
    for item in items:
        if len(new_items) >= remaining:
            break
        if item.p_learned is not None:
            continue
        if budget_seconds is not None and used + NEW_ITEM_SECONDS > budget_seconds:
            break
        p0, helpers = _starting_p(item, by_id)
        reason = "new" + (f"; builds on {helpers} item{'s' if helpers != 1 else ''} you know" if helpers else "")
        new_items.append(Action("learn", item.item_id, "mc_meaning", NEW_ITEM_SECONDS, reason, starting_p=p0))
        used += NEW_ITEM_SECONDS

    # Reviews first: forgetting what you know costs more than starting something new a day later.
    next_action = reviews[0] if reviews else new_items[0] if new_items else None
    return Plan(next=next_action, reviews=reviews, new_items=new_items, due=due, new_quota=quota,
                quit_risk=None if risk is None else round(risk, 3), light_day=light_day, minutes=minutes,
                estimated_minutes=round(used / 60, 1), memory=memory)
