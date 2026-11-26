"""A simple, stateful interface over the models.

    from adaptive_learning_engine import AdaptiveLearner

    learner = AdaptiveLearner()
    learner.add_items(["spanish_001", "spanish_002", "spanish_003"])
    learner.record_answer("spanish_001", correct=True, response_time=2.4)
    plan = learner.recommend(available_time=900)     # seconds
    plan.next                                        # the next best study action

One AdaptiveLearner holds everything about one learner (per subject): what they know,
their personal memory model, their answer speed and their study history. It keeps no
files or connections; save it with `to_dict()` (JSON-safe) and restore it with
`AdaptiveLearner.from_dict()`.

The individual models stay available as plain functions in the other modules.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Iterable

from . import bkt, difficulty, dropout, hesitation, personalize, related, srs
from .budget import ItemId
from .recommend import Item, Learner, Plan, recommend

FREE_RECALL_GUESS = 0.02   # a typed or spoken answer is rarely a lucky guess
TIMING_HISTORY = 50
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AnswerResult:
    item_id: ItemId
    correct: bool
    p_learned: float           # confidence the item is learned, after this answer
    stability: float           # base stability in days
    next_review: date
    hesitation: float          # 0 = usual speed, 1 = very slow (only for right answers)
    first_time: bool


def _day(when: date | datetime | None) -> int:
    if when is None:
        return date.today().toordinal()
    if isinstance(when, datetime):
        return when.date().toordinal()
    if isinstance(when, date):
        return when.toordinal()
    raise TypeError(f"when must be a date or datetime, not {type(when).__name__}")


def _exercise(choices: int | None) -> str:
    # Maps onto the question types the models were calibrated with.
    return "typed_meaning" if choices is None else "mc_meaning"


class AdaptiveLearner:
    """Everything the engine knows about one learner, with a small API around it."""

    def __init__(self, daily_new: int = 10):
        if not isinstance(daily_new, int) or daily_new < 0:
            raise ValueError("daily_new must be a whole number, 0 or more")
        self.daily_new = daily_new
        self._items: dict[ItemId, Item] = {}
        self._order: list[ItemId] = []                       # curriculum order for new items
        self._reviews: list[personalize.Observation] = []
        self._days: dict[int, list[int]] = {}                 # day -> [answers, correct]
        self._times: dict[str, list[int]] = {}                # exercise -> recent right-answer times (ms)
        self._learned_on: dict[int, int] = {}                 # day -> new items started
        self._ability = (0.0, 0)                              # Elo ability, answers rated
        self._confusions: Counter[tuple[ItemId, ItemId]] = Counter()

    # ── Items ────────────────────────────────────────────────────────────────

    def add_item(self, item_id: ItemId, level: str = "Beginner", difficulty: float | None = None,
                 related: Iterable[tuple[ItemId, float]] = ()) -> None:
        """Add an item to study. Items are introduced in the order they're added.
        Adding an existing item updates its level, difficulty and related items."""
        if not isinstance(item_id, (int, str)) or isinstance(item_id, bool):
            raise TypeError("item_id must be an int or str")
        if level not in bkt.LEVEL_PARAMS:
            raise ValueError(f"level must be one of {sorted(bkt.LEVEL_PARAMS)}")
        if difficulty is not None and not math.isfinite(difficulty):
            raise ValueError("difficulty must be a finite number")
        links = tuple((other, float(sim)) for other, sim in related)
        if any(not 0 <= sim <= 1 for _, sim in links):
            raise ValueError("related similarities must be between 0 and 1")
        if item_id in self._items:
            self._items[item_id] = replace(self._items[item_id], level=level, difficulty=difficulty, related=links)
            return
        self._items[item_id] = Item(item_id, level=level, difficulty=difficulty, related=links)
        self._order.append(item_id)

    def add_items(self, item_ids: Iterable[ItemId], level: str = "Beginner") -> None:
        for item_id in item_ids:
            self.add_item(item_id, level=level)

    def link_related(self, items: dict[ItemId, str] | None = None, readings: dict[ItemId, str] | None = None) -> int:
        """Link items by tf-idf text similarity (see related.py). `items` maps id -> text.
        Returns how many links were made."""
        texts = items or {}
        unknown = [i for i in texts if i not in self._items]
        if unknown:
            raise KeyError(f"unknown items: {unknown[:5]}")
        links = related.links([(i, t, (readings or {}).get(i, "")) for i, t in texts.items()])
        by_item: dict[ItemId, list[tuple[ItemId, float]]] = {}
        for a, b, sim in links:
            by_item.setdefault(a, []).append((b, sim))
        for item_id, rel in by_item.items():
            self._items[item_id] = replace(self._items[item_id], related=tuple(rel))
        return len(links)

    @property
    def items(self) -> dict[ItemId, Item]:
        return dict(self._items)

    # ── Answers ──────────────────────────────────────────────────────────────

    def record_answer(self, item_id: ItemId, correct: bool, response_time: float | None = None,
                      choices: int | None = None, when: date | datetime | None = None,
                      confused_with: ItemId | None = None) -> AnswerResult:
        """Record one answer.

        response_time  seconds from seeing the question to answering (optional)
        choices        number of options for multiple choice; None for free recall (typed, spoken)
        when           date or datetime of the answer; defaults to today
        confused_with  for a wrong answer: the item whose answer was given instead (optional)
        """
        if item_id not in self._items:
            raise KeyError(f"unknown item {item_id!r}; add it with add_item() first")
        if not isinstance(correct, bool):
            raise TypeError("correct must be True or False")
        if response_time is not None and (not math.isfinite(response_time) or response_time < 0):
            raise ValueError("response_time must be a number of seconds, 0 or more")
        if choices is not None and (not isinstance(choices, int) or choices < 2):
            raise ValueError("choices must be 2 or more, or None for free recall")
        if confused_with is not None and confused_with not in self._items:
            raise KeyError(f"unknown item {confused_with!r} in confused_with")
        day = _day(when)
        item = self._items[item_id]
        if item.last_review is not None and day < item.last_review:
            raise ValueError("answers must be recorded in date order for each item")

        exercise = _exercise(choices)
        guess = FREE_RECALL_GUESS if choices is None else 1 / choices
        params = bkt.params_for(item.level)
        response_ms = None if response_time is None else round(response_time * 1000)
        first_time = item.p_learned is None

        if first_time:
            p_before, _ = self._starting_confidence(item)
            stability = srs.INITIAL_STABILITY
            successes, days_since = 0, None
            self._learned_on[day] = self._learned_on.get(day, 0) + 1
        else:
            p_before, stability, successes = item.p_learned, item.stability, item.successes
            days_since = day - item.last_review
        spaced = days_since is not None and days_since >= 1

        # Response time: a slow right answer is weaker evidence and grows memory less.
        typical = hesitation.typical_ms(self._times.get(exercise, []))
        hesitant = hesitation.hesitation(response_ms, typical) if correct else 0.0
        p_new = bkt.update(p_before, correct, params, p_guess=hesitation.adjusted_guess(guess, hesitant))

        diff = item.difficulty if item.difficulty is not None else difficulty.LEVEL_PRIOR[item.level]
        if spaced:
            self._reviews.append(personalize.Observation(days_since, stability, correct, guess, successes))
            # One learner's answers rate the learner, not the items. Item difficulty is a
            # cross-learner estimate: compute it with difficulty.update over everyone's answers.
            skill, n = self._ability
            self._ability = (difficulty.update(skill, diff, n, 10**6, correct, exercise)[0], n + 1)

        memory = personalize.fit(self._reviews)
        if correct:
            if spaced:
                boost = hesitation.growth_boost(hesitant) * difficulty.growth_boost(diff, item.level)
                stability = srs.grow(stability, p_new, days_since, boost)
                successes += 1
            next_day = day + srs.interval_days(memory.scale(successes) * stability)
        else:
            stability = srs.lapse(stability)
            next_day = day + 1
            if confused_with is not None and confused_with != item_id:
                self._confusions[tuple(sorted((item_id, confused_with), key=str))] += 1

        self._items[item_id] = replace(item, p_learned=p_new, stability=stability, successes=successes,
                                       last_review=day, next_review=next_day)
        stats = self._days.setdefault(day, [0, 0])
        stats[0] += 1
        stats[1] += int(correct)
        if correct and response_ms:
            times = self._times.setdefault(exercise, [])
            times.append(min(response_ms, hesitation.CAP_MS))
            del times[:-TIMING_HISTORY]
        return AnswerResult(item_id, correct, p_new, stability, date.fromordinal(next_day), hesitant, first_time)

    # ── Recommendations ─────────────────────────────────────────────────────

    def recommend(self, available_time: float | None = None, when: date | datetime | None = None) -> Plan:
        """What to study now. available_time is in seconds (None = no limit)."""
        if available_time is not None and (not math.isfinite(available_time) or available_time < 0):
            raise ValueError("available_time must be a number of seconds, 0 or more")
        day = _day(when)
        typical = {ex: t for ex in self._times if (t := hesitation.typical_ms(self._times[ex])) is not None}
        learner = Learner(
            reviews=list(self._reviews),
            history=[dropout.DayStats(d, a, c) for d, (a, c) in sorted(self._days.items())],
            typical_ms=typical, daily_new=self.daily_new, learned_today=self._learned_on.get(day, 0))
        ordered = [self._items[i] for i in self._order]
        minutes = None if available_time is None else available_time / 60
        return recommend(ordered, learner, day, minutes=minutes)

    def due(self, when: date | datetime | None = None) -> list[ItemId]:
        """Items due for review on that day, using the learner's current memory model."""
        day = _day(when)
        return [i for i in self._order
                if self._items[i].next_review is not None and day >= self._items[i].next_review]

    # ── Insight ──────────────────────────────────────────────────────────────

    @property
    def memory(self) -> personalize.MemoryFit:
        """This learner's memory model: time scale k and growth g, with uncertainty."""
        return personalize.fit(self._reviews)

    @property
    def ability(self) -> float:
        return self._ability[0]

    def confusions(self, min_times: int = 2) -> list[tuple[ItemId, ItemId, int]]:
        """Pairs of items this learner has mixed up at least `min_times` times."""
        return [(a, b, n) for (a, b), n in self._confusions.most_common() if n >= min_times]

    def typical_response_time(self, choices: int | None = None) -> float | None:
        """The learner's usual time (seconds) for a right answer of this kind, once known."""
        ms = hesitation.typical_ms(self._times.get(_exercise(choices), []))
        return None if ms is None else ms / 1000

    # ── Saving ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-safe snapshot of everything; restore with AdaptiveLearner.from_dict()."""
        return {
            "schema": SCHEMA_VERSION,
            "daily_new": self.daily_new,
            "items": [{"id": i.item_id, "level": i.level, "difficulty": i.difficulty, "p_learned": i.p_learned,
                       "stability": i.stability, "successes": i.successes, "last_review": i.last_review,
                       "next_review": i.next_review,
                       "related": [list(r) for r in i.related]} for i in (self._items[k] for k in self._order)],
            "reviews": [[o.elapsed_days, o.stability, o.correct, o.guess_rate, o.successes_before] for o in self._reviews],
            "days": [[d, a, c] for d, (a, c) in sorted(self._days.items())],
            "times": self._times,
            "learned_on": [[d, n] for d, n in sorted(self._learned_on.items())],
            "ability": list(self._ability),
            "confusions": [[a, b, n] for (a, b), n in self._confusions.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveLearner":
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {data.get('schema')!r}; expected {SCHEMA_VERSION}")
        learner = cls(daily_new=data["daily_new"])
        for it in data["items"]:
            item = Item(it["id"], level=it["level"], difficulty=it["difficulty"], p_learned=it["p_learned"],
                        stability=it["stability"], successes=it["successes"], last_review=it["last_review"],
                        next_review=it["next_review"],
                        related=tuple((r[0], r[1]) for r in it["related"]))
            learner._items[item.item_id] = item
            learner._order.append(item.item_id)
        learner._reviews = [personalize.Observation(*r) for r in data["reviews"]]
        learner._days = {d: [a, c] for d, a, c in data["days"]}
        learner._times = {k: list(v) for k, v in data["times"].items()}
        learner._learned_on = {d: n for d, n in data["learned_on"]}
        learner._ability = (data["ability"][0], data["ability"][1])
        learner._confusions = Counter({(a, b): n for a, b, n in data["confusions"]})
        return learner

    # ── Internals ────────────────────────────────────────────────────────────

    def _starting_confidence(self, item: Item) -> tuple[float, float]:
        diff = item.difficulty if item.difficulty is not None else difficulty.LEVEL_PRIOR[item.level]
        base = difficulty.initial_p(bkt.params_for(item.level).p_init, diff, item.level)
        known = [(sim, self._items[o].p_learned) for o, sim in item.related
                 if o in self._items and self._items[o].p_learned is not None]
        boost = related.transfer_boost(known)
        return related.boosted_p(base, boost), boost

