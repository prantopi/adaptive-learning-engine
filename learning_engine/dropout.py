"""Predict when someone is about to stop studying, and make that day lighter.

A logistic regression predicts P(no study in the next 7 days) from four features of the
recent history:

    gap        log(1 + days since the last study day)
    active     share of the last 14 days with any study
    trend      accuracy over the last 7 days minus the 7 days before
    accuracy   accuracy over the last 7 days, minus 0.8

Starting weights are set by hand from common sense (long gaps and falling accuracy are bad
signs). They are then refitted every day on all users' histories as a Bayesian (MAP)
logistic regression with a Gaussian prior centred on the starting weights. With little
data, the weights stay close to the starting values; with lots of data, the data wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FEATURES = ("gap", "active", "trend", "accuracy")
PRIOR_WEIGHTS = np.array([-2.2, 1.3, -2.0, -1.5, -1.0])   # bias, then FEATURES
PRIOR_SD = 1.0
HORIZON_DAYS = 7
MEDIUM, HIGH = 0.35, 0.55


@dataclass(frozen=True)
class DayStats:
    day: int          # day number (date.toordinal())
    answers: int
    correct: int


def features(history: list[DayStats], today: int) -> np.ndarray | None:
    """Features as of the start of `today`, from study days strictly before it."""
    past = [d for d in history if d.day < today]
    if not past:
        return None
    gap = today - max(d.day for d in past)
    active = sum(1 for d in past if today - d.day <= 14) / 14

    def accuracy(lo: int, hi: int) -> float | None:
        days = [d for d in past if lo <= today - d.day <= hi]
        answers = sum(d.answers for d in days)
        return sum(d.correct for d in days) / answers if answers else None

    recent, before = accuracy(1, 7), accuracy(8, 14)
    trend = recent - before if recent is not None and before is not None else 0.0
    acc = (recent - 0.8) if recent is not None else 0.0
    return np.array([1.0, math.log1p(gap), active, trend, acc])


def training_rows(history: list[DayStats], last_labelled_day: int) -> list[tuple[np.ndarray, int]]:
    """One row per day after the first study day: did they study in the next 7 days?"""
    if not history:
        return []
    days = {d.day for d in history}
    rows = []
    for today in range(min(days) + 1, last_labelled_day + 1):
        x = features(history, today)
        if x is None:
            continue
        quit_ = not any(today <= d < today + HORIZON_DAYS for d in days)
        rows.append((x, int(quit_)))
    return rows


def fit(rows: list[tuple[np.ndarray, int]], iterations: int = 25) -> np.ndarray:
    """MAP logistic regression (Newton's method) with prior N(PRIOR_WEIGHTS, PRIOR_SD^2)."""
    w = PRIOR_WEIGHTS.copy()
    if not rows:
        return w
    X = np.array([x for x, _ in rows])
    y = np.array([label for _, label in rows], dtype=float)
    precision = np.eye(len(w)) / PRIOR_SD ** 2
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) + precision @ (w - PRIOR_WEIGHTS)
        hess = (X.T * (p * (1 - p))) @ X + precision
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.abs(step).max() < 1e-6:
            break
    return w


def risk(weights: np.ndarray, x: np.ndarray | None) -> float | None:
    if x is None:
        return None
    return float(1 / (1 + np.exp(-(x @ weights))))


def level(p: float | None) -> str:
    if p is None:
        return "unknown"
    return "high" if p >= HIGH else "medium" if p >= MEDIUM else "low"
