"""Learn each user's memory from their own review results.

The base spaced-repetition model predicts recall after t days as 0.9 ** (t / S), and grows
S by fixed constants after each successful review. People differ in two ways, so we fit
two numbers per user and language:

    P(recall) = 0.9 ** (t / (k * g**n * S))

k  time scale:  k > 1 you remember longer than assumed from the start; k < 1 you forget faster.
g  growth:      how much more (g > 1) or less (g < 1) each successful spaced review strengthens
                your memory than the default constants assume. n is how many successful
                spaced reviews that phrase has had. A single k can't correct a growth rate
                that's wrong, because the error compounds with every review.

The fit is Bayesian on a 2-D grid over (log k, log g). Priors are centred on the default
model (k = 1, g = 1), so a few reviews barely move it and the estimate shifts as evidence
builds up. Each observation is one review answered after a gap of at least a day. A correct
multiple-choice answer can be a lucky guess, so the likelihood includes the guess rate:

    P(correct) = r + (1 - r) * guess,   where r = 0.9 ** (t / (k * g**n * S))
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .srs import TARGET_RETENTION

K_PRIOR_SD = 0.5        # on log k: before any data, k is probably between about 0.37 and 2.7
G_PRIOR_SD = 0.1        # on log g: growth probably within about ±20% of the default
MIN_FACTOR, MAX_FACTOR = 0.05, 4.0      # wide on purpose: a 0.25 floor clipped weak learners in the simulation
MIN_GROWTH, MAX_GROWTH = 0.6, 1.5
MIN_SCALE, MAX_SCALE = 0.02, 8.0        # bounds on k * g**n when scheduling
MAX_SUCCESSES = 12                      # n is capped so g**n can't run away
MAX_OBSERVATIONS = 500  # the most recent reviews; memory habits can change over time
CONFIDENT_AFTER = 10    # reviews needed before the UI describes the timing as personal

_LOG_K = np.linspace(math.log(MIN_FACTOR), math.log(MAX_FACTOR), 57)
_LOG_G = np.linspace(math.log(MIN_GROWTH), math.log(MAX_GROWTH), 31)
_LN_TARGET = math.log(TARGET_RETENTION)


@dataclass(frozen=True)
class Observation:
    elapsed_days: float     # days since the previous review
    stability: float        # base stability before this review
    correct: bool
    guess_rate: float       # chance of a lucky correct answer for this exercise type
    successes_before: int = 0  # successful spaced reviews of this phrase before this one


@dataclass(frozen=True)
class MemoryFit:
    factor: float           # posterior mean of k
    low: float              # 80% credible interval for k
    high: float
    growth: float           # posterior mean of g
    growth_low: float
    growth_high: float
    observations: int

    @property
    def personalised(self) -> bool:
        return self.observations >= CONFIDENT_AFTER

    def scale(self, successes: int) -> float:
        """Multiplier on base stability for a phrase with this many successful spaced reviews."""
        s = self.factor * self.growth ** min(max(successes, 0), MAX_SUCCESSES)
        return min(max(s, MIN_SCALE), MAX_SCALE)


_Z80 = 1.2816
DEFAULT_FIT = MemoryFit(factor=1.0, low=round(math.exp(-_Z80 * K_PRIOR_SD), 3), high=round(math.exp(_Z80 * K_PRIOR_SD), 3),
                        growth=1.0, growth_low=round(math.exp(-_Z80 * G_PRIOR_SD), 3),
                        growth_high=round(math.exp(_Z80 * G_PRIOR_SD), 3), observations=0)


def _interval(log_grid: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    cdf = np.cumsum(weights)
    lo = log_grid[min(int(np.searchsorted(cdf, 0.10)), len(log_grid) - 1)]
    hi = log_grid[min(int(np.searchsorted(cdf, 0.90)), len(log_grid) - 1)]
    return round(math.exp(lo), 3), round(math.exp(hi), 3)


def fit(observations: list[Observation]) -> MemoryFit:
    usable = [o for o in observations if o.elapsed_days >= 1][-MAX_OBSERVATIONS:]
    if not usable:
        return DEFAULT_FIT
    a = np.array([_LN_TARGET * o.elapsed_days / max(o.stability, 1e-6) for o in usable])
    n = np.array([min(o.successes_before, MAX_SUCCESSES) for o in usable], dtype=float)
    guess = np.array([o.guess_rate for o in usable])
    correct = np.array([o.correct for o in usable], dtype=bool)

    # Shape (k, g, observation): log recall = a / (k * g**n)
    inv_scale = np.exp(-_LOG_K[:, None, None] - _LOG_G[None, :, None] * n[None, None, :])
    recall = np.exp(a[None, None, :] * inv_scale)
    p = np.clip(recall + (1 - recall) * guess, 1e-6, 1 - 1e-6)
    loglik = np.where(correct, np.log(p), np.log1p(-p)).sum(axis=2)
    log_prior = (-(_LOG_K[:, None] ** 2) / (2 * K_PRIOR_SD ** 2)
                 - (_LOG_G[None, :] ** 2) / (2 * G_PRIOR_SD ** 2))
    log_post = loglik + log_prior
    post = np.exp(log_post - log_post.max())
    post /= post.sum()

    k_marginal, g_marginal = post.sum(axis=1), post.sum(axis=0)
    k_low, k_high = _interval(_LOG_K, k_marginal)
    g_low, g_high = _interval(_LOG_G, g_marginal)
    return MemoryFit(factor=round(math.exp(float(k_marginal @ _LOG_K)), 3), low=k_low, high=k_high,
                     growth=round(math.exp(float(g_marginal @ _LOG_G)), 3), growth_low=g_low, growth_high=g_high,
                     observations=len(usable))
