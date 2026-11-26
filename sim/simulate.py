"""Simulated learners: does the app's adaptive schedule beat fixed review schedules?

Each simulated learner has a hidden memory for every phrase, modelled with a forgetting
curve: recall probability after t days is 0.9 ** (t / S_true). Reviewing when recall is
still likely but no longer certain strengthens memory the most ("desirable difficulty");
forgetting weakens it. Learners differ in how fast they learn.

Every policy sees the same learners and the same new phrases (10 a day for 30 days) and
is scored at day 60 on: average recall across all 300 phrases, and how many reviews it
cost. The adaptive policy calls the app's real BKT and SRS functions.

Usage:  uv run python sim/simulate.py [--learners 300] [--seed 7]
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_learning_engine import bkt, personalize, srs  # noqa: E402

DAYS = 60
NEW_PER_DAY = 10
INTRO_DAYS = 30
PARAMS = bkt.params_for("Beginner")


@dataclass
class Learner:
    initial_stability: float  # days of memory after first exposure
    growth: float             # how strongly a successful review strengthens memory


class Memory:
    """The hidden truth about one phrase for one learner."""

    def __init__(self, learner: Learner, day: int, rng: random.Random):
        self.learner = learner
        self.stability = max(0.3, rng.gauss(learner.initial_stability, 0.2))
        self.last = day

    def recall_probability(self, day: int) -> float:
        return 0.9 ** ((day - self.last) / self.stability)

    def review(self, day: int, rng: random.Random) -> bool:
        r = self.recall_probability(day)
        recalled = rng.random() < r
        if recalled:
            # Bigger boost when the memory had faded more (but was still recalled).
            self.stability *= 1 + self.learner.growth * (1.05 - r) * 2.2 + 0.15
        else:
            self.stability = max(0.3, self.stability * 0.35)
        self.last = day
        return recalled


def make_learners(n: int, rng: random.Random) -> list[Learner]:
    return [Learner(initial_stability=rng.uniform(0.6, 1.6), growth=rng.uniform(0.7, 1.5)) for _ in range(n)]


# ── Policies ──────────────────────────────────────────────────────────────────

class AdaptivePolicy:
    """The app: BKT belief + forgetting-curve stability decide the next review day."""
    name = "Adaptive (this app)"

    def __init__(self, k: float = 1.0):
        self.state: dict[int, dict] = {}
        self.k = k
        self.predicted = 0.0   # calibration: predicted vs actual recall at each review
        self.actual = 0

    def introduce(self, item: int, day: int, recalled: bool) -> None:
        p = bkt.update(PARAMS.p_init, recalled, PARAMS, p_guess=0.02)
        stability = srs.INITIAL_STABILITY if recalled else srs.lapse(srs.INITIAL_STABILITY)
        self.state[item] = {"p": p, "s": stability, "last": day, "due": day + 1, "n": 0}

    def due(self, day: int) -> list[int]:
        return [i for i, st in self.state.items() if st["due"] <= day]

    def scale(self, st: dict) -> float:
        return self.k

    def record(self, item: int, day: int, recalled: bool) -> None:
        st = self.state[item]
        self.predicted += srs.retention(self.scale(st) * st["s"], day - st["last"])
        self.actual += recalled
        st["p"] = bkt.update(st["p"], recalled, PARAMS, p_guess=0.02)
        if recalled:
            if day - st["last"] >= 1:
                st["n"] += 1
            st["s"] = srs.grow(st["s"], st["p"], day - st["last"])
            st["due"] = day + srs.interval_days(self.scale(st) * st["s"])
        else:
            st["s"] = srs.lapse(st["s"])
            st["due"] = day + 1
        st["last"] = day


class PersonalisedPolicy(AdaptivePolicy):
    """The app: per-user time scale k and growth g fitted from this learner's own reviews;
    intervals become k * g**n * S. Refitted every few days to keep the run fast."""
    name = "Adaptive + personalised (k, g)"
    REFIT_EVERY = 3
    USE_GROWTH = True

    def __init__(self):
        super().__init__()
        self.observations: list[personalize.Observation] = []
        self.fit = personalize.DEFAULT_FIT
        self.fitted_on = -1

    @property
    def k(self) -> float:  # for the summary statistics
        return self.fit.factor

    @k.setter
    def k(self, value: float) -> None:
        pass

    def scale(self, st: dict) -> float:
        return self.fit.scale(st["n"])

    def due(self, day: int) -> list[int]:
        if day - self.fitted_on >= self.REFIT_EVERY:
            self.fit = personalize.fit(self.observations)
            self.fitted_on = day
        return super().due(day)

    def record(self, item: int, day: int, recalled: bool) -> None:
        st = self.state[item]
        n = st["n"] if self.USE_GROWTH else 0  # without n, growth can't be told apart and stays at its prior
        self.observations.append(personalize.Observation(day - st["last"], st["s"], recalled, 0.02, n))
        super().record(item, day, recalled)


class ScaleOnlyPolicy(PersonalisedPolicy):
    """Ablation: personal time scale k only, no growth correction (the previous version)."""
    name = "Adaptive + personalised (k only)"
    USE_GROWTH = False

    def scale(self, st: dict) -> float:
        return self.fit.factor


class SharedFitPolicy(PersonalisedPolicy):
    """Control: one memory model for every learner (the average of the personalised fits),
    never refitted. If this did as well as personalising, the gain would be recalibration
    of the default model, not personalisation."""

    def __init__(self, k: float, g: float):
        super().__init__()
        self.fit = personalize.MemoryFit(k, k, k, g, g, g, 0)
        self.name = f"Adaptive, shared k={k:.2f}, g={g:.2f}"

    def due(self, day: int) -> list[int]:
        return AdaptivePolicy.due(self, day)


class FixedIntervalPolicy:
    def __init__(self, every: int):
        self.every = every
        self.name = f"Fixed: every {every} day{'s' if every > 1 else ''}"
        self.next: dict[int, int] = {}

    def introduce(self, item: int, day: int, recalled: bool) -> None:
        self.next[item] = day + self.every

    def due(self, day: int) -> list[int]:
        return [i for i, d in self.next.items() if d <= day]

    def record(self, item: int, day: int, recalled: bool) -> None:
        self.next[item] = day + self.every


class LadderPolicy:
    """Classic expanding ladder (1, 3, 7, 14, 30 days), ignoring how the learner does."""
    name = "Fixed ladder: 1-3-7-14-30"
    STEPS = [1, 3, 7, 14, 30]

    def __init__(self):
        self.step: dict[int, int] = {}
        self.next: dict[int, int] = {}

    def introduce(self, item: int, day: int, recalled: bool) -> None:
        self.step[item] = 0
        self.next[item] = day + self.STEPS[0]

    def due(self, day: int) -> list[int]:
        return [i for i, d in self.next.items() if d <= day]

    def record(self, item: int, day: int, recalled: bool) -> None:
        self.step[item] = min(self.step[item] + 1, len(self.STEPS) - 1)
        self.next[item] = day + self.STEPS[self.step[item]]


# ── Run ───────────────────────────────────────────────────────────────────────

def run(policy_factory, learners: list[Learner], seed: int) -> tuple[float, float, list]:
    """Returns (mean recall at the end, mean reviews per learner, each learner's policy)."""
    recalls, reviews, finished = [], [], []
    for li, learner in enumerate(learners):
        rng = random.Random(seed * 100_003 + li)  # same randomness for every policy
        policy = policy_factory()
        memories: dict[int, Memory] = {}
        count = 0
        for day in range(DAYS):
            for item in policy.due(day):
                count += 1
                policy.record(item, day, memories[item].review(day, rng))
            if day < INTRO_DAYS:
                for k in range(NEW_PER_DAY):
                    item = day * NEW_PER_DAY + k
                    memories[item] = Memory(learner, day, rng)
                    # First exposure: shown the answer, then checked straight away.
                    policy.introduce(item, day, memories[item].review(day, rng))
        policy.final_recall = statistics.fmean(m.recall_probability(DAYS) for m in memories.values())
        policy.review_count = count
        recalls.append(policy.final_recall)
        reviews.append(count)
        finished.append(policy)
    return statistics.fmean(recalls), statistics.fmean(reviews), finished


def calibration(policies: list) -> str:
    predicted = statistics.fmean(p.predicted for p in policies)
    actual = statistics.fmean(p.actual for p in policies)
    return f"predicted {predicted:.0f} correct reviews per learner, actual {actual:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    learners = make_learners(args.learners, random.Random(args.seed))
    print(f"{args.learners} simulated learners, {INTRO_DAYS * NEW_PER_DAY} phrases, scored on day {DAYS}\n")
    print(f"{'Policy':34} {'Recall at day 60':>17} {'Reviews':>9} {'Recall per 100 reviews':>23}")

    def show(name, recall, reviews):
        print(f"{name:34} {recall:>16.1%} {reviews:>9.0f} {100 * recall / reviews:>22.2%}")

    base = run(AdaptivePolicy, learners, args.seed)
    show(AdaptivePolicy.name, *base[:2])
    pers = run(PersonalisedPolicy, learners, args.seed)
    show(PersonalisedPolicy.name, *pers[:2])
    scale_only = run(ScaleOnlyPolicy, learners, args.seed)
    show(ScaleOnlyPolicy.name, *scale_only[:2])
    shared_k = math.exp(statistics.fmean(math.log(p.fit.factor) for p in pers[2]))
    shared_g = math.exp(statistics.fmean(math.log(p.fit.growth) for p in pers[2]))
    shared = run(lambda: SharedFitPolicy(shared_k, shared_g), learners, args.seed)
    show(SharedFitPolicy(shared_k, shared_g).name, *shared[:2])
    for factory in [lambda: FixedIntervalPolicy(1), lambda: FixedIntervalPolicy(3),
                    lambda: FixedIntervalPolicy(7), LadderPolicy]:
        show(factory().name, *run(factory, learners, args.seed)[:2])

    # Does each learner's fitted k track how good their memory really is?
    true_strength = [lr.initial_stability * lr.growth for lr in learners]
    fitted = [math.log(p.k) for p in pers[2]]
    print("\nPersonalisation checks")
    print(f"  fitted k: median {statistics.median(p.k for p in pers[2]):.2f}, "
          f"range {min(p.k for p in pers[2]):.2f}-{max(p.k for p in pers[2]):.2f}")
    growths = [p.fit.growth for p in pers[2]]
    print(f"  fitted g: median {statistics.median(growths):.2f}, range {min(growths):.2f}-{max(growths):.2f}, "
          f"correlation with true growth {statistics.correlation([math.log(g) for g in growths], [math.log(lr.growth) for lr in learners]):.2f}")
    print(f"  correlation of fitted log k with true memory strength: "
          f"{statistics.correlation(fitted, [math.log(t) for t in true_strength]):.2f}")
    print(f"  calibration, default model:  {calibration(base[2])}")
    print(f"  calibration, k only:         {calibration(scale_only[2])}")
    print(f"  calibration, k and g:        {calibration(pers[2])}")

    # Who gains? Split learners into quartiles by true memory strength.
    order = sorted(range(len(learners)), key=lambda i: true_strength[i])
    print("\nBy learner quartile (true memory strength): recall / reviews")
    print(f"  {'':10} {'median k':>9} {'default':>16} {'k and g':>16} {'k only':>16} {'shared':>16}")
    for q, label in enumerate(["weakest", "2nd", "3rd", "strongest"]):
        idx = order[q * len(order) // 4:(q + 1) * len(order) // 4]

        def cell(result):
            pol = result[2]
            return (f"{statistics.fmean(pol[i].final_recall for i in idx):.1%} / "
                    f"{statistics.fmean(pol[i].review_count for i in idx):.0f}")
        print(f"  {label:10} {statistics.median(pers[2][i].k for i in idx):>9.2f} "
              f"{cell(base):>16} {cell(pers):>16} {cell(scale_only):>16} {cell(shared):>16}")

if __name__ == "__main__":
    main()
