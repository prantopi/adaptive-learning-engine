<h1 align="center">Adaptive Learning Engine</h1>

<p align="center"><strong>A lightweight, model-driven engine for personalised learning.</strong><br>
Knows what you know, predicts when you'll forget it, and picks what to study next.</p>

<p align="center">
  <a href="https://github.com/prantopi/adaptive-learning-engine/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/prantopi/adaptive-learning-engine/tests.yml?branch=main&style=flat-square&label=tests" alt="Tests"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Bayesian-knowledge_tracing-8A63D2?style=flat-square" alt="Bayesian knowledge tracing">
  <img src="https://img.shields.io/badge/spaced-repetition-C8442A?style=flat-square" alt="Spaced repetition">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT license">
</p>

It combines Bayesian Knowledge Tracing, spaced-repetition modelling, learner-specific forgetting curves, item difficulty estimation, response-time signals, knowledge transfer, study-budget optimisation and dropout-risk modelling to decide what a learner should study next.

```
Learner answers            right or wrong, response time, which wrong option was picked
      ↓
Knowledge state            bkt.py, hesitation.py, related.py
      ↓
Memory prediction          srs.py, personalize.py (this learner's k and g)
      ↓
Difficulty + ability       difficulty.py
      ↓
Daily load                 dropout.py (a lighter day when quitting looks likely)
      ↓
Time constraints           budget.py (most recall per second)
      ↓
NEXT BEST STUDY ACTION     recommend.py
```

`recommend()` runs the whole pipeline. You give it what you've stored about a learner and their items, and it returns the reviews to do (most valuable first), the new items to start, and the single next action, each with a reason:

```python
from adaptive_learning_engine.recommend import Item, Learner, recommend

plan = recommend(items, Learner(reviews=past_reviews, history=study_days), today, minutes=10)
plan.next      # from examples/quickstart.py: Action(kind="review", item_id=2,
               #   reason="predicted recall 90%; reviewing now adds 19 points of recall a week from now", ...)
```

Each stage is also its own module of plain functions with its own tests, so you can use any of them alone. The only dependency is numpy.

## Install

```bash
pip install adaptive-learning-engine     # once 0.1.0 is released on PyPI
pip install git+https://github.com/prantopi/adaptive-learning-engine   # works now
```

```python
import adaptive_learning_engine
```

To work on it:

```bash
uv sync
uv run pytest                       # 75 tests
uv run python examples/quickstart.py
uv run python sim/simulate.py       # about 5 minutes
```

## The models

| Module | What it does |
|---|---|
| `bkt.py` | **Bayesian Knowledge Tracing.** After each answer, Bayes' rule updates P(learned), allowing for lucky guesses and careless slips, then adds the chance the item was learned during that attempt. |
| `srs.py` | **Spaced repetition** on a forgetting curve: recall after `t` days is `0.9^(t/S)`. A right answer after a gap grows stability `S`, and a miss shrinks it. |
| `personalize.py` | **Personal memory model.** Fits `k` (how fast this learner forgets) and `g` (how much each review strengthens memory) from their own reviews, using a Bayesian 2-D grid with priors centred on the default model. Next review is in `k · g^n · S` days. |
| `hesitation.py` | **Response time as evidence.** A right answer at 1.5–4× the learner's own usual time counts as weaker evidence and grows stability less, like Anki's "hard" grade. |
| `difficulty.py` | **Item difficulty and learner ability**, Elo / one-parameter IRT style: `P(correct) = guess + (1−guess)·σ(ability − difficulty + ease)`. Hard items start with lower confidence and come back sooner. |
| `related.py` | **Related items** by tf-idf cosine similarity. Items you already know raise the starting confidence of similar new ones (knowledge transfer). |
| `budget.py` | **Short on time?** Scores each due review by how much it raises recall a week from now, per second it takes, and fills a time budget greedily (knapsack). |
| `dropout.py` | **Quit risk.** Logistic regression on study gaps, recent activity and accuracy trend, refitted as a Bayesian (MAP) model with a prior on hand-set starting weights. |
| `recommend.py` | **The pipeline.** Fits the memory model, checks quit risk to set the day's load, finds due reviews, ranks them by recall gained per second within the time budget, and adds new items with starting confidence from difficulty and related known items. Reviews come first. |
| `scheduler.py`, `grading.py` | Daily new-item allowance from test scores; harder question types as confidence rises; typed answers that accept small typos. |

`examples/quickstart.py` uses the engine on world capitals instead of language, to show the core doesn't depend on the subject.

## Does it work? A simulation

`sim/simulate.py` creates 300 simulated learners. Each one has a hidden forgetting curve per item and their own memory strength. They learn 10 items a day for 30 days, and each review schedule is scored on day 60.

| Schedule | Recall at day 60 | Reviews |
|---|---|---|
| **Personalised (`k` and `g`)** | **91.7%** | **2,380** |
| Personalised (`k` only) | 85.3% | 2,052 |
| One shared model for everyone (control) | 93.5% | 2,663 |
| Default model, not personalised | 70.1% | 1,698 |
| Every day | 100.0% | 13,350 |
| Every 3 days | 97.4% | 4,350 |
| Every 7 days | 68.5% | 1,780 |
| Ladder 1-3-7-14-30 | 54.1% | 1,250 |

**What personalising does is spend reviews where they're needed.** It keeps every quarter of learners, from weakest to strongest, near the 90% recall target (90.7–92.1%). One shared model gives strong learners 96% recall, but with 30% more reviews than they need. Overall, personalising uses 11% fewer reviews than the shared model, for 1.8 points less recall.

**The model's predictions also match the simulated reality more closely.** It predicted 4% more correct reviews than actually happened. The `k`-only version was 9% too optimistic, and the default model 23%.

**Limits:**

- **`k` is genuinely personal:** fitted `k` correlates 0.87 with each learner's true memory strength.
- **`g` mostly isn't:** its correlation with each learner's true growth is only 0.19. It mainly corrects the default growth rate for everyone.
- **I wrote the simulator's memory model,** so these results are a sanity check, not proof. In it, the default model is far too optimistic (median fitted `k` is 0.29), and real learners may differ.
- **Only the memory model is simulated.** Response time, difficulty, related items, time budgets and quit risk are tested for correct behaviour, but showing they improve learning needs real users.
- **Difficulty and quit risk learn across users,** so with very few users they stay close to their starting values.
- **Related-item links compare the surface of words.** In Spanish, *haces* ("you do") and *hace* ("it makes", used for weather) end up linked.

## Using it

Everything is plain functions on numbers, so you keep the state wherever you like:

```python
from datetime import date
from adaptive_learning_engine import bkt, srs, personalize

params = bkt.params_for("Beginner")
p = bkt.update(params.p_init, correct=True, params=params, p_guess=0.25)   # 4-option multiple choice

stability = srs.grow(1.0, p, days_since_last=3)
fit = personalize.fit(observations)            # this learner's past reviews
next_day = srs.next_review(date.today(), fit.scale(successes) * stability)
```

## License

MIT
