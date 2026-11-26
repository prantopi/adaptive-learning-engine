# Adaptive Learning Engine

These are the models behind a language-learning project of mine. Together they decide three things: how likely a learner is to know an item, when they'll forget it, and what they should study next. Each model is a small Python function with no database or web framework attached, and each has tests.

The only dependency is numpy.

```bash
uv sync
uv run pytest                       # 64 tests
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
from learning_engine import bkt, srs, personalize

params = bkt.params_for("Beginner")
p = bkt.update(params.p_init, correct=True, params=params, p_guess=0.25)   # 4-option multiple choice

stability = srs.grow(1.0, p, days_since_last=3)
fit = personalize.fit(observations)            # this learner's past reviews
next_day = srs.next_review(date.today(), fit.scale(successes) * stability)
```

## License

MIT
