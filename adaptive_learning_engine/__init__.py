"""Adaptive learning engine: pure functions for deciding what a learner knows, when they'll
forget it, and what to study next.

Modules
    bkt          Bayesian Knowledge Tracing: P(item is learned) after each answer
    srs          Forgetting-curve spaced repetition: stability, retention, next review
    personalize  Per-learner memory model (time scale k, growth g), Bayesian grid fit
    hesitation   Response time as evidence: slow right answers count for less
    difficulty   Elo / 1PL-IRT ratings for item difficulty and learner ability
    related      tf-idf similarity between items and knowledge transfer to new ones
    budget       Pick the reviews worth most recall per second within a time budget
    dropout      Bayesian logistic regression for the risk a learner is about to stop
    recommend    The whole pipeline: the next best study action for a learner
    scheduler    Daily allowance from test scores, exercise progression
    grading      Typo-tolerant checking of typed answers
    learner      AdaptiveLearner: a simple stateful API over all of the above

Quick start:

    from adaptive_learning_engine import AdaptiveLearner
    learner = AdaptiveLearner()
    learner.add_items(["q1", "q2", "q3"])
    learner.record_answer("q1", correct=True, response_time=2.4)
    learner.recommend(available_time=900).next
"""
from .learner import AdaptiveLearner, AnswerResult
from .recommend import Action, Item, Learner, Plan, recommend

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("adaptive-learning-engine")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0.0.0"
__all__ = ["AdaptiveLearner", "AnswerResult", "Action", "Item", "Learner", "Plan", "recommend", "__version__"]
