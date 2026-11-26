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
    scheduler    Daily allowance from test scores, exercise progression
    grading      Typo-tolerant checking of typed answers
"""
