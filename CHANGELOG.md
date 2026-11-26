# Changelog

## 0.2.0

- `AdaptiveLearner`: a simple stateful API (`add_items`, `record_answer`, `recommend`, `to_dict`/`from_dict`) over all the models.
- Input validation with clear errors.
- Item ids can be `int` or `str`. `recommend.Item` takes an optional `next_review` day.
- `examples/times_tables.py`: a non-language example.
- CI on Python 3.10–3.13. A test checks the simulation is reproducible from its seed.
- `adaptive_learning_engine.__version__`.

## 0.1.0

- First release: knowledge tracing, spaced repetition, personal memory model, response time, item difficulty, related items, time budget, quit risk and `recommend()`.
