"""Related phrases: knowing "Buenos días" should help with "Buenas noches".

Phrases are compared by the words they share, weighted by tf-idf so common words like
"es" or "de" count for little and distinctive ones like "buenos" count for a lot.
Similarity is the cosine between tf-idf vectors. Japanese text has no spaces, so its
romaji reading is used instead.

When you start a new phrase, its starting confidence gets a boost from related phrases
you already know well (knowledge transfer):

    boost = TRANSFER * max over related phrases of (similarity * P(L) of that phrase)
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

MIN_SIMILARITY = 0.3
TOP_LINKS = 5
TRANSFER = 0.5
MAX_BOOST = 0.4
_WORD = re.compile(r"[^\W\d_]{2,}")


def stem(word: str) -> str:
    """Very light stemming so word forms match: buenos/buenas/bueno -> buen, noches -> noch."""
    if len(word) > 4 and word.endswith("s"):
        word = word[:-1]
    if len(word) > 4 and word[-1] in "aeiou":
        word = word[:-1]
    return word


def tokens(text: str, reading: str) -> list[str]:
    source = text if " " in text.strip() or text.isascii() else reading
    folded = unicodedata.normalize("NFKD", source.casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))  # cómo -> como
    return [stem(w) for w in _WORD.findall(folded)]


def links(phrases: list[tuple[int, str, str]]) -> list[tuple[int, int, float]]:
    """(item_id, related_id, similarity) for each item's closest neighbours."""
    docs = {pid: Counter(tokens(text, reading)) for pid, text, reading in phrases}
    df = Counter(word for counts in docs.values() for word in counts)
    n = len(docs)
    vectors: dict[int, dict[str, float]] = {}
    for pid, counts in docs.items():
        vec = {w: c * math.log((1 + n) / (1 + df[w])) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors[pid] = {w: v / norm for w, v in vec.items() if v > 0}

    index: dict[str, list[int]] = defaultdict(list)
    for pid, vec in vectors.items():
        for w in vec:
            index[w].append(pid)

    out = []
    for pid, vec in vectors.items():
        scores: dict[int, float] = defaultdict(float)
        for w, v in vec.items():
            for other in index[w]:
                if other != pid:
                    scores[other] += v * vectors[other][w]
        best = sorted(((s, o) for o, s in scores.items() if s >= MIN_SIMILARITY), reverse=True)[:TOP_LINKS]
        out.extend((pid, other, round(s, 3)) for s, other in best)
    return out


def transfer_boost(known: list[tuple[float, float]]) -> float:
    """known: (similarity, P(L)) for related phrases the learner has studied."""
    if not known:
        return 0.0
    return round(min(MAX_BOOST, TRANSFER * max(sim * p for sim, p in known)), 4)


def boosted_p(p_init: float, boost: float) -> float:
    return round(p_init + (1 - p_init) * boost, 4)
