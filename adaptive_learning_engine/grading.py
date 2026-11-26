"""Checking typed answers with tolerance for small typos and alternative phrasings."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

SIMILARITY = 0.82
_PARENS = re.compile(r"\([^)]*\)")
_SPLIT = re.compile(r"\s*(?:/|;|\bor\b)\s*")
_LEADING = re.compile(r"^(?:to|the|a|an)\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    text = " ".join(text.split())
    return _LEADING.sub("", text)


def accepted_answers(translation: str) -> set[str]:
    """'Thank you (very much)' accepts both forms; 'Yes / Okay' accepts either."""
    forms = {translation, _PARENS.sub(" ", translation), translation.replace("(", " ").replace(")", " ")}
    variants: set[str] = set()
    for form in forms:
        variants.add(normalize(form))
        for part in _SPLIT.split(form):
            if part.strip():
                variants.add(normalize(part))
    return {v for v in variants if v}


def check_typed(answer: str, translation: str) -> bool:
    given = normalize(answer)
    if not given:
        return False
    for target in accepted_answers(translation):
        if given == target:
            return True
        if len(target) >= 4 and SequenceMatcher(None, given, target).ratio() >= SIMILARITY:
            return True
    return False
