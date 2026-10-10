"""Shared visible-request cue extraction for STATE-Bench adapters."""

from __future__ import annotations

import re
from collections.abc import Sequence

REQUEST_CUE_VERSION_V1_1 = "v1.1"

DISCOURSE_MARKERS_V1_1 = (
    "hi",
    "hey",
    "hello",
    "ok",
    "okay",
    "yes",
    "no",
    "actually",
    "also",
    "and",
    "oh",
    "wait",
    "thanks",
    "perfect",
    "hmm",
)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_LEADING_NOISE_RE = re.compile(r"^[\s\"'`([{]+")
_LEADING_DISCOURSE_RE = re.compile(
    rf"^(?:{'|'.join(re.escape(marker) for marker in DISCOURSE_MARKERS_V1_1)})"
    r"\b[\s,:\-\u2013\u2014]+",
    flags=re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"[,:\-\u2013\u2014]+")


def normalize_whitespace(value: str) -> str:
    return " ".join(str(value).split())


def sentences(content: str) -> list[str]:
    rows: list[str] = []
    for match in _SENTENCE_RE.finditer(content):
        sentence = normalize_whitespace(match.group(0))
        if sentence:
            rows.append(sentence)
    return rows


def begins_with_request_cue(sentence: str, cue_terms: Sequence[str]) -> bool:
    for candidate in cue_candidate_texts(sentence):
        lowered = candidate.lower()
        for cue in cue_terms:
            if lowered == cue or lowered.startswith(f"{cue} "):
                return True
    return False


def cue_candidate_texts(sentence: str) -> list[str]:
    stripped = strip_leading_discourse_markers(_LEADING_NOISE_RE.sub("", sentence))
    candidates = [stripped]
    candidates.extend(
        strip_leading_discourse_markers(part.strip())
        for part in _CLAUSE_SPLIT_RE.split(stripped)[1:]
        if part.strip()
    )
    return [candidate for candidate in candidates if candidate]


def strip_leading_discourse_markers(sentence: str) -> str:
    stripped = sentence
    while True:
        updated = _LEADING_DISCOURSE_RE.sub("", stripped, count=1).lstrip()
        if updated == stripped:
            return stripped
        stripped = updated
