"""Lightweight Spanish/English detection for prompts and empty states.

This is a heuristic, not a language identifier. It only needs to choose
between the two languages Folio promises to handle well. The language model
is still instructed to follow the language of the question.
"""

from __future__ import annotations

import re

_ACCENTED = set("áéíóúüñ¿¡")
_SPANISH_MARKERS = frozenset(
    {
        "al",
        "como",
        "con",
        "cuál",
        "cuáles",
        "cuándo",
        "cuánto",
        "cuántas",
        "cuántos",
        "de",
        "del",
        "dónde",
        "el",
        "en",
        "está",
        "están",
        "hay",
        "la",
        "las",
        "los",
        "para",
        "por",
        "qué",
        "que",
        "una",
        "unas",
        "uno",
        "unos",
    }
)
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    """Return ``es`` or ``en``.

    Accented characters and inverted punctuation are decisive. Otherwise the
    function looks for two or more common Spanish function words.
    """
    lowered = text.lower()
    if any(character in lowered for character in _ACCENTED):
        return "es"
    tokens = set(_TOKEN.findall(lowered))
    if len(tokens & _SPANISH_MARKERS) >= 2:
        return "es"
    return "en"


def language_name(code: str) -> str:
    """Return the English name used inside model prompts."""
    return "Spanish" if code == "es" else "English"


def empty_index_message(language: str) -> str:
    """Message used when a question arrives before any document is indexed."""
    if language == "es":
        return "No hay documentos indexados. Carga al menos un archivo antes de preguntar."
    return "No documents are indexed yet. Ingest at least one file before asking a question."


def no_match_message(language: str) -> str:
    """Message used when retrieval finds nothing above the score threshold."""
    if language == "es":
        return "No encontré pasajes relevantes en los documentos cargados."
    return "I could not find relevant passages in the loaded documents."
