"""Small string helpers shared by retrieval and the extractive answerer."""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")
_HEADING = re.compile(r"#{1,6}\s*")


def excerpt(text: str, *, limit: int = 280) -> str:
    """Collapse whitespace and trim ``text`` to ``limit`` characters.

    Markdown heading markers are removed so a citation reads as prose.
    """
    compact = _WHITESPACE.sub(" ", _HEADING.sub("", text)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
