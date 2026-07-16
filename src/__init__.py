"""
hsu-ai-driven-challenge-2026 — shared utilities
"""

import re


def clean_text(text: str) -> str:
    """Normalize a raw text string.

    - Strips URLs
    - Removes non-printable / control characters
    - Collapses whitespace
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def is_garbage(text: str, min_words: int = 3, min_chars: int = 10) -> bool:
    """Return True when a text is too short / non-alphabetic to be useful."""
    if not isinstance(text, str) or len(text.strip()) < min_chars:
        return True
    if len(text.strip().split()) < min_words:
        return True
    # Must contain at least one Latin / Cyrillic / Vietnamese character
    if not re.search(r"[a-zA-Z\u00C0-\u024F\u0400-\u04FF]", text):
        return True
    return False
