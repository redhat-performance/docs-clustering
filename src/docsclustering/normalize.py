"""Text normalization for document similarity comparisons."""

import re

# Regexes applied against raw log text, in order.  Each strips a class of
# noise (timestamps, threads, hashes, URLs, long numbers) so documents
# cluster on meaning rather than on incidental log detail.
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d*(?:[+-]\d{2}:?\d{2})?"
)
_KERNEL_LOG_RE = re.compile(r"[IEWF]\d{4} \d{2}:\d{2}:\d{2}\.\d+")
_URL_RE = re.compile(r"https?://\S+")
_HEX_RE = re.compile(r"\b[0-9a-f]{7,}\b")
_LONG_NUM_RE = re.compile(r"\b\d{5,}\b")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Reduce a raw document to a single whitespace-normalized string of keywords."""
    # Drop a leading log header that mentions investigate.py (tooling preamble).
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("[") and "investigate.py" in lines[0]:
        lines = lines[1:]
    text = "\n".join(lines)

    text = _URL_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    text = _KERNEL_LOG_RE.sub(" ", text)
    text = _HEX_RE.sub(" ", text)
    text = _LONG_NUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
