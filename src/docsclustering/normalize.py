"""Text normalization for document similarity comparisons."""

import re

# Regexes applied against raw text, in order.  Each strips a class of
# generic noise (dates, times, hashes, URLs, long numbers) so documents
# cluster on meaning rather than on incidental detail. Domain-specific
# cleanup (custom headers, app-specific identifiers, etc.) is expected to
# happen upstream, before text reaches this tool.
_ISO_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d*(?:Z|[+-]\d{2}:?\d{2})?"
)
_SYSLOG_DATETIME_RE = re.compile(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b")
_SLASHED_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
_URL_RE = re.compile(r"https?://\S+")
_HEX_RE = re.compile(r"\b[0-9a-f]{7,}\b")
_LONG_NUM_RE = re.compile(r"\b\d{5,}\b")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Reduce a raw document to a single whitespace-normalized string of keywords."""
    text = _URL_RE.sub(" ", text)
    text = _ISO_DATETIME_RE.sub(" ", text)
    text = _SYSLOG_DATETIME_RE.sub(" ", text)
    text = _SLASHED_DATE_RE.sub(" ", text)
    text = _ISO_DATE_RE.sub(" ", text)
    text = _TIME_RE.sub(" ", text)
    text = _HEX_RE.sub(" ", text)
    text = _LONG_NUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
