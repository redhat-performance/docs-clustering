"""Document loading from a data directory or a JSON dictionary."""

import json
from pathlib import Path

from docsclustering.normalize import normalize

# Glob pattern matched against the data directory to pick up documents.
FILE_TYPE = "*.log"


def load_docs(data_dir: Path):
    """Return (names, texts) for every FILE_TYPE file in data_dir, sorted by name.

    Names are filenames; texts are normalized document bodies.  Malformed
    UTF-8 bytes are replaced rather than raising, so one bad file cannot
    abort the whole run.
    """
    raw = {
        p.name: p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(data_dir.glob(FILE_TYPE))
    }
    return _normalize_docs(raw)


def load_docs_json(data_json: Path):
    """Return (names, texts) for every document in a JSON dictionary file.

    The JSON file must map document IDs (strings) to their raw text.  Names
    are the dictionary keys; texts are normalized document bodies.
    """
    with data_json.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{data_json} must contain a JSON dictionary")
    for key, text in data.items():
        if not isinstance(text, str):
            raise TypeError(f"value for {key!r} in {data_json} is not a string")
    return _normalize_docs(data)


def _normalize_docs(raw: dict):
    """Normalize a {name: text} mapping into (names, texts) sorted by name."""
    pairs = sorted((name, normalize(text)) for name, text in raw.items())
    return [n for n, _ in pairs], [t for _, t in pairs]
