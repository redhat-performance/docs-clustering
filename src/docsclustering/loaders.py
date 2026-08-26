"""Document loading from the data directory."""

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
    paths = sorted(data_dir.glob(FILE_TYPE))
    names = [p.name for p in paths]
    texts = [normalize(p.read_text(encoding="utf-8", errors="replace")) for p in paths]
    return names, texts
