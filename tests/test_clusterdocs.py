"""Tests for the pure logic of cluster-docs-cli (no heavy ML deps)."""

import numpy as np
import pytest

from clusterdocs.clustering import clusters
from clusterdocs.normalize import normalize


def test_normalize_strips_preambles_and_noise():
    # First line is an investigate.py tooling header -> dropped entirely.
    text = (
        '[2024-01-01 12:00:00] [investigate.py] keep me\n'
        "warning [W0000 00:00:00.000000] url https://example.com/abc hash "
        "deadbeef1234567 num 123456"
    )
    out = normalize(text)
    assert "keep me" not in out
    assert "warning" in out
    assert "https" not in out
    assert "deadbeef1234567" not in out
    assert "123456" not in out


def test_normalize_keeps_non_investigate_header():
    # Timestamps are stripped, but the line survives (no investigate.py).
    text = "[2024-01-01 12:00:00] something else\nbody text"
    # Brackets survive; only the timestamp is stripped.
    assert normalize(text) == "[ ] something else body text"


def test_clusters_connected_components():
    # Row i is a similarity score to the other docs; threshold 0.5 links
    # docs 0-1 and 1-2 into one component, leaving doc 3 isolated.
    S = np.array(
        [
            [1.0, 0.8, 0.1, 0.0],
            [0.8, 1.0, 0.9, 0.0],
            [0.1, 0.9, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    names = ["d0", "d1", "d2", "d3"]
    groups = clusters(names, S, 0.5)
    assert sorted("".join(sorted(g)) for g in groups) == ["d0d1d2", "d3"]


def test_clusters_singletons_below_threshold():
    S = np.eye(3)
    assert clusters(["a", "b", "c"], S, 0.5) == [["a"], ["b"], ["c"]]
