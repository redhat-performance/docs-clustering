"""Tests for the pure logic of docs-clustering-cli (no heavy ML deps)."""

import json

import numpy as np
import pytest

from docsclustering.clustering import clusters
from docsclustering.loaders import load_docs, load_docs_json
from docsclustering.normalize import normalize
from docsclustering.similarity import sim_multiset


def test_normalize_strips_preambles_and_noise():
    # First line is an investigate.py tooling header -> dropped entirely.
    text = (
        "[2024-01-01 12:00:00] [investigate.py] keep me\n"
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


def test_load_docs_sorted_and_normalized(tmp_path):
    (tmp_path / "b.log").write_text("[2024-01-01 12:00:00] hello world")
    (tmp_path / "a.log").write_text("plain body")
    (tmp_path / "ignored.txt").write_text("not picked up")
    names, texts = load_docs(tmp_path)
    assert names == ["a.log", "b.log"]
    assert texts == [
        normalize("plain body"),
        normalize("[2024-01-01 12:00:00] hello world"),
    ]


def test_load_docs_json_sorted_and_normalized(tmp_path):
    doc = tmp_path / "docs.json"
    doc.write_text(
        json.dumps({"456": "second doc", "123": "[2024-01-01 12:00:00] first"})
    )
    names, texts = load_docs_json(doc)
    assert names == ["123", "456"]
    assert texts == [
        normalize("[2024-01-01 12:00:00] first"),
        normalize("second doc"),
    ]


def test_load_docs_json_non_dict_raises(tmp_path):
    doc = tmp_path / "docs.json"
    doc.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(TypeError):
        load_docs_json(doc)


def test_sim_multiset_counts_repetition_against_match():
    # Doc 1 is doc 0 with one extra copy of "c"; doc 2 only shares "a".
    S = sim_multiset(["a b c", "a b c c", "a x"])
    # 0 vs 1: min/max over {a,b,c} -> (1+1+1)/(1+1+2) = 0.75
    assert S[0, 1] == 0.75
    # 0 vs 2: shared "a" only -> 1/(1+1+1+1) = 0.25
    assert S[0, 2] == 0.25
    # 1 vs 2 -> 1/(1+1+2+1) = 0.2
    assert S[1, 2] == 0.2
    assert np.allclose(np.diag(S), 1.0)
