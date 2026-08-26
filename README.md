# cluster-docs-cli

Group similar documents by embedding or TF-IDF cosine similarity.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .
# optional heavy stack for the default embedding method:
pip install -e ".[st]"
```

## Usage

```sh
cluster-docs-cli --data-dir data --method tfidf --out similarity_matrix.csv
cluster-docs-cli --data-dir data --method st --model sentence-transformers/all-MiniLM-L6-v2
```

Prints per-file rankings, ranked similar pairs, and clusters; writes the
pairwise similarity matrix as CSV.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `data` | Directory to scan for `*.log` files |
| `--method` | `st` | `st` (sentence-transformers) or `tfidf` |
| `--model` | all-MiniLM-L6-v2 | Sentence-transformer model name (st only) |
| `--threshold` | 0.6 (st) / 0.3 (tfidf) | Minimum similarity for clustering |
| `--top-k` | all | Limit per-file ranking rows |
| `--out` | `similarity_matrix.csv` | CSV output path |

## Development

```sh
pip install -e ".[test]"
pytest
```
