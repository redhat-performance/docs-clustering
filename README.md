# docs-clustering-cli

Group similar documents by embedding or TF-IDF cosine similarity.

## Install

Requires [uv](https://docs.astral.sh/uv/).

Install the `docs-clustering-cli` command from this repo:

```sh
# Minimal deps supporting `--method tfidf` only:
uv tool install .
# Optional heavy stack for the `--method st` embedding method:
uv tool install --extra st .
```

or directly from an upstream git remote:

```sh
uv tool install git+https://github.com/example/docs-clustering-cli
uv tool install --extra st git+https://github.com/example/docs-clustering-cli
```

For an ad-hoc run without installing:

```sh
uvx --from git+https://github.com/example/docs-clustering-cli docs-clustering-cli --help
```

## Usage

```sh
docs-clustering-cli --data-dir data --method tfidf --out similarity_matrix.csv
docs-clustering-cli --data-dir data --method st --model sentence-transformers/all-MiniLM-L6-v2
```

Prints per-file rankings, ranked similar pairs, and clusters of document picked
by similarity threshold; writes the pairwise similarity matrix as CSV.

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
make bootstrap   # one time venv setup and pre-commit installation
make check-all   # run linters
make test   # run tests
```
