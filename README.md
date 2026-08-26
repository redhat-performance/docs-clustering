# cluster-docs-cli

Group similar documents by embedding or TF-IDF cosine similarity.

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

## Install

Install the `cluster-docs-cli` command from this repo:

```sh
uv tool install .
# optional heavy stack for the default embedding method:
uv tool install --extra st .
```

or directly from an upstream git remote:

```sh
uv tool install git+https://github.com/example/cluster-docs-cli
uv tool install --extra st git+https://github.com/example/cluster-docs-cli
```

The binary lands in `~/.local/bin`. For an ad-hoc run without installing:

```sh
uvx --from . cluster-docs-cli --data-dir data --method tfidf
```

## Usage

```sh
cluster-docs-cli --data-dir data --method tfidf --out similarity_matrix.csv
cluster-docs-cli --data-dir data --method st --model sentence-transformers/all-MiniLM-L6-v2
```

Prints per-file rankings, ranked similar pairs, and clusters; writes the
pairwise similarity matrix as CSV.

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
uv sync                 # create .venv, install project + dev group, write uv.lock
uv run pytest           # run tests
uv run cluster-docs-cli --data-dir data --method tfidf
```
