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
uv tool install git+https://github.com/redhat-performance/docs-clustering
uv tool install --extra st git+https://github.com/redhat-performance/docs-clustering
```

For an ad-hoc run without installing:

```sh
uvx --from git+https://github.com/redhat-performance/docs-clustering docs-clustering-cli --help
```

## Usage

```sh
docs-clustering-cli --data-dir data --method tfidf --out similarity_matrix.csv
docs-clustering-cli --data-dir data --method st --model sentence-transformers/all-MiniLM-L6-v2
docs-clustering-cli --data-json docs.json --method multiset
```

`--data-json` accepts a JSON file mapping document IDs to their text:

```json
{
  "123": "some text of first document",
  "456": "another document"
}
```

Prints per-file rankings, ranked similar pairs, and clusters of document picked
by similarity threshold; writes the pairwise similarity matrix as CSV.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `data` | Directory to scan for `*.log` files (mutually exclusive with `--data-json`) |
| `--data-json` | - | JSON file mapping document IDs to text (mutually exclusive with `--data-dir`) |
| `--method` | `tfdif` | `tfidf`, `st`, `setjacc`, or `multiset` |
| `--model` | all-MiniLM-L6-v2 | Sentence-transformer model name (st only) |
| `--threshold` | 0.6 (st) / 0.3 (tfidf) | Minimum similarity for clustering |
| `--top-k` | all | Limit per-file ranking rows |
| `--out` | `similarity_matrix.csv` | CSV output path |

## Similarity methods

- `tfidf` — **TF-IDF cosine similarity** (no ML deps). Words are weighted by
  term frequency × inverse document frequency, so corpus-common words are
  down-weighted and rare/distinctive ones up-weighted; similarity is the cosine
  between the weighted term vectors. Pure lexical.
- `st` — **Sentence-BERT (SBERT) embeddings + cosine similarity** (uses the optional
  `st` extra dependencies). Each document is embedded into a dense vector by a
  sentence-transformers model (`all-MiniLM-L6-v2` by default) and similarity is
  the cosine of the vectors. Captures semantics (synonyms, paraphrase), but only
  sees the first ~256 tokens (depends on a model) of a document.
- `setjacc` — **Jaccard similarity coefficient (binary)**. Each
  document is a bag of unique words; similarity is the sum of the per-word
  minimum counts over the sum of the maximum counts. Word counts are reduced
  to present/absent per word. Suited to documents where repetition carries
  no meaning.
- `multiset` — **Multiset Jaccard similarity (Ruzicka coefficient)**. Like
  `setjacc` but counts are weights per word. Suited to documents where
  repetition carries no meaning. Count-aware: repeating a term N times vs once
  lowers similarity, so repetition differences (e.g. 1 error row vs 4 identical
  ones) are visible.

## Preparing input data

`normalize()` only does generic cleanup: URLs, common date/time formats,
long hex strings, and runs of 5+ digits. Everything else is your job. A
cookbook for preparing documents before feeding them to this tool:

1. **Replace unique IDs with placeholders.** Component names, namespaces,
   build/run slugs, hostnames, ticket IDs → `<COMPONENT>`, `<RUN>`, etc.
   Otherwise every method clusters by "same component" instead of "same
   issue".
2. **Keep the error, drop the location.** Keep messages, reasons, status
   codes. Generalize or remove *where*/*which instance* it happened, unless
   that's what you want to group by.
3. **Don't dedupe repeated blocks if the count matters.** "Failed once" vs.
   "retried 4 times" should look different. Either keep the repeats and use
   `--method multiset` (count-aware), or encode the count as a token, e.g.
   `retries=4`.
4. **Strip your own tooling headers/preambles** (log wrappers, custom
   prefixes) before writing the JSON/log files — this tool won't recognize
   them.

## Development

```sh
make bootstrap   # one time venv setup and pre-commit installation
make check-all   # run linters
make test   # run tests
```
