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

## Method comparison on example data

The table below compares every method / tested model on
`tests/data/errors-example.json`, a file with 5 documents:

- `un1`, `un2`, `un3s` describe the same underlying error (failed to pull the
  `rhacs-roxctl` image, Pod creation failed). `un3s` is significantly
  shorter: the repeated `Unknown error` block appears once instead of four
  times.
- `fo1`, `fo2` are also the same issue (repo fork blocked, account blocked),
  but a different one from the `un*` group.

Ideally any method clusters `{un1, un2, un3s}` and `{fo1, fo2}` into two
separate groups and nothing else. Metrics per method/model:

- `margin` = mean within-cluster similarity − mean cross-cluster similarity
  (bigger is a clearer separation)
- `gap` = lowest within-cluster − highest cross-cluster similarity (positive
  means a threshold exists that clusters perfectly; bigger is more headroom)
- `default thr` = does the output produce the correct two clusters with the
  tool's default `--threshold` (0.3 for lexical methods, 0.6 for `st`)

| Method | Intra mean | Inter mean | Margin | Min intra | Max inter | Gap | Default thr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| setjacc | 0.8130 | 0.0408 | 0.7722 | 0.7157 | 0.0444 | 0.6713 | correct (0.3) |
| tfidf | 0.8161 | 0.0444 | 0.7717 | 0.7041 | 0.0457 | 0.6584 | correct (0.3) |
| all-mpnet-base-v2 | 0.8940 | 0.3456 | 0.5484 | 0.7779 | 0.3681 | 0.4098 | correct (0.6) |
| paraphrase-MiniLM-L6-v2 | 0.9348 | 0.4974 | 0.4374 | 0.8925 | 0.5164 | 0.3761 | correct (0.6) |
| all-MiniLM-L6-v2 | 0.8134 | 0.3160 | 0.4974 | 0.7461 | 0.3870 | 0.3591 | correct (0.6) |
| bge-base-en-v1.5 | 0.9636 | 0.6263 | 0.3373 | 0.9362 | 0.6434 | 0.2928 | wrong — merges (0.6) |
| multiset | 0.5635 | 0.0248 | 0.5387 | 0.2606 | 0.0357 | 0.2249 | correct (0.3) / wrong (0.6) |
| bge-small-en-v1.5 | 0.9625 | 0.7185 | 0.2440 | 0.9379 | 0.7430 | 0.1949 | wrong — merges (0.6) |
| gte-small | 0.9803 | 0.8545 | 0.1258 | 0.9661 | 0.8655 | 0.1006 | wrong — merges (0.6) |

Notes:

- Lexical methods (`tfidf`, `setjacc`) are best here because the two issues
  share almost no vocabulary, so cross-cluster similarity is near zero.
- Among sentence-transformer models `all-mpnet-base-v2` is the best: high
  within-cluster similarity and a comfortable margin below the 0.6 default
  threshold. `bge-*` and `gte-small` are score-inflated — everything looks
  similar — so the default threshold merges the two issues (they need a much
  higher threshold, ≥ 0.8).
- `multiset` is count-aware: it sees `un3s` (one repeated error block) as
  different from `un1`/`un2` (four blocks), fragmenting the `un*` cluster at
  `--threshold 0.6`. At the lexical default 0.3 it still groups all three
  correctly.

Reproduce e.g. with:

```sh
docs-clustering-cli --data-json tests/data/errors-example.json --method st --model sentence-transformers/all-mpnet-base-v2
```

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
