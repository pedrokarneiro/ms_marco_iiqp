# MS MARCO Inverted Index Query Processing

**Project:** `ms_marco_iiqp`
**Course:** Multimedia Information Retrieval & Computer Vision (MIRCV)
**University:** University of Pisa
**Academic Year:** 2026/2027

## Overview

This project implements a **search engine from scratch** for the **MS MARCO Passage Ranking Collection**.

The main objective is to build an inverted index for the complete MS MARCO passage collection and implement ranked query processing over that index without using existing search-engine or information-retrieval libraries.

The system processes approximately **8.8 million passages**, builds a disk-based inverted index using bounded memory, and supports ranked **AND** and **OR** queries using **TF–IDF** and **BM25** scoring.

The project is developed and evaluated in **Google Colab** using a free CPU runtime.

## Project Objectives

The system is designed to:

* Process the complete MS MARCO passage collection.
* Transform raw text into normalized index terms.
* Build a memory-bounded inverted index on disk.
* Store document and collection statistics required for ranking.
* Process conjunctive (**AND**) and disjunctive (**OR**) queries.
* Implement TF–IDF ranking from scratch.
* Implement BM25 ranking from scratch.
* Evaluate retrieval effectiveness using standard IR evaluation measures.
* Measure indexing and query-processing efficiency.
* Provide regression tests using a small, hand-crafted collection.
* Experiment with optional information-retrieval extensions.

## Dataset

The main collection is the **MS MARCO Passage Ranking Collection**.

It contains approximately **8.8 million passages**:

```text
8,841,823 passages
```

The collection is provided as:

```text
collection.tsv
```

with one passage per line:

```text
<pid>\t<text>
```

where:

* `pid` is the original MS MARCO passage identifier.
* `text` is the passage content.

The collection is downloaded and processed by the project's own code.

The project also uses `ir_datasets` for the official test queries and relevance judgements.

### Evaluation Query Sets

Three query sets are used:

| Query Set            | Description                                    |
| -------------------- | ---------------------------------------------- |
| TREC DL 2019         | 43 judged queries with graded relevance        |
| TREC DL 2020         | 54 judged queries with graded relevance        |
| MS MARCO dev (small) | 6,980 queries with binary relevance judgements |

The project uses the corresponding `ir_datasets` identifiers:

```python
msmarco-passage/trec-dl-2019/judged
msmarco-passage/trec-dl-2020/judged
msmarco-passage/dev/small
```

## System Architecture

The search engine is organized into several main components:

```text
                    MS MARCO Collection
                            │
                            V
                  Document Processing
                            │
                            V
                  Intermediate Postings
                            │
                            V
                 Sorting and Merging
                            │
                            V
                    Final Inverted Index
                 ┌──────────┼──────────┐
                 V          V          V
              Lexicon   Postings   Document Table
                 │          │          │
                 └──────────┼──────────┘
                            V
                     Query Processing
                            │
                 ┌──────────┴──────────┐
                 V                     V
                AND                    OR
                 │                     │
                 └──────────┬──────────┘
                            V
                      Ranking Model
                       ┌────┴────┐
                       V         V
                    TF–IDF     BM25
                       │         │
                       └────┬────┘
                            V
                       Top-k Results
```

## Text Processing

The same text-processing pipeline is applied to both passages and queries.

The core pipeline consists of:

1. Case normalization
2. Punctuation removal
3. Tokenization
4. Stop-word removal
5. Stemming

The implementation provides configuration flags allowing stemming and stop-word removal to be enabled or disabled.

The default tokenization approach is based on converting text to lowercase, replacing non-alphanumeric characters with spaces, and splitting on whitespace.

## Inverted Index

The index is built using a **memory-bounded indexing algorithm**.

The implementation creates intermediate posting runs when the in-memory buffer reaches a configurable size. These runs are then sorted and merged into the final index.

Conceptually:

```text
Documents
   │
   V
Parse + Process
   │
   V
Posting Buffer
   │
   ├── Run 1
   ├── Run 2
   ├── Run 3
   └── ...
        │
        V
      Merge
        │
        V
   Final Index
```

The final index contains:

### Lexicon

For each term, the lexicon stores:

* Document frequency (`df`)
* Collection frequency
* Posting-list offset
* Posting-list length

### Posting Lists

Each posting list contains:

* Document IDs
* Term frequencies

Posting lists are stored in increasing document-ID order.

### Document Table

The document table stores:

* Internal `docid`
* Original MS MARCO `pid`
* Document length

The project explicitly maintains the mapping between `docid` and `pid`.

### Collection Statistics

The index also stores:

* Number of documents `N`
* Average document length

## Query Processing

The main search interface is:

```python
search(query: str, k: int, mode: str, scoring: str)
    -> list[tuple[int, float]]
```

Parameters:

| Parameter | Values          | Description                          |
| --------- | --------------- | ------------------------------------ |
| `query`   | string          | User query                           |
| `k`       | integer         | Maximum number of results            |
| `mode`    | `and`, `or`     | Conjunctive or disjunctive retrieval |
| `scoring` | `tfidf`, `bm25` | Ranking function                     |

The function returns up to `k` results:

```text
(pid, score)
```

ordered by decreasing score.

Ties are resolved using increasing `docid`.

### AND Queries

A document must contain **every query term** to become a candidate.

If a query term does not exist in the lexicon, the result is empty.

### OR Queries

A document becomes a candidate when it contains **at least one query term**.

Terms that are not present in the lexicon are ignored.

## Ranking Models

### TF–IDF

The implemented TF–IDF score is:

```text
TFIDF(q,d) =
Σ [ (1 + log(tf(t,d))) × log(N / df(t)) ]
```

for query terms occurring in the document.

### BM25

The project implements BM25 with:

```text
k1 = 0.9
b  = 0.4
```

as the default parameters.

The implementation follows the formula specified in the project requirements.

## Evaluation

The required experiments use `k = 1000`.

The core runs include:

| Run | Scoring | Mode | Processing                         |
| --- | ------- | ---- | ---------------------------------- |
| R1  | BM25    | OR   | Stemming + stop-word removal       |
| R2  | TF–IDF  | OR   | Stemming + stop-word removal       |
| R3  | BM25    | AND  | Stemming + stop-word removal       |
| R4  | BM25    | OR   | No stemming + no stop-word removal |

### Effectiveness Measures

For TREC DL 2019 and 2020:

* nDCG@10
* Average Precision
* Recall@1000
* AP with relevance threshold 2
* Recall@1000 with relevance threshold 2

For MS MARCO dev (small):

* MRR@10
* Recall@1000

The project uses `ir_measures` for effectiveness evaluation.

## Efficiency Measurements

The project measures:

### Indexing

* Total indexing time
* Intermediate-run generation time
* Merge time
* Peak memory usage
* Index size

### Query Processing

For each required run:

* Mean query latency
* Median query latency
* 95th-percentile latency

Query processing is single-threaded as required by the project specification.

## Testing

A small hand-crafted test collection is used for regression testing.

Tests cover:

* Text processing
* Punctuation
* Non-ASCII characters
* Malformed input
* Lexicon construction
* Posting lists
* Document table
* Intermediate-run merging
* AND query processing
* OR query processing
* TF–IDF scoring
* BM25 scoring

Query-processing results are compared against a brute-force implementation on the toy collection.

The tests are designed to run independently of the full MS MARCO collection.

## Development Workflow

During development, a small subset of the collection is used:

```text
First 100,000 lines of collection.tsv
```

This subset is intended for debugging and development only. Its retrieval effectiveness is **not representative of the complete collection**.

The complete collection is required for the final experiments.

## Google Colab

The project is designed to run in a free Google Colab CPU runtime.

The notebook contains:

1. Setup and package installation
2. Google Drive mounting
3. Configuration
4. Data download
5. Document processing
6. Index construction
7. Query processing
8. Tests
9. Experiments
10. Evaluation
11. Results and report

Long-running outputs, particularly the final index, are stored on Google Drive so that they can survive Colab runtime disconnections.

The final index can be copied to the local `/content` filesystem before query processing to avoid the performance cost of reading posting lists directly from Google Drive.

## Allowed Technologies

The project uses Python and libraries permitted by the project specification.

Examples include:

* `ir_datasets` — test queries and relevance judgements
* `nltk` or `PyStemmer` — stemming
* `ir_measures` — evaluation measures
* `scipy.stats` — statistical significance tests
* `numpy`
* `pandas`
* `tqdm`
* `psutil`

Tokenization and normalization are implemented by the project itself.

## Restrictions

The core search engine does **not** use existing indexing or ranking implementations.

The following types of tools are not used to implement the search engine:

* Lucene
* Pyserini
* PyTerrier
* Whoosh
* Tantivy
* `rank_bm25`
* `bm25s`
* Elasticsearch
* OpenSearch
* `scikit-learn` indexing/ranking functionality
* Database or key-value stores for postings
* Sparse matrices as the index
* General-purpose compressors for index storage

The inverted index, posting lists, query processing, and ranking algorithms are implemented as part of this project.

## Optional Extensions

The project specification provides several optional extensions.

Potential extensions include:

* **E1 — Index compression**

  * d-gaps
  * Variable-byte encoding

* **E2 — Skipping and dynamic pruning**

  * Skip pointers
  * `nextGEQ`
  * MaxScore
  * WAND

* **E3 — Term-at-a-time processing**

  * Accumulator-based query processing
  * Comparison with document-at-a-time processing

* **E4 — Additional integer codes and dictionary compression**

  * Gamma
  * Golomb
  * Rice
  * Frame-of-reference
  * Front coding

* **E5 — Positional index and phrase queries**

* **E6 — Pseudo-relevance feedback**

  * RM3 or Rocchio

* **E7 — BM25 parameter tuning**

Extensions are evaluated against the core system according to the project specification.

## Repository Structure

The repository is organized approximately as follows:

```text
ms_marco_iiqp/
│
├── README.md
│
├── notebook/
│   └── ms_marco_iiqp.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── indexing.py
│   ├── merging.py
│   ├── query_processing.py
│   ├── scoring.py
│   └── evaluation.py
│
├── tests/
│   └── test_toy_collection.py
│
├── data/
│   └── README.md
│
└── .gitignore
```

> The actual repository structure may evolve during development. Large datasets, generated indexes, and intermediate indexing runs should not be committed to Git.

## Run Files

Evaluation runs follow the standard TREC format:

```text
<qid> Q0 <pid> <rank> <score> <run-tag>
```

Example:

```text
1105793 Q0 1234567 1 12.345678 R1
```

The run files contain the original MS MARCO **pids**, not the internal `docids`.

## Reproducibility

The project aims to make the complete experiment reproducible from a fresh Google Colab runtime.

After configuring the paths and parameters, the notebook should be executable from top to bottom.

Long-running operations such as indexing are designed to detect existing outputs and avoid repeating work unnecessarily.

The notebook also records runtime characteristics such as:

* Python version
* Number of CPU cores
* Available RAM

## AI-Assisted Development

AI assistants may be used during the development of this project in accordance with the course rules.

All AI-assisted usage is declared in the final report, and the project team remains responsible for understanding and explaining every line of submitted code.

## Academic Context

This repository contains the implementation for:

**Project 1 — Building a Search Engine from Scratch**

Course: **Multimedia Information Retrieval & Computer Vision**
University of Pisa
Academic Year: **2026/2027**

The project follows the requirements provided by the course specification, Version 0.3.

## Authors

- Andrea Zanin
- Biya Girma
- Pedro Carneiro Jr.
- Piotr Piotrlastname

---

## Status

🚧 **Project in development**

Current development stages include:

* [x] Repository created
* [x] Git initialized
* [x] Remote GitHub repository configured
* [ ] Environment setup
* [ ] MS MARCO data download
* [ ] Document processing
* [ ] Toy collection and regression tests
* [ ] Inverted index implementation
* [ ] Index merging
* [ ] TF–IDF ranking
* [ ] BM25 ranking
* [ ] AND query processing
* [ ] OR query processing
* [ ] Required evaluation runs
* [ ] Efficiency measurements
* [ ] Statistical comparisons
* [ ] Optional extensions
* [ ] Final Colab notebook
* [ ] Final report
