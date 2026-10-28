# SYSTEM SPECIFICATION & MASTER IMPLEMENTATION PLAN: MS MARCO SEARCH ENGINE FROM SCRATCH

## 1. AGENT ROLE AND OBJECTIVE

You are a Principal Systems & Information Retrieval Engineer. Your task is to design, implement, test, and document a complete search engine from scratch in a single Jupyter Notebook compatible with Google Colab (free CPU runtime: ~12 GB RAM, 2 CPU cores, local `/content` disk, and persistence on Google Drive).
The system must index the entire MS MARCO Passages collection (8,841,823 passages, ~3 GB uncompressed) under memory constraints, execute conjunctive (AND) and disjunctive (OR) ranked queries using TF-IDF and BM25 models, pass regression tests, and produce all experimental effectiveness and efficiency evaluations.

---

## 2. HARD PROJECT CONSTRAINTS

1. **Allowed Libraries:**
* Standard Library Python: `re`, `unicodedata`, `array`, `struct`, `mmap`, `heapq`, `tarfile`, `pickle`, `os`, `sys`, `time`, `math`, `resource`, `gc`, `collections`, `itertools`.
* External libraries: `numpy`, `cython` (`%%cython`) or `numba` (to compile critical compression/scoring loops), `PyStemmer` (or `nltk.stem` for Porter/Snowball), `ir_datasets` (ONLY to load queries and qrels), `ir_measures` (for evaluation), `scipy.stats` (for `ttest_rel`), `pandas` (ONLY for report tables, FORBIDDEN in the index), `tqdm`, `psutil`, `matplotlib`/`seaborn`.

2. **Prohibited Libraries and Practices:**
* FORBIDDEN to use IR/ranking/indexing libraries: `Lucene`, `Pyserini`, `PyTerrier`, `Whoosh`, `Tantivy`, `rank_bm25`, `bm25s`, `scikit-learn` (`TfidfVectorizer`), `scipy.sparse`.
* FORBIDDEN to use databases or key-value stores: `SQLite`, `LMDB`, `shelve`, `dbm`.
* FORBIDDEN to use general-purpose compressors (`gzip`, `zlib`, `zstd`, `lzma`, `bz2`) on index files (using `tarfile`/`gzip` is allowed only to read input `collection.tar.gz`).
* FORBIDDEN to create a separate file for each vocabulary term: all posting lists must reside in a single binary file (or at most two: docids and freqs).
* FORBIDDEN to pre-load the entire posting lists file into RAM at startup or cache decoded posting lists between queries: at query time only Lexicon, Document Table, and Collection Statistics reside in RAM; posting lists for query terms must be read from local disk (`/content`) via file seek (`pread`/`seek`+`read`) or `mmap`.
* FORBIDDEN to use multithreading/multiprocessing during query processing (it must be strictly single-threaded).

---

## 3. NOTEBOOK ARCHITECTURE AND SECTION STRUCTURE

The code must be organized into a single `.ipynb` notebook structured into exactly 7 sequential sections:

1. **Set-up:** Package installation with pinned versions (`pip install ir_datasets==... ir_measures==... PyStemmer==...`), Google Drive mount, single global configuration cell (`CONFIG`), printout of hardware/runtime specifications (Python version, CPU cores, total/available RAM).

2. **Data Acquisition:** Idempotent download of `collection.tar.gz` to `/content`, extraction/streaming read of `collection.tsv`, and loading of the 3 query sets via `ir_datasets` (`msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`, `msmarco-passage/dev/small`).

3. **Implementation (Core + Extensions):** Modules written via `%%writefile` (e.g., `text_processor.py`, `indexer.py`, `compression.py`, `query_processor.py`, `evaluation.py`) and imported into the notebook to keep the code clean, modular, and compilable with Cython/Numba.

4. **Tests (Requirement C5):** Regression test suite on a toy collection executable in a few seconds using `assert`.

5. **Indexing Runs:** Construction of the two required indices (with and without stemming/stop-words) + compressed index (E1), featuring incremental/restartable checkpointing to Google Drive, copying to `/content` for queries, and logging of timing (run generation vs merge), RAM peak (`resource.getrusage`), and disk footprint.

6. **Evaluation & Experiments:** Generation of run files (`R1`, `R2`, `R3`, `R4` + extensions `E1`, `E2`, `E3`, `E7`), verification of the Sanity Check, effectiveness/efficiency tables, paired t-tests, and plots.

7. **Final Report:** Markdown cells in English (~10 equivalent pages) covering the 7 points required by Section 8 of the specifications.

---

## 4. DETAILED COMPONENT TECHNICAL SPECIFICATIONS (STEP-BY-STEP)

### PHASE 1: Text Processing Pipeline (Requirement C1)
Implement a `TextProcessor` class/function applied identically to both documents and queries:

1. **Case Normalisation & Punctuation Removal:** Convert to lowercase (`text.lower()`) and replace all non-alphanumeric characters with spaces using a precompiled regex `re.compile(r'[^a-z0-9]+')` (or equivalent translation matching the standard `not c.isalnum()`).

2. **Tokenisation:** Split on whitespace (`.split()`).

3. **Stop-word Removal (Switchable):** Controlled by the flag `use_stopwords: bool`. Use a standard English list (e.g., NLTK English stopwords, citing the source) converted to a `frozenset` for $O(1)$ lookup.

4. **Stemming (Switchable + Memoization):** Controlled by the flag `use_stemming: bool`. Use `Stemmer.Stemmer('english')` from `PyStemmer` (Snowball/Porter).

* **CRITICAL OPTIMIZATION:** Maintain a cache dictionary `_stem_cache: dict[str, str]` inside the processor. Across 8.8M passages, unique tokens prior to stemming represent a minimal fraction of total occurrences (Heaps' Law and Zipf's Law): compute the stem only the first time a token is seen and perform direct lookup for all subsequent occurrences.

### PHASE 2: Handling "Messy" Input and Identifiers (Section 4.1)
During reading of `collection.tsv`:
* Open the UTF-8 stream with `errors="replace"`.
* For each line: check for the presence of the `\t` character. If `\t` is missing, or if the `<pid>` field before the tab cannot be converted to `int`, increment `skipped_lines_count` and skip the line.
* If the line is valid, assign a sequential integer `docid` starting from `0` ($0, 1, 2, \dots, N-1$).
* Process `<text>` with `TextProcessor`. If the resulting token list is empty (`len(tokens) == 0`), **DO NOT discard the passage**: the document still receives its `docid`, has length $\vert{}d\vert{} = 0$, increments `empty_passages_count`, and contributes to the total count $N$ and the average length $\vert{}d\vert{}_{avg}$.
* At the end of indexing, print `skipped_lines_count` and `empty_passages_count`.

### PHASE 3: Inverted Index Construction with SPIMI (Requirement C2)
Implement the **SPIMI (Single Pass In-Memory Indexing)** algorithm described in Algorithm 3.2 of the lecture notes:

1. **In-Memory Data Structures during scanning:**
* **Document Table:** Avoid Python dictionaries. Use two pre-allocated or expandable `array.array('I')` (32-bit unsigned ints): `doc_pids` (maps `docid -> pid`) and `doc_lengths` (maps `docid -> |d|`). Save them to disk as binary/numpy files (`doc_pids.npy`, `doc_lengths.npy`) upon completion.
* **Block SPIMI Buffer:** A dictionary `block_postings: dict[str, tuple[array.array, array.array]]` mapping each term `t` to two parallel `array.array('I')`: a list of `docid`s and a list of `tf`s.
* For each document `docid`, count its term frequencies using `collections.Counter(tokens)` and append `docid` and `tf` to the arrays of the corresponding term.
* Increment a `postings_in_buffer` counter. When `postings_in_buffer >= MAX_BUFFER_POSTINGS` (configurable parameter, e.g., `10_000_000` postings per block on Colab, reducible by 10x or to `50` for C5 tests), flush the block to disk.

2. **Intermediate Block Writing (Restartable):**
* Sort keys in `block_postings` lexicographically (`sorted(block_postings.keys())`).
* Write the block `run_{block_id}.bin` (and its term index) in a compact binary format: for each sorted term, write string length, term bytes, posting count $n$, followed by $n$ 4-byte `docid` integers and $n$ 4-byte `tf` integers.
* Clear `block_postings` and call `gc.collect()`.
* **Colab Fault Tolerance:** If intermediate block files already exist on disk/Drive, allow resuming from the next block without restarting from scratch.

3. **k-way Merge Phase (Algorithm 3.2) and Upper Bound Computation:**
* Open a buffered iterator across all $k$ intermediate blocks.
* Use a `min-heap` (`heapq`) initialized with `(term, block_id)` of the first term of each block.
* While the heap is not empty, pop the minimum term `t` and all subsequent blocks in the heap sharing the same term `t`.
* Since blocks span disjoint, monotonically increasing `docid` ranges, simply concatenate `docid` arrays and `tf` arrays in order of `block_id` to obtain the globally sorted posting list of `t`.
* Compute `df = len(docids)`, `cf = sum(tfs)`.
* Immediately compute the **BM25 Upper Bound ($UB_t$)** for term `t` (required for Extension E2):
$$UB_t = \max_{d \in L_t} \left[ \ln\left(1 + \frac{N - df(t) + 0.5}{df(t) + 0.5}\right) \cdot \frac{(k_1 + 1) \cdot tf(t, d)}{tf(t, d) + k_1 \left(1 - b + b \cdot \frac{\vert{}d\vert{}}{\vert{}d\vert{}_{avg}}\right)} \right]$$
storing the exact maximum over all documents in the posting list of `t`.
* Write the posting list to the final file `postings.bin` (in Core: uncompressed binary `uint32` array of all `docid`s followed by all `tf`s for that term, recording byte `offset` and `df` in the Lexicon).
* Save the **Lexicon** (mapping `term -> (df, cf, offset, byte_length, max_bm25_score)`) and **Collection Statistics** (`N`, `avg_dl`, `total_tokens`) to disk using `pickle`.

### PHASE 4: Query Processing and Scoring (Requirement C3)
Implement the public function with the exact signature:
```python
def search(query: str, k: int, mode: str, scoring: str) -> list[tuple[int, float]]:
    ...

```

1. **Query Pre-processing Semantics:**
* Pass `query` into the same `TextProcessor` function used for the active index.
* Extract unique terms (`unique_terms = list(dict.fromkeys(tokens))`) — each distinct term counts only once.
* If `unique_terms` is empty, return `[]`.
* If `mode == "and"` and even a single term from `unique_terms` is absent from the Lexicon, immediately return `[]`.
* If `mode == "or"`, ignore terms absent from the Lexicon; if all terms are absent, return `[]`.

2. **Exact Scoring Formulas (Requirement C3):**
Use `math.log` (natural logarithm) consistently throughout the entire codebase:
* **TF-IDF:**
$$\text{TFIDF}(q, d) = \sum_{t \in q \cap d} \left(1 + \ln tf(t, d)\right) \cdot \ln \frac{N}{df(t)}$$
* **BM25** (Mandatory defaults: $k_1 = 0.9$, $b = 0.4$):
$$\text{BM25}(q, d) = \sum_{t \in q \cap d} \ln\left(1 + \frac{N - df(t) + 0.5}{df(t) + 0.5}\right) \cdot \frac{(k_1 + 1) \cdot tf(t, d)}{tf(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}d\vert{}}{\vert{}d\vert{}_{avg}}\right)}$$
* **OPTIMIZATION:** Precompute the vector `norm_factor[docid] = k1 * (1.0 - b + b * (doc_lengths[docid] / avg_dl))` once when loading the index into RAM, so at query time the BM25 denominator requires only a single addition: `tf + norm_factor[docid]`.

3. **DAAT (Document-At-A-Time) Algorithm and Tie-Breaking:**
* **Top-$k$ Min-Heap and Tie-Breaking:** When scores are tied, the document with the smaller `docid` wins. To ensure that `heapq` always evicts the worst document (lowest score or, in case of a tie, largest `docid`), store tuples in the heap of the form `(score, -docid)`.
* A candidate `(cand_score, -cand_docid)` enters the heap if `len(heap) < k` or if `(cand_score, -cand_docid) > heap[0]`.
* At the end, sort heap elements by `(-score, docid)` and convert each `docid` to its corresponding `pid = int(doc_pids[docid])`, returning `list[tuple[int, float]]`.
* **Conjunctive DAAT (`mode == "and"`):** Sort posting lists of query terms by ascending length ($\vert{}L_1\vert{} \le \vert{}L_2\vert{} \le \dots \le \vert{}L_m\vert{}$). Use the shortest list $L_1$ as the driver: for each `docid` in $L_1$, advance cursors of other lists to $\ge docid$. If all lists align on `docid`, accumulate score contributions from all terms and update the min-heap.
* **Disjunctive DAAT (`mode == "or"`):** Implement Algorithm 5.4 from the lecture notes (or coordinated cursor scanning / a min-heap of cursors `(current_docid, list_idx)` across $m$ terms), accumulating the score of the current minimum `docid` across all lists aligned on it. Compile the DAAT loop in Numba or Cython to maximize speed.

---

## 5. PLAN FOR THE 4 EXTENSIONS

Implement the following 4 extensions (all evaluated with BM25 and verified to return `pid`s, ranking order, and scores identical to Core within $10^{-6}$):

### Extension E1: Index Compression (d-gaps + Variable-Byte Encoding)
* **Theory (Ch. 4, Def. 4.4, Alg. 4.1 & 4.2):** Transform the sorted sequence of docids $d_1, d_2, \dots, d_n$ into strictly positive gaps: since $d_1 \ge 0$ in 0-based indexing, use $g_1 = d_1 + 1$ (to guarantee an integer $\ge 1$, or handle $0$ explicitly in VByte) and $g_i = d_i - d_{i-1}$ for $i \ge 2$.
* **VByte Encoder/Decoder (in Cython/Numba):** Split each integer into 7-bit blocks (least-significant byte first); continuation bytes have the most significant bit (MSB) set to 1 (+128), while the terminating byte has MSB set to 0 (< 128). Compress both d-gaps and term frequencies `tf`.
* Maintain the required flag to export the index in human-readable text format (on the 100k passages development subset) or compressed binary format.
* **Experiment E1:** Compare posting lists file size (in MB/GB and bytes per posting) and latency on `dev (small)` (BM25 OR) between uncompressed and VByte-compressed indices.

### Extension E2: Skipping (nextGEQ) and Dynamic Pruning (WAND or MaxScore)
* **Skip Pointers & nextGEQ(d) (Ch. 5, Def. 5.2, Theorem 5.1, Def. 5.3):** Partition each posting list of length $n$ into blocks/skips of size $s = \lfloor\sqrt{n}\rfloor$ (or fixed 128-element blocks) storing `(last_docid_in_block, byte_offset)` to skip entire blocks during `nextGEQ(d)` without decoding them. Use `nextGEQ(d)` to accelerate AND queries.
* **Disjunctive Dynamic Pruning (MaxScore - Algorithm 5.5 or WAND - Algorithm 5.6):**
Recommended: **MaxScore (Algorithm 5.5)** because it avoids re-sorting cursors at every single step and is exceptionally fast in Python/Numba:
1. Sort query terms by ascending upper bound: $UB_{t_1} \le UB_{t_2} \le \dots \le UB_{t_m}$.
2. Compute prefix sums $S[i] = \sum_{j=1}^i UB_{t_j}$.
3. Maintain threshold $\theta$ equal to the minimum score in the min-heap (0 if $|H| < k$) and boundary index $p$ such that $S[p] \le \theta$: terms $1 \dots p$ are non-essential, terms $p+1 \dots m$ are essential.
4. Select the next candidate document $d$ strictly as the minimum `docid` among essential cursors.
5. Sum the scores of essential terms containing $d$; only if `partial_score + S[p] > theta`, inspect non-essential cursors via `nextGEQ(c_i, d)` to check if they contain $d$ and add their contribution.
6. As $\theta$ increases, update $p$ while $S[p+1] \le \theta$.
* **Experiment E2:** Verify results are strictly identical to exhaustive DAAT (safe pruning). On `dev (small)`, report latency and average number of scored documents for $k \in \{10, 100, 1000\}$ and as a function of query length (distinct terms after processing).

### Extension E3: Term-At-A-Time (TAAT) Processing
* **Theory (Ch. 5, Def. 5.4, Algorithm 5.3):** Scan one posting list at a time for query terms, accumulating partial scores in an accumulator structure $A$ (dictionary/hash map or sparse array for OR; progressive intersection of accumulators for AND).
* **Experiment E3:** Verify result identity against DAAT. On `dev (small)`, compare TAAT and DAAT in terms of latency and memory footprint (number of allocated accumulators) for $k \in \{10, 100, 1000\}$ and across query lengths, analyzing cache locality trade-offs and top-$k$ extraction overhead.

### Extension E7: Parameter Tuning of BM25
* **Theory (Ch. 8, Section 8.5):** Run a Grid Search on BM25 parameters across the required grid: $k_1 \in \{0.6, 0.9, 1.2, 1.5, 2.0\}$ and $b \in \{0.2, 0.4, 0.6, 0.75, 0.9\}$ (25 configurations).
* **Speed Optimization:** Because posting lists for TREC DL 2019 (43 queries) involve very few terms, keep only the posting lists for those 43 queries cached in memory during grid search to evaluate all 25 combinations in under 5 seconds total.
* **Experiment E7:** Optimize $(k_1^*, b^*)$ maximizing nDCG@10 on TREC DL 2019, then evaluate $(k_1^*, b^*)$ on TREC DL 2020 and `dev (small)`, comparing it against default parameters $(k_1=0.9, b=0.4)$ and plotting a 2D heatmap of nDCG@10 over the grid.

---

## 6. TESTING PROTOCOL (REQUIREMENT C5) AND SANITY CHECK

Before launching indexing on the full collection, execute the regression test suite in Section 4 of the notebook on a hand-crafted 5-10 document toy collection:
* **Test C1 (Text Processing):** Complex punctuation strings, corrupted UTF-8 characters, lines without tab or with non-numeric pid, passages consisting exclusively of stop-words (which must become length-0 documents).
* **Test C2 (Index & Merge Identity):** Verify that `df`, `cf`, document lengths, and posting lists match manually computed values. Build the index twice: once with a large buffer (1 single intermediate run) and once with a tiny buffer (e.g., `max_postings = 3`, forcing $\ge 4$ intermediate runs), asserting that final Lexicon, Document Table, and Posting Lists are bit-for-bit identical.
* **Test C3 (Search vs Brute-Force):** Implement a `brute_force_search` function (Algorithm 1.1 from the lecture notes) computing TF-IDF and BM25 formulas from scratch across the 10 toy collection documents. Verify that `search()` in AND and OR modes, under `tfidf` and `bm25`, returns exact same `(pid, score)` pairs (within $10^{-6}$), and that extensions E1, E2, E3 produce identical output.

---

## 7. EXPERIMENTAL PROTOCOL, METRICS, AND STATISTICS (SECTION 7)

### 7.1 Mandatory Runs ($k = 1000$)
Construct two complete indices:
* **Index A:** `stemming=True`, `stopwords=True`
* **Index B:** `stemming=False`, `stopwords=False`
And execute:
* **R1:** BM25, OR, Index A $\rightarrow$ on DL 2019, DL 2020, dev (small).
* **R2:** TF-IDF, OR, Index A $\rightarrow$ on DL 2019, DL 2020, dev (small).
* **R3:** BM25, AND, Index A $\rightarrow$ on DL 2019, DL 2020, dev (small).
* **R4:** BM25, OR, Index B $\rightarrow$ on DL 2019, DL 2020.
Save run files to Google Drive in standard TREC format:
```text
<qid> Q0 <pid> <rank> <score> <run-tag>

```
(where `rank` starts at 1; queries returning 0 results have no lines in the file).

### 7.2 Correct Metric Computation with `ir_measures` (WARNING!)
* **Metrics for TREC DL 2019 and 2020:** nDCG@10, AP(rel=2), R(rel=2)@1000.
* **Metrics for MS MARCO dev (small):** RR@10 (MRR@10), R@1000.
**CRITICAL RULE FOR MEANS AND T-TESTS:** Do not use `calc_aggregate` directly because it silently drops queries returning 0 documents (very frequent in R3 AND mode)!
Build a helper function that:
1. Extracts the sorted list of all `judged_qids` present in query set qrels.
2. Initializes a dictionary `scores_per_query = {qid: 0.0 for qid in judged_qids}` for each metric.
3. Populates computed values via `ir_measures.iter_calc(measures, qrels, run)`.
4. Computes exact mean by dividing by `len(judged_qids)` and preserves the aligned array `[scores_per_query[qid] for qid in judged_qids]` to execute `scipy.stats.ttest_rel(vec_sys1, vec_sys2)` at $\alpha = 0.05$.

### 7.3 Mandatory Sanity Check
Immediately verify that **R1** achieves:
* nDCG@10 $\approx 0.50$ on TREC DL 2019
* nDCG@10 $\approx 0.48$ on TREC DL 2020
* MRR@10 (RR@10) $\approx 0.18$ on dev (small)
If values deviate by more than $\pm 0.03$, stop immediately and check query preprocessing, BM25 IDF formula, post-stopword $|d|$ computation, and `docid -> pid` mapping.

### 7.4 Efficiency Measurement Protocol
* Always copy the index from Google Drive to local disk `/content` prior to query latency benchmarking.
* Perform a complete warm-up across the entire query set for DL 2019 and DL 2020, and over the first 500 queries for dev (small).
* Time every single call to `search()` with `time.perf_counter()` (including query preprocessing and posting list disk read/decoding, excluding run file writing).
* Report in milliseconds: Mean, Median ($L_{50}$), and 95th Percentile ($L_{95}$) via `numpy.percentile`.

---

## 8. AGENT EXECUTION PROCEDURE

Proceed step-by-step:
1. Generate notebook skeleton starting from Set-up, Configuration, and Data Acquisition cells.
2. Implement core modules (`text_processor.py`, `indexer.py`, `query_processor.py`) and Test Suite C5, running tests immediately on toy collection and development subset (first 100,000 lines of `collection.tsv`) to validate correctness and speed.
3. Implement extension modules (E1, E2, E3, E7) and verify via `assert` that rankings and scores match Core identically.
4. Set up full indexing pipeline, evaluation (R1-R4 + extensions), statistical significance testing, plot generation, and authoring of the Final Report in English.