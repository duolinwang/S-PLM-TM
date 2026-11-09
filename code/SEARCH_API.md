# Persistent sequence search API

`Index` (`SearchIndex` is an alias) provides separate build, load and search
operations. Public API development is independent of `reproduce/`. Historical
`benchmark_tm` scoring and speed-only measurements live exclusively in the
large-scale reproduction folder. `splmtm embed` and the equivalent `embed.py`
encode full sequences by default; NPZ `search.py` retains its existing defaults
and arguments.

## Installation and CLI

Install the dependencies in `requirements.txt` for sequence encoding.
From a checkout, install the console command with:

```bash
pip install -e .
# Optional integrations:
pip install -e '.[dataframe]'

splmtm index build --fasta targets.fasta --out mylib --mode tm
splmtm index info --index mylib
splmtm search --query queries.fasta --index mylib --topk 10 --out hits.tsv
```

`PYTHONPATH=code python -m splm_tm ...` runs the same commands without installing
an entry point. `python code/scripts/search_index.py build ...` and `query ...`
remain compatibility entry points. The new command does not take historical
`--benchmark-repeats` or `--score-mode benchmark_tm` arguments.

## Python API

```python
from splm_tm import Index, T5SearchEncoder

encoder = T5SearchEncoder.load(mode="tm")
idx = Index.build("targets.fasta", "mylib", encoder=encoder)
# Omitting encoder loads the default paired T5 backbone/head for mode="tm".

idx = Index.load("mylib", encoder=encoder)  # immediately checks model identity
print(len(idx), idx.meta)
result = idx.search("queries.fasta", topk=10, chunk_size=100_000)
result = idx.search({"Q1": "ACDEFG", "Q2": "WQH"}, topk=10)
print(result.hits["Q1"])  # ordered (target_id, score, metadata) tuples
result.to_tsv("hits.tsv")
frame = result.to_dataframe()  # optional pandas
```

`search()` also accepts a sequence list with optional `ids=`. Without IDs, it
assigns stable `query0`, `query1`, ... names. The older explicit-encoder forms
`build(encoder, sequences, ids, directory)`, `build_fasta(encoder, fasta, directory)`
and `search(encoder, sequences, ids)` remain available. FASTA IDs must be unique.
No additional residue cleaning is introduced by the index layer.

`Index.load()` maps vectors read-only and checks compatibility before returning.
With `encoder=`, it checks that encoder. Otherwise it loads/checks the default T5
model; other backends require an explicit compatible encoder. To inspect a library
or search provenance-tagged vectors without model weights, explicitly use
`Index.load(path, check_model=False)`. This never disables the identity/length
checks performed by vector or sequence queries. CLI `index info` uses that
model-free path. `mmap=False` explicitly loads vectors into memory;
`verify=True` additionally streams/checks their SHA-256.

The directory contains `meta.json`, `ids.txt` and `vectors.npy`, plus a derived
`records.sqlite` lookup table (including original full target sequences) and optional `metadata.tsv`, `pfam.json`,
`pfam_clans.json`. IDs, metadata and Pfam lookups are disk-backed with a
bounded SQLite cache; loading does not allocate one Python object per target.
Auxiliary-file checksums are streamed on load. Old `index.json` / `ids.json` /
`embeddings.npy` libraries remain readable but keep their older in-memory ID and
annotation representation. Rebuild those libraries to obtain the bounded-memory
format; in-place construction resumption is only supported for the current format.
Metadata includes format
version, count/dimension, mode, length, normalization, creation time and package
version. T5 backbone/head hashes are also exposed as top-level metadata fields;
the complete model/source identity remains under `setup.encoder`. Custom encoders
without file hashes can leave those convenience fields null.

## Representation, identity and length

T5 supports `mode="raw"`, `"tm"` and `"ec"`. EC encoding applies the paired TM
head before the EC retrieval head. TM/EC searches use their existing normalized
vectors directly; no extra L2 normalization is applied by default. Raw-vector
searches require explicit `normalize=True` / `--normalize`; `metric="cosine"`
also explicitly requests normalization. Unnormalized raw retrieval is rejected.

`metric="inner"` is the default; `"cosine"` explicitly normalizes both sides, and
`"euclidean"` returns negative Euclidean distance. The legacy `score_mode` names
`inner`, `cosine` and `negative_l2` are accepted. Scores are always sorted in
descending order, with exact ties resolved by original target order.

Model compatibility checks include backbone/config/head hashes, representation
mode, head batching and encoder-source identity. A changed model requires a new
index. Build resumption also checks sequence content, IDs, annotation content and
encoding settings. Missing or modified metadata/Pfam files are rejected on load.

The `Index.build` API and CLI encode full sequences by default. Omit the length
option, or use `--max-length full` explicitly. Positive limits such as
`--max-length 512` or `1024` truncate before tokenization. In Python, use `None`
for full length; `0` remains accepted as a compatibility alias in both interfaces.
The standalone protein embedding API and `splmtm embed` (`embed.py`) also encode
full sequences by default. For embedding, use a positive `--max-length N` to
truncate or `0` to explicitly disable truncation; Python uses `max_length=None`
for full sequences. Frozen reproductions retain their case-specific settings.
An omitted query length inherits the index length; an
explicitly different query length raises before encoding. CLI `--emb emb_tm`,
`--emb emb_ec` and `--emb emb_raw` are aliases for the corresponding `--mode`;
conflicting aliases or a query mode different from the library raise an error.

## Existing vectors with provenance

Bare arrays cannot establish that their vectors belong to the index model.
Use the explicit wrapper:

```python
from splm_tm import QueryEmbeddings

queries = QueryEmbeddings(query_ids, vectors, encoder.identity, max_length=None)
queries.save("queries.npz")  # numeric vectors + Unicode IDs; JSON provenance sidecar
result = idx.search(QueryEmbeddings.load("queries.npz"), topk=10)
```

Alternatively, `idx.search(vectors, ids=query_ids, identity=encoder.identity,
max_length=None)` declares the same provenance. An absent/mismatched identity or
length is rejected. Provenance is a caller declaration; it cannot prove the origin
of externally computed or deliberately mislabeled vectors.

The CLI accepts `--query-vectors queries.npz`. Sequence searches can export this
format with `--save-query-embeddings queries.npz`; regular `splmtm embed` (`embed.py`) files
are not silently interpreted as provenance-tagged query files.

## Target blocks, precision and timing

Exhaustive search scores query blocks (default 32 rows) against target blocks
(default 100,000 rows) and merges their top-k candidates. Target normalization and
Euclidean norm calculation occur per block, without a full normalized database
copy. Score working memory is bounded by query-block-by-target-block buffers plus
top-k state and one target-vector block. The current library format also stores
IDs and annotations on disk. Exhaustive reads use bounded file buffers so scanning
does not retain all previously touched vector mmap pages in the process RSS.

Retrieval is always exhaustive (`exact=True` / `--exact` are accepted and change
nothing; `exact=False` is rejected). Every target is scored using a fixed
feature-order float32 reduction:
each product and addition rounds separately, and normalization uses the same
fixed feature order. A dense call to `splm_tm.scoring.fixed_scores` and any
query/target partition produce identical score bits and stable top-k ordering on
the same platform. Negative Euclidean distance accumulates squared coordinate
differences in that order. Equal scores use original database order. This policy
can differ in low bits from the old BLAS `q @ t.T`; the user explicitly selected
block invariance for the new API. Fixed reductions trade BLAS throughput for that
guarantee. Frozen paper reproductions and the original `search.py` retain their
original matrix operations and defaults. Results record the numerical policy.

`exclude_self=True` removes an identical query ID. `topk` larger than the library
returns all eligible targets; filtering may return fewer hits or no hits.
`score_matrix="scores.npy"` / `--save-matrix` explicitly streams a full matrix to
disk, including Pfam/self-exclusion masks as negative infinity. Matrix output
requires Q-by-N disk storage. Output blocks use ordinary file writes, avoiding an
ever-growing resident output mmap. Returning all targets by requesting a large
`topk` necessarily increases result memory; fixed-memory retrieval assumes fixed
query count, top-k, block sizes, vector dimension and per-record annotation size.

`result.timings` records encoding, optional warm-up, normalization, scoring,
filtering/transforms, ranking and output stages. Model setup and index loading are
outside online time.
Use `--timing-json` to save timings. These are wall-clock measurements of the
invocation, not claims of paper timing reproduction.

## Pfam family/clan filtering

```bash
splmtm index build --fasta targets.fasta --out mylib --mode tm \
  --pfam-json target_pfam.json --pfam-clan-tsv Pfam-A.clans.tsv
splmtm search --query queries.fasta --index mylib --out hits.tsv \
  --prefilter pfamclan --query-pfam-json query_pfam.json
```

Annotations are JSON maps from sequence IDs to family-accession lists. Accessions
lose version suffixes; clan mapping preserves unassigned families as their own
accessions. Build stores target annotations and the clan map in the index.
Python equivalents are `build(..., pfam=annotations, clan_tsv=...)` and
`search(..., prefilter="pfamclan", query_pfam=annotations)`.

Pfam masks are applied after each score block. If a query has no matching target
annotations anywhere in the library, its entire row falls back to unfiltered
retrieval. A block without candidates does not independently trigger fallback.
Self-exclusion occurs after that decision. Negative-infinity cells never become
hits. The SCOPe reproduction's special singleton-self rule is separate.

For automatic annotation, install PfamScan with `code/scripts/setup_pfam.sh`.
Build can run it explicitly with `--scan-pfam`; a filtered FASTA query without
`--query-pfam-json` runs/caches PfamScan. `--pfam-data-root` selects the installation.
Pfam changes rankings, not sequence-encoding work. It does not run by default.

## Metadata and result export

`build(..., metadata="metadata.tsv")` / `--metadata` accepts a TSV whose first
column is `id`; Python can also supply `{id: {column: value}}`. IDs must belong
to the library. Metadata fields cannot use reserved result names such as query,
target, rank or score. Partial metadata is allowed and missing values export as
empty cells. The standard four TSV columns are followed by metadata columns.
`result.hits[query_id]` returns ordered `(target_id, score, metadata)` tuples.
Those tuples additionally accept named lookups such as `hit["target"]` and
`hit["meta"]`. Positional `result.hits[0]` retains the previous dictionaries,
including rank and target index, for candidate-alignment compatibility.
DataFrame export preserves columns even when no hits are returned.

## ESM compatibility

`LegacyESMSearchEncoder.load(research_repo=..., config_path=..., ckpt_path=...)`
uses the original research encoder and dependencies, currently in raw mode.
Pass that encoder to `Index.build` / `Index.load`. CLI equivalents use
`--backend esm2_legacy --mode raw --legacy-repo ... --config ... --checkpoint ...`.
T5 and ESM heads are not interchangeable. This adapter is not a standalone ESM
model distribution, and real ESM parity remains pending.

## Search and alignment: implementation plan and status

| Task | Public entry points | Status |
|---|---|---|
| Build and search a TM or EC library | `Index.build/load/search`, `T5SearchEncoder.load` | Implemented; local API tests pass |
| Align selected hits using residue features | `CandidateAligner.load/from_search_encoder/align_candidates` | Implemented; local integration tests pass |
| Reuse target sequences without loading the original FASTA | `Index.get_sequences`, index as alignment target source | Implemented for newly built libraries |
| Search and align in separate processes | `SearchResult.to_tsv/from_tsv`, `splmtm search/align` | Implemented with provenance sidecars |
| Confirm released checkpoints and historical results | Real-model runs on the standardized node | Pending; deliberately not started |

EC retrieval applies the paired TM head followed by the EC head to both targets
and queries. Build a separate `mode="ec"` library; changing only the query head
of a TM library is rejected. Alignment always uses residue features plus its
separate head, irrespective of whether retrieval used TM or EC vectors.

## Align retrieved candidates

```python
from splm_tm import CandidateAligner, Index, T5SearchEncoder

encoder = T5SearchEncoder.load(mode="ec")  # use "tm" for structural retrieval
idx = Index.build("targets.fasta", "ec_library", encoder=encoder)
result = idx.search("queries.fasta", topk=20)
aligner = CandidateAligner.from_search_encoder(encoder, align_head_path="align_head.pt")
aligned = aligner.align_candidates(result, "queries.fasta", idx, top_k=10)
aligned.write_jsonl("alignments.jsonl")
frame = aligned.to_dataframe()  # optional pandas
```

Candidate alignment uses the separate residue branch and the compatible T5
alignment head, not pooled TM/EC vectors. The alignment kernel is vendored and
needs only numba; an installed CLAlign takes precedence. Global
alignment uses gap penalty zero; output preserves search rank/score and target
metadata, and reports zero-based matched residue pairs. It does not rerank hits.
`top_k` selects hits per query. Alignment defaults to full sequences independently
of retrieval truncation; pass `max_length=` to align prefixes. An LRU cache bounds
retained features, and only one pairwise residue matrix is constructed at a time.

Sequence sources can be ID mappings or FASTA paths. FASTA input is streamed and
only selected sequences are retained. The target source can also be the `Index`:
new construction stores original full sequences on disk, so only selected targets
are fetched. Older libraries remain searchable; for alignment, provide target
FASTA or rebuild into a fresh directory. The old library is not silently modified.

For a separate alignment process, load only the raw backbone and alignment head:

```python
from splm_tm import CandidateAligner, Index, SearchResult

result = SearchResult.from_tsv("hits.tsv")
idx = Index.load("ec_library", check_model=False)
aligner = CandidateAligner.load(align_head_path="align_head.pt")
aligner.align_candidates(result, "queries.fasta", idx, top_k=10).to_jsonl("alignments.jsonl")
```

`to_tsv()` writes a `.tsv.json` sidecar with score mode, all query IDs (including
empty rows), model/library provenance and a hit-table checksum. `from_tsv()` checks
that checksum and preserves metadata. Older tables without a score-mode sidecar
require `score_mode="inner"` (or their actual metric).

Alignment checks backbone config/checkpoint/source identity when both the aligner
and search result provide it; retrieval heads/mode may differ from the raw aligner.
If the target is an index, its setup must match the search result. Manually
constructed/legacy results can lack provenance; alignment metadata explicitly
records `backbone_checked=False` then. FASTA/mapping sequences remain caller-supplied
and are not authenticated by the hit-table checksum. Selected IDs/sequences are
validated before residue inference.

```bash
splmtm index build --fasta targets.fasta --out ec_library --mode ec
splmtm search --query queries.fasta --index ec_library --out hits.tsv --topk 20
splmtm align --hits hits.tsv --queries queries.fasta --index ec_library \
  --out alignments.jsonl --topk 10
# Older libraries or external target sequences:
splmtm align --hits hits.tsv --queries queries.fasta --targets targets.fasta \
  --out alignments.jsonl --topk 10
```

`python code/scripts/align_candidates.py` remains a compatibility entry point;
`--score-mode` is optional when present in the sidecar. Empty results produce
empty JSONL without loading weights or CLAlign. The API's serial residue encoding
and NumPy matrix path stay independent of the paired-GPU historical alignment
benchmark. Real-model parity is pending.

## Local validation

```bash
python -m unittest discover -s code/tests -v
python code/scripts/selftest_search.py
```

Tests use synthetic vectors and mock sequence encoders. They do not load released model
weights, run PfamScan, or certify historical scientific results. Benchmark
reproduction is documented separately under `reproduce/`.

`test_search_spec.py` checks exact score bits and top-k across query and target
partitions for all three metrics, immediate load-time model mismatch, documented
files/defaults, tuple hits, CLI aliases and raw normalization. The API tests also
cover empty input rejection, singleton libraries, oversized top-k, filters leaving
no hits, metadata/tampering and resume. `test_index_memory.py` creates
2,000- and 120,000-target on-disk libraries with long IDs and 1-KB annotations,
then measures peak RSS in fresh subprocesses during load and filtered exact
search. Model memory is deliberately excluded with `check_model=False`; for a
fixed loaded model it adds a constant cost independent of target count.
