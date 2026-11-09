# S-PLM-TM — code and model details

This document describes the model outputs, the `splmtm` commands in more depth, the
Python API and the code layout. For installation and the short version, see the
[top-level README](../README.md). Commands below are run from the repository root.

Every command has a full option list under `splmtm <command> --help`. The original
scripts in `code/scripts/` remain available with identical arguments and defaults:

| Command | Equivalent script |
|---|---|
| `splmtm embed` | `python code/scripts/embed.py` |
| `splmtm index build` | `python code/scripts/search_index.py build` |
| `splmtm search` | `python code/scripts/search_index.py query` |
| `splmtm align` | `python code/scripts/align_candidates.py` |

`PYTHONPATH=code python -m splm_tm ...` runs the same commands without installing.
`splm` is kept as an alias of `splmtm`. Besides `index build`, `splmtm index info`
prints the metadata of an existing index.

---

## Model outputs

A frozen ProtT5-XL encoder with adapters produces a pooled protein vector.
Two small heads refine that vector for different retrieval tasks, and a third head
operates on per-residue features for alignment.

The three protein-level outputs are **chained, not independent**:

```
sequence ──backbone──▶ emb_raw ──TM head──▶ emb_tm ──EC head──▶ emb_ec
                                   (= emb_h)
```

| Output | Shape | Normalised | Use it for |
|---|---|---|---|
| `emb_raw` | `[N, 1024]` | no | Training your own downstream models (classification, regression) |
| `emb_tm` | `[N, 1024]` | yes | Structure / remote-homology retrieval. Inner product ≈ predicted TM-score |
| `emb_ec` | `[N, 1024]` | yes | EC-number / functional retrieval |

Because the chain is serial, asking for `ec` automatically computes `tm`, and a
single forward pass through the backbone yields all three.

A separate **residue-level** branch feeds an alignment head:

```
sequence ──backbone──▶ residue features [L, 1024] ──align head──▶ [L, 1024]
```

> `emb_ec` is a **retrieval** embedding, not a classifier. It has no class
> logits and outputs no probabilities.

---

## Embedding

```bash
# Request only the modes you need; dependencies are pulled in automatically
splmtm embed --fasta proteins.fasta --modes raw --out raw.npz
splmtm embed --fasta proteins.fasta --modes all --out all.npz

# A couple of sequences, printed rather than saved
splmtm embed --seq MVLSEGEWQLVLHVWAKVEAD --seq KVFGRCELAAAMKRHGLDNYRG --print-tm
```

```python
import numpy as np
d = np.load("out.npz", allow_pickle=True)
d["ids"]        # (N,)      sequence identifiers
d["emb_raw"]    # (N, 1024)
d["emb_tm"]     # (N, 1024) also available as d["emb_h"]
d["emb_ec"]     # (N, 1024)
```

Python:

```python
from splm_tm import SPLMTM

model = SPLMTM.load(modes=("raw", "tm", "ec"))
out = model.embed(["MVLSEGEWQ...", "KVFGRCELAA..."], modes=("raw", "tm", "ec"))

sim = model.predicted_tm_matrix(out["emb_tm"])   # pairwise predicted TM-score
```

Useful options: `--modes` defaults to `raw tm`; sequences are encoded in full by
default. A positive `--max-length N` keeps only the first N residues; `0` also
disables truncation. `--batch-size` and
`--max-tokens` control memory; `--device` forces CPU or a specific GPU;
`--show-paths` prints the resolved model files.

---

## Build and search

Build a reusable library from target FASTA, then search it directly from query
FASTA. Targets are encoded once; subsequent searches load the saved library.

```bash
# Structural retrieval
splmtm index build --fasta targets.fasta --out tm_library --mode tm
splmtm search --query queries.fasta --index tm_library --topk 20 --out tm_hits.tsv

# Functional retrieval: a separate EC library
splmtm index build --fasta targets.fasta --out ec_library --mode ec
splmtm search --query queries.fasta --index ec_library --topk 20 --out ec_hits.tsv

# Library information, without loading a model
splmtm index info --index tm_library
```

Like `splmtm embed`, library construction encodes **full sequences** by default.
To encode with a residue limit, use a separate library for each setting:

```bash
splmtm index build --fasta targets.fasta --out tm_library_512 --mode tm --max-length 512
# Query encoding automatically uses the same 512-residue limit
splmtm search --query queries.fasta --index tm_library_512 --out tm_hits_512.tsv --topk 20
```

Any positive `--max-length N` keeps the first N residues of longer sequences;
shorter sequences retain their original length. `--max-length full` is also
accepted. Queries inherit the saved library setting when the option is omitted; an
explicitly different query limit is rejected. In Python, `max_length=None` means
full sequences.

EC retrieval applies the paired TM head followed by the EC head to both targets and
queries; an EC query cannot be searched against a TM library. New libraries retain
original full target sequences on disk for subsequent candidate alignment.

Python:

```python
from splm_tm import Index, T5SearchEncoder

encoder = T5SearchEncoder.load(mode="tm")  # "ec" for functional retrieval
idx = Index.build("targets.fasta", "tm_library", encoder=encoder)
# Later: idx = Index.load("tm_library", encoder=encoder)
result = idx.search("queries.fasta", topk=20)
result.to_tsv("tm_hits.tsv")
```

The hit table contains `query`, `rank`, `target`, `score` and any supplied metadata
columns. Its `.tsv.json` sidecar records provenance for subsequent alignment.
Build supports resumable encoding, and retrieval is exhaustive, with target
blocks, optional Pfam filtering and metadata. Model and length mismatches raise
errors. See [SEARCH_API.md](SEARCH_API.md) for the full
API, index format, numerical policy and ESM compatibility limits.

**Pfam prefilter.** Restricting candidates to targets that share a Pfam family or
clan is supported via `--prefilter pfamfam` / `--prefilter pfamclan`. Annotations
come from PfamScan and are applied as a post-hoc mask on the embedding scores;
queries with no Pfam hit fall back to searching the full library. See
[splm_tm/pfam.py](splm_tm/pfam.py) and [scripts/setup_pfam.sh](scripts/setup_pfam.sh).

**Existing NPZ files.** Embeddings from `splmtm embed` can be searched without a
library using the original script:

```bash
splmtm embed --fasta queries.fasta --modes raw tm --out q.npz
splmtm embed --fasta targets.fasta --modes raw tm --out t.npz
python code/scripts/search.py --query q.npz --target t.npz --topk 10 --out hits.tsv
```

| Flag | Meaning |
|---|---|
| `--emb` | Which embedding to rank on (`emb_tm` by default; use `emb_ec` for functional retrieval) |
| `--metric` | `inner` (default) or `euclidean`. For normalised embeddings the two give the same ordering |
| `--normalize` | L2-normalise before scoring. Required when ranking on `emb_raw` |
| `--topk` | Hits per query |
| `--save-matrix` | Also dump the full score matrix as `.npz` |

---

## Alignment

Align only the candidates returned by search, using the separate residue-level
alignment head. This works from this repository alone: the global alignment
kernel is vendored from [CLAlign](https://github.com/yourh/CLAlign) (MIT) into
`splm_tm/_clalign_core.py` and needs only numba, which the repository's
`requirements.txt` installs (`pip install -e ".[align]"` covers it when
installing the package alone). If upstream CLAlign is importable — for example
with `PYTHONPATH=/path/to/CLAlign/src` — it is used instead, which keeps the
exact code path behind the published numbers. The `engine` metadata field of
every run records which one ran.

```bash
splmtm align --hits tm_hits.tsv --queries queries.fasta --index tm_library \
  --out alignments.jsonl --topk 10

# An older library without stored sequences can use the original target FASTA:
splmtm align --hits tm_hits.tsv --queries queries.fasta --targets targets.fasta \
  --out alignments.jsonl --topk 10
```

Python, continuing from the search example to reuse the loaded backbone:

```python
from splm_tm import CandidateAligner

aligner = CandidateAligner.from_search_encoder(encoder)
aligned = aligner.align_candidates(result, "queries.fasta", idx, top_k=10)
aligned.to_jsonl("alignments.jsonl")
```

This also works after EC retrieval. Alignment preserves search rank and score,
encodes only selected candidates, and reports zero-based matched residue pairs.
It defaults to full sequences independently of retrieval truncation; set
`max_length=` / `--max-length` explicitly to align prefixes.

For a separate process, load `CandidateAligner.load()` and read saved hits with
`SearchResult.from_tsv("tm_hits.tsv")`. JSONL output has a metadata/timing sidecar;
`aligned.to_dataframe()` is available when pandas is installed. See the
[complete alignment API](SEARCH_API.md#align-retrieved-candidates).

Candidate alignment currently supports T5. Local integration tests use synthetic
models; end-to-end validation with the released checkpoint remains pending. The
vendored kernel was checked against upstream CLAlign on random pairs: identical
alignment strings, scores, coverage, matched residue pairs and F1.

To obtain only the residue scoring matrix through the lower-level API:

```python
from splm_tm import Backbone, heads, paths

bb = Backbone.load(paths.default_config(), paths.default_ckpt())
head, meta = heads.load_align_head(paths.default_align_head(), device=str(bb.device))

seqs = ["MVLSEGEWQ...", "KVFGRCELAA..."]
feats = bb.encode_residues(seqs, max_length=1024)                # list of [L, 1024]
z = heads.apply_align_head(head, feats, device=str(bb.device))   # L2-normalised

S = heads.residue_similarity(z[0], z[1])                         # [L1, L2]
```

`S` is the residue–residue scoring matrix that an alignment algorithm consumes.
Published results use the global Needleman–Wunsch implementation from
[CLAlign](https://github.com/yourh/CLAlign) (MIT) with zero gap penalty.

- `normalize=True` is the default and is part of the published setup — with it, the
  inner product is a cosine. Disabling it changes the scale of every score.
- Do not feed `emb_raw` or `emb_tm` to this head. Those are protein-level vectors;
  the head expects `[L, 1024]` per sequence.

See [models/align_head/README.md](../models/align_head/README.md) for the head's
architecture and for the variant used in earlier published tables.

---

## Repository layout

```
S-PLM-TM/
├── code/
│   ├── splm_tm/            embedding, search and alignment APIs; the splmtm command
│   ├── splm_backbone/      model definitions
│   ├── scripts/            script entry points equivalent to splmtm commands
│   ├── tests/              local API, integration and memory tests
│   ├── README.md           this document
│   └── SEARCH_API.md       detailed search and candidate-alignment API
├── models/                 model cards and provenance (weights downloaded separately)
├── reproduce/              reproductions of published numbers
├── pfam_data/              Pfam family→clan table, for the optional prefilter
├── examples/demo.fasta
├── README.md               installation and core commands
├── pyproject.toml          package installation and the splmtm command
└── requirements.txt
```

Each head's directory under `models/` carries its own README describing what the
head consumes and how it scored.

## Verifying the implementation

Local API and integration tests run without model weights or a GPU:

```bash
python -m unittest discover -s code/tests -v
python code/scripts/selftest_search.py
```

These tests do not certify real-model benchmark reproduction; see the per-case status in
[reproduce/README.md](../reproduce/README.md).

The heads are reimplemented in this package rather than imported from the research
code, so that the release stands alone. Parity checks (maintainers only; they need
the research repository) confirm the reimplementations are numerically identical:

```bash
python code/scripts/check_ec_head_parity.py
python code/scripts/check_residue_encode_parity.py
```

## Notes and limitations

- **Output dimension is 1024.** A 1280-dimensional vector means a different backbone
  was loaded.
- **Each head is paired with this exact backbone checkpoint.** Combining a head with
  another checkpoint can produce invalid embeddings. The index API checks library
  and query model fingerprints; lower-level head loading does not validate the
  provenance of a custom head.
- **The backbone checkpoint is a slim archive** — it omits the frozen ProtT5 trunk,
  which `transformers` supplies. A message about skipped frozen tensors while loading
  is expected.
