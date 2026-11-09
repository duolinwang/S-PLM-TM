# S-PLM-TM

Structure-aware protein sequence embeddings for structure and function retrieval.
Sequence in, 1024-dimensional vector out — no structure required when encoding.

- **Weights:** [`deepduoduo/S-PLM-TM`](https://huggingface.co/deepduoduo/S-PLM-TM) on Hugging Face
- **Benchmark data:** [`deepduoduo/S-PLM-TM-benchmarks`](https://huggingface.co/datasets/deepduoduo/S-PLM-TM-benchmarks) on Hugging Face
- **Paper:** *(in preparation)*
- **License:** PolyForm Noncommercial 1.0.0 — see [LICENSE](LICENSE)

## Installation

Python ≥ 3.10; a CUDA GPU is recommended.

```bash
git clone https://github.com/duolinwang/S-PLM-TM.git
cd S-PLM-TM
pip install -r requirements.txt
pip install -e .
hf download deepduoduo/S-PLM-TM --local-dir models/   # model weights (pip install "huggingface_hub[cli]")
splmtm embed --show-paths                              # check that all model files resolve
```

`torch_geometric` and `torch_scatter` must match your PyTorch/CUDA build; see the
[PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if they fail to install. The ProtT5-XL trunk (~11 GB) downloads on first use (set
`HF_HOME` to choose where). To keep weights elsewhere, set `SPLM_TM_MODELS`; see
[models/DOWNLOAD.md](models/DOWNLOAD.md) for checksums.

## Usage

| Command | What it does |
|---|---|
| `splmtm embed` | Sequences → `raw` / `tm` / `ec` embeddings (`.npz`) |
| `splmtm index build` | Encode target sequences once into a reusable library |
| `splmtm index info` | Show a library's model, mode, length and size |
| `splmtm search` | Search query sequences against a library |
| `splmtm align` | Residue-level alignment of retrieved candidates |

A typical run:

```bash
# Embeddings
splmtm embed --fasta proteins.fasta --modes raw tm ec --out emb.npz

# Structure retrieval (use --mode ec for function / EC retrieval)
splmtm index build --fasta targets.fasta --out tm_library --mode tm
splmtm search --query queries.fasta --index tm_library --topk 20 --out hits.tsv

# Align the retrieved candidates
splmtm align --hits hits.tsv --queries queries.fasta --index tm_library --out alignments.jsonl
```

Main options are listed below; run `splmtm <command> --help` for all of them. Every
command that loads the model also accepts `--device` (`cpu`, `cuda`, `cuda:1`;
default: automatic) and `--max-tokens` (per-batch residue budget, default 8192;
lower it if the GPU runs out of memory).

### `splmtm embed`

| Option | Default | Meaning |
|---|---|---|
| `--fasta` | — | Input FASTA. Alternatively, pass sequences directly with repeated `--seq` |
| `--modes` | `raw tm` | Any of `raw`, `tm`, `ec`, or `all` (see below) |
| `--out` | — | Output `.npz`: `ids`, `emb_raw` (always), and `emb_tm` / `emb_ec` for the requested modes, each `[N, 1024]` |
| `--max-length` | full length | Encode only the first N residues of each sequence |
| `--batch-size` | 8 | Maximum sequences per batch |

- `raw`: the unnormalised protein vector; use it to train your own downstream models.
- `tm`: structure-aware vector; the inner product of two vectors approximates their TM-score.
- `ec`: function-aware vector for EC / enzyme-function retrieval. It is computed from
  `tm`, so requesting `ec` also returns `emb_tm`. It is a retrieval embedding, not a
  classifier, and produces no probabilities.

### `splmtm index build`

| Option | Default | Meaning |
|---|---|---|
| `--fasta` | required | Target sequences |
| `--out` | required | Library directory to create |
| `--mode` | `tm` | `tm` for structure retrieval, `ec` for functional retrieval; build one library per mode |
| `--max-length` | full length | Encode only the first N residues; queries automatically use the same setting |
| `--metadata` | — | TSV whose first column is the target ID; its columns are copied into hit tables |
| `--pfam-json` / `--scan-pfam` | — | Target Pfam annotations from a JSON file, or computed with PfamScan; needed only for the optional Pfam prefilter |

An interrupted build resumes from where it stopped when rerun with the same inputs;
`--no-resume` starts over. `splmtm index info --index tm_library` prints a library's
settings without loading the model (`--verify` also checks file checksums).

### `splmtm search`

| Option | Default | Meaning |
|---|---|---|
| `--query` | required | Query FASTA |
| `--index` | required | Library from `index build`; its model, mode and length setting are reused |
| `--out` | required | Hit table (see below) |
| `--topk` | 10 | Hits per query |
| `--exclude-self` | off | Drop a hit whose target ID equals the query ID |
| `--prefilter` | `full` | `pfamfam` / `pfamclan`: keep only targets sharing a Pfam family / clan with the query. Needs a library built with Pfam annotations, and PfamScan for the queries |

The hit table is a TSV with columns `query`, `rank`, `target`, `score` (plus any
metadata columns), and a `.tsv.json` file recording how it was produced. In a `tm`
library the score approximates the TM-score; in an `ec` library it is a cosine
similarity. Search is exhaustive: every query is scored against every target.

### `splmtm align`

| Option | Default | Meaning |
|---|---|---|
| `--hits` | required | Hit table from `splmtm search` |
| `--queries` | required | The query FASTA used for the search |
| `--index` | required | The searched library, which stores the target sequences (or give `--targets targets.fasta`) |
| `--out` | required | Output JSONL |
| `--topk` | 10 | Candidates aligned per query |
| `--max-length` | full length | Align only the first N residues |

Each JSONL line is one query–candidate pair with its search rank and score, an
alignment score, the aligned sequences and zero-based matched residue pairs.
Alignment needs no extra repository: the global alignment kernel is vendored
from [CLAlign](https://github.com/yourh/CLAlign) (MIT) and its only extra
requirement is numba, which `requirements.txt` above already installs (if you
install the package alone, use `pip install -e ".[align]"`). An installed
CLAlign takes precedence when it is importable (e.g.
`PYTHONPATH=/path/to/CLAlign/src`); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Documentation

- [code/README.md](code/README.md) — model outputs, command details, Python API, code layout
- [code/SEARCH_API.md](code/SEARCH_API.md) — search and alignment API, library format
- [reproduce/README.md](reproduce/README.md) — reproducing published results
- [models/MODEL_CARD.md](models/MODEL_CARD.md) — model files, results and limitations

## License

**PolyForm Noncommercial 1.0.0** ([LICENSE](LICENSE)): academic, teaching, nonprofit,
public-research and government use is permitted; **commercial use is not** —
contact the authors. Modified versions must be marked as such and not presented as
official S-PLM-TM.

Third-party components keep their own terms: DEER benchmark data in
`reproduce/ec_deer_figure7/data/` ([DEER](https://github.com/WangJiuming/deer),
Apache-2.0); the Pfam family→clan table in `pfam_data/`
([Pfam/InterPro](https://www.ebi.ac.uk/interpro/), CC0 1.0); the ProtT5-XL encoder
downloaded at run time ([ProtTrans](https://github.com/agemagician/ProtTrans), MIT).

## Citation

A citation will be added once the paper is published. Until then, please cite this
repository and the Hugging Face record.
