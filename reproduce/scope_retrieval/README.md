# SCOPe family / superfamily retrieval

**Implemented; real-model parity pending.** Uses 29,421 targets and 5,082
family / 2,065 superfamily queries. The inputs are published — see *Running from
the published inputs* below; the frozen reference vectors and metrics are not.
Queries reuse their target embeddings; self-hits remain. Top-k recall and
AUC-1FP retain the original soft negative rules. Sequence-identity bins use the
frozen NW/BLOSUM62 identity table. SCOPe's Pfam rule always keeps self and falls
back to the full library when only a high-scoring self candidate remains.

This folder is self-contained; outside it, only the release model code in
`code/splm_tm` is used.

| File | Purpose |
|---|---|
| `prepare_scope.py` | Export the frozen inputs **and references** into a data pack (internal; compute node) |
| `prepare_scope_public.py` | Build a pack from the published inputs on Hugging Face (no references) |
| `run_scope.py` | Encode, score and compare exactly with the pack's references |
| `run_scope.sh` | Hellbender A100 job: export if needed, then run |
| `retrieval_core.py` | Unchanged copy of the original `scope_benchmark/scope_retrieval_core.py` |
| `multibin.py` | Copy of the original `scope_benchmark/scope_multibin.py`; only the import of `retrieval_core` was renamed |
| `test_scope.py` | CPU checks: `python -m unittest discover -s reproduce/scope_retrieval` |

## Exporting the internal pack, with references

Export on a compute node, with the encoding settings of the original raw cache.
This path needs the read-only research tree and is the one that can check a
rerun against the published metrics exactly:

```bash
python reproduce/scope_retrieval/prepare_scope.py \
  --research-repo /path/to/original/benchmark/checkout --source-root /path/to/source/root \
  --max-length <original-length-or-0> --batch-size <original-batch> --max-tokens <original-budget> \
  --identity-tsv /path/to/frozen_identity.tsv \
  --frozen-raw-pickle /path/to/original/db_embeddings.pkl \
  --query-pfam /path/to/frozen_queries_union_clan.json \
  --target-pfam /path/to/frozen_scope_target_clan.json --out /your/data/scope
python reproduce/scope_retrieval/run_scope.py --data /your/data/scope --out /your/results/scope
```

## Running from the published inputs

The inputs are published in
[`deepduoduo/S-PLM-TM-benchmarks`](https://huggingface.co/datasets/deepduoduo/S-PLM-TM-benchmarks)
under `scope/` (CC BY 4.0): the target table, the family and superfamily query
tables, the Pfam clan annotations for both sides, and the NW/BLOSUM62
sequence-identity table used for the identity bins. Together with the Pfam clan
map in this repository's `pfam_data/`, that is everything the pipeline reads
apart from the model.

```bash
hf download deepduoduo/S-PLM-TM-benchmarks --repo-type dataset \
  --include "scope/*" --local-dir /your/benchmarks

python reproduce/scope_retrieval/prepare_scope_public.py \
  --hf-dir /your/benchmarks/scope --out /your/data/scope \
  --max-length 0 --batch-size 32 --max-tokens 8192

python reproduce/scope_retrieval/run_scope.py \
  --data /your/data/scope --out /your/results/scope --skip-reference
```

`prepare_scope_public.py` copies the published files, derives `target.fasta`
from the target table in its order, and checks the benchmark sizes (29,421 /
5,082 / 2,065), unique target identifiers, and that every query and target
carries a Pfam annotation — without which the published Pfam-clan prefilter
cannot be reproduced. It writes the same `manifest.json` with
`prefilter: pfamclan`.

Two things are not in the published data, and the pack records their absence:

- **The reference metrics** (`reference.json`). `run_scope.py` therefore has
  nothing to compare against and is run with `--skip-reference`; it still writes
  `metrics.json` with the per-query and per-bin numbers. Compare those with the
  published figures and tables.
- **The frozen reference vectors** (`reference_raw_0.npy`). These are what
  catches a wrong encoding profile, so that check is unavailable here.

The published numbers were computed from a frozen embedding cache rather than by
re-encoding, and the encoding profile of that cache is not part of the public
data. Pass the profile you intend to use; it is recorded in the pack's
`manifest.json` and in the run's `run.json`. `--max-length 0` encodes full
length. Expect close but not necessarily identical numbers.

`run_scope.sh` does the same on Hellbender; the export inputs are passed as the
environment variables listed at the top of the script.

The data pack is not in Git. The exporter reads final
`tmcalib_<model>_<checkpoint>{,_topk}.json` and `_perquery.npz` from
`tm_calib/tmcalib_results`, selecting `head_pfam`. Strict checks include the
original raw vectors, full-precision overall metrics, and each binned query's
AUC. A wrong encoding profile is rejected by the raw-vector comparison rather
than accepted on matching rounded averages. No new identity alignments or
PfamScan runs occur implicitly. Pickle is read only by the exporter, from
trusted internal files; the runner reads JSON / FASTA / TSV / NumPy only.

`run_scope.py` writes `metrics.json`, `run.json` and `comparison.json` before
reporting a failed comparison. `--skip-reference` records an unvalidated status; it is
what a pack without references is run with, and otherwise is for development. `--embeddings <previous embeddings.npz>` evaluates cached
vectors (with their provenance sidecar) and says so in the report.
