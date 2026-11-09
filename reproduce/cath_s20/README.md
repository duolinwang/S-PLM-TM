# CATH-S20 retrieval

**Implemented; real-model parity and published data pack pending.** Official
CATH-S20 v4.4.0, 15,043 domains, C/A/T/H labels. Each domain queries the other
15,042 domains. No Pfam filtering is applied.

This folder is self-contained; outside it, only the release model code in
`code/splm_tm` is used.

| File | Purpose |
|---|---|
| `prepare_cath.py` | Export the frozen inputs and references into a data pack (compute node) |
| `run_cath.py` | Encode, score and compare exactly with the pack's references |
| `run_cath.sh` | Hellbender A100 job: export if needed, then run |
| `test_cath.py` | CPU checks: `python -m unittest discover -s reproduce/cath_s20` |

```bash
python reproduce/cath_s20/prepare_cath.py --source-root /path/to/splm_gvpgit --out /your/data/cath
python reproduce/cath_s20/run_cath.py --data /your/data/cath --out /your/results/cath
```

or on Hellbender:

```bash
sbatch --export=ALL,RELEASE_ROOT=/your/release,SOURCE_ROOT=/path/to/splm_gvpgit,DATA_PACK=/your/data/cath,OUTPUT_DIR=/your/results/cath \
  /your/release/reproduce/cath_s20/run_cath.sh
```

The data pack is not in Git: it contains the original FASTA, official label list,
frozen raw/TM vectors from `remote_homology_benchmark/CATH/emb_<tag>.npz` and
full-precision per-query references computed from those frozen TM vectors. Export
additionally checks the four-decimal F1max/AUC values and query counts against
the original summary files. These summaries alone do not establish exact
per-query parity. `run_cath.py` checks that every file in the pack manifest is
present; it does not checksum inputs or model weights.

The encoding profile is fixed to the published run: max length 512,
batch size 64, 16,384 tokens. The TM head is applied once to the whole raw matrix.
A fresh run must reproduce the frozen raw and TM vectors exactly, as well as
every reference metric.

Two ranking scopes are preserved: sensitivity-to-first-FP uses original
argpartition / argsort top-2000 results, while F1max uses the complete argsort
ranking. AUC integrates the descending sensitivity distribution using x=i/n.
The final T5 dense F1max implementation **excludes queries with no positives at
the relevant CATH level**, matching `compute_f1max_dense.py`; it is different from
the zero-positive convention in another historical baseline script. Matrix
blocks remain 1,024 query rows, and TM vectors are not normalized again.

`run_cath.py` writes `metrics.json`, `run.json` and `comparison.json` before
reporting a failed comparison. `--skip-reference` is for development and records
an unvalidated status. `--embeddings <previous embeddings.npz>` evaluates cached
vectors (with their provenance sidecar) and says so in the report.
