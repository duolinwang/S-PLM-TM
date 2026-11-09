# GO functional consistency

**Implemented; real-model parity and published data pack pending.** The original
1,000 queries search 7,735 disjoint Swiss-Prot targets. Retrieval uses the paired
TM head, no Pfam filtering, full NumPy query-matrix scoring, and the original
argpartition/argsort top-10 selection. This is retrieval-based GO consistency,
not a trained GO classifier.

This folder is self-contained; outside it, only the release model code in
`code/splm_tm` is used. Install `goatools` in the evaluation environment.

| File | Purpose |
|---|---|
| `prepare_go.py` | Export the frozen inputs and references into a data pack (compute node) |
| `run_go.py` | Encode, retrieve, score and compare exactly with the pack's references |
| `run_go.sh` | Hellbender A100 job: export if needed, then run |
| `legacy_score.py` | Unchanged copy of the original standalone scorer, for reference |
| `test_go.py` | CPU checks: `python -m unittest discover -s reproduce/go_functional` |

```bash
python reproduce/go_functional/prepare_go.py --source-root /path/to/splm_gvpgit --out /your/data/go
python reproduce/go_functional/run_go.py --data /your/data/go --out /your/results/go
```

or on Hellbender:

```bash
sbatch --export=ALL,RELEASE_ROOT=/your/release,SOURCE_ROOT=/path/to/splm_gvpgit,DATA_PACK=/your/data/go,OUTPUT_DIR=/your/results/go \
  /your/release/reproduce/go_functional/run_go.sh
```

The data pack is not in Git. It fixes `functional_all.fasta`, query/target lists,
both `GO_experimental.json` and `GO_paper_exact_official1587.json`, the exact
`go-basic.obo` snapshot, and frozen raw/TM vectors from `emb_<tag>.npz`. No
annotation refetch is performed. References are computed from the frozen
vectors; export also checks all top-10 target IDs against the original search TSV
and the original per-pair CSVs for both annotation views.

The encoding profile is fixed to the historical GO run: max length 512, batch
size 32 (CATH used 64), 16,384 tokens. A fresh run must reproduce the frozen raw
and TM vectors exactly, as well as every reference metric.

Wang similarity includes `part_of`, with symmetric BMA between query/target term
sets. Removed GO terms are filtered before scoring; pairs require annotations on
both proteins. Outputs include six-decimal pairs (the original CSV representation)
and full-precision per-query means. Dense SPLM search returns exactly ten hits;
the original random fallback for methods with missing hits is not invoked.

`legacy_score.py` is copied unchanged from
`splm_gvpgit/remote_homology_benchmark/remote_go/score_go_bma.py`; `run_go.py`
keeps its BMA formula.
