# MALIDUP / MALISAM residue alignment

**Implemented; real-model parity pending.** Uses the CLAlign HEC CSV sequences
and reference alignments, the released T5 backbone and default residual align
head (hidden=2048, T=0.07, seed 1). This case does not apply the pooled TM or EC
head. The old hidden=1024 head is a separate configuration.

This folder is self-contained, including its data; outside it, only the release
model code in `code/splm_tm` is used. The alignment kernel and its F1 metric are
vendored from [CLAlign](https://github.com/yourh/CLAlign) (MIT) into
`splm_tm/_clalign_core.py`, so no extra checkout is needed — only numba. An
importable upstream CLAlign takes precedence, and `run.json` records which
kernel ran.

| File | Purpose |
|---|---|
| `data/malidup_hec.csv`, `data/malisam_hec.csv` | 241 / 130 pairs: sequences and reference alignments |
| `data/LICENSE.CLAlign` | MIT license of the CSVs |
| `reference/*_align_head.json` | Original per-pair precision / recall / F1 and means |
| `run_alignment.py` | Encode pairs, align, and compare exactly with `reference/` |
| `run_alignment.sh` | Hellbender A100 job |
| `test_alignment.py` | CPU checks: `python -m unittest discover -s reproduce/malidup_malisam` |

```bash
python reproduce/malidup_malisam/run_alignment.py --out /your/results/alignment
```

or on Hellbender:

```bash
sbatch --export=ALL,RELEASE_ROOT=/your/release,CLALIGN_ROOT=/path/to/CLAlign,OUTPUT_DIR=/your/results/alignment \
  /your/release/reproduce/malidup_malisam/run_alignment.sh
```

The CSVs are from the public CLAlign repository (`data/`, MIT). Below the header
they are byte-identical to the copies used for the original run
(`splm_gvpgit/CLAlign/data`); only the header differs (commas here, spaces
there), and the header line is skipped. The reference JSONs are byte-identical to
the original per-pair evaluation output for the released head.

The two sequences are encoded together in the original padded batch. Residue
features pass through the align head and L2 normalization, then their cosine
matrix is computed **on the GPU**. CLAlign performs global alignment, gap penalty
0, and evaluates per-pair precision, recall and F1.

The comparison requires each mean to round to the same three decimals as the
reference — those are the reported values — and each per-pair value to stay
within half a unit of the third decimal. It also records whether the run was
bit-exact and the largest absolute deviation seen, so last-digit noise is
visible without being treated as a failure: a different GPU, BLAS build or TF32
setting perturbs the cosine matrix in its last digits, which can flip a tie
inside the global alignment and move one pair's F1.

This runner preserves the historical pair-encoding/GPU matrix path. The general
candidate alignment API uses serial encoding / NumPy matrices and still needs its
own numerical comparison; a working API alone does not certify this benchmark.
`--skip-reference` is for development and records an unvalidated status.
