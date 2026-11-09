# Benchmark reproduction

Each case folder is self-contained: its own data (or data exporter), encoding,
scoring, reference comparison, Hellbender job script and CPU tests. Folders do
not import each other or any shared reproduction helper; outside its folder a
case only uses the release model code in `code/splm_tm` (and `code/scripts` for
DEER). Model weights follow [`models/DOWNLOAD.md`](../models/DOWNLOAD.md).

The new cases below are implemented but **real-model reproduction remains
pending**. Do not interpret passing CPU tests, or a successful cached-vector
evaluation, as a fresh inference validation.

| Case | Scope | Data | Status |
|---|---|---|---|
| [`ec_deer_figure7`](ec_deer_figure7/README.md) | DEER Figure 7 | In folder | Existing release case, unchanged |
| [`large_scale_search`](large_scale_search/README.md) | 957 SCOPe × 430,140 Swiss-Prot, full length | Published, with embeddings | CPU-only reproduction verified; GPU re-encoding pending |
| [`scope_retrieval`](scope_retrieval/README.md) | Family/superfamily and identity bins | Published inputs | Implemented; inputs on Hugging Face, references internal |
| [`cath_s20`](cath_s20/README.md) | CATH-S20 AUC and dense F1max | Exported | Implemented; data export and real parity pending |
| [`go_functional`](go_functional/README.md) | GO-MF Wang BMA | Exported | Implemented; data export and real parity pending |
| [`malidup_malisam`](malidup_malisam/README.md) | Residue alignment F1 | In folder | Implemented; real parity pending |

TED is excluded from the release and validation plan by the authors' decision.

"Exported" data is too large for Git (sequences plus frozen original vectors).
Each such case has its own `prepare_*.py`, which reads the frozen research files
from a path you give, writes a new data pack (inputs, a manifest of file names,
source paths and inference settings, and reference metrics), and refuses to
overwrite anything. Run export on a compute node. Runners take the pack with
`--data` and check that every listed file is present; they do not checksum
inputs or model weights. Local source paths are export inputs, not download
instructions.

The benchmark data are published as
[`deepduoduo/S-PLM-TM-benchmarks`](https://huggingface.co/datasets/deepduoduo/S-PLM-TM-benchmarks) under
CC BY 4.0, **prepared by Yichuan Zhang**: the `large_scale_search` inputs (queries,
target library, TM-align ground truth and the per-query scores of every method in
the paper), the SCOPe query and target sets with their sequence-identity and Pfam
annotations, and the data for `ec_deer_figure7` and `malidup_malisam`. The CATH-S20
and GO packs are not published; they are exported locally from their official
public splits.

Strict comparison is exact: reference metrics, available per-query results and,
for CATH/GO/SCOPe, the frozen original vectors. Each runner writes
`metrics.json`, `run.json` and `comparison.json` before reporting a failure.

CPU tests run per folder, e.g.
`python -m unittest discover -s reproduce/cath_s20`. Run folders separately;
each test adds its own folder to the import path.

Hellbender job scripts request the original A100 / icelake / 16-core setup and
are only run when submitted with `sbatch`. Other baseline methods are not rerun.
