# EC retrieval — DEER Figure 7

Reproduces the EC head's published numbers: S-PLM-TM + TM head + EC head on the
retrieval set released with the DEER paper.

**Self-contained** — sequences, ground truth and scoring code are all in this
directory. Nothing outside the release is required beyond its own `models/` and
`code/`.

> Only our own model's results are reproduced here. Numbers for DEER and other
> external methods are not recomputed in this directory.

## The benchmark

DEER's own Figure-7 retrieval set: **90 templates / 4,213 database proteins / 449
true positives**. It deliberately pairs bacterial and human isozymes with **below 10%
sequence identity**, making it a pure remote-homology setting.

The ground truth is DEER's, and the metrics follow the definitions in their paper's
STAR Methods (reproduced below), so **the only variable is the protein
representation** — which is what makes the comparison against DEER direct.

## Published numbers (`models/ec_head/ec_head.pt`)

| Scope | Top-10 | Top-30 | Top-100 |
|---|---|---|---|
| **Full (449 true positives)** | **0.4944** | **0.7194** | **0.8419** |
| ≤ 1024 residues (396) | 0.5101 | 0.7146 | 0.8359 |
| ≤ 512 residues (207) | 0.5217 | 0.7633 | 0.8261 |

Mean recall 0.3819; relative precision @10/30/100 = 0.5162 / 0.6576 / 0.7358.
Full file: `reference/metrics_full.json`.

## Running it

```bash
RELEASE=<absolute path to the release root>
mkdir -p ~/ec_fig7 && cd ~/ec_fig7
sbatch --export=ALL,RELEASE_ROOT=$RELEASE $RELEASE/reproduce/ec_deer_figure7/run_figure7.sh
```

Three steps:

**A — Encode.** The release's own `code/scripts/embed.py` (`--modes tm`) turns the
90 + 4,213 sequences in `data/` into `emb_h`. **No cached embeddings are reused**, so
this step also exercises the backbone and the TM head. About 90 seconds on an L40S.

**B — Score.** `score_figure7.py` applies the EC head, L2-normalises, ranks by
ascending Euclidean distance, and computes the metrics. Pure numpy; no GPU needed.

**C — Compare.** `compare_metrics.py` checks the result against
`reference/metrics_full.json`.

To reuse encodings from a previous run and skip step A:

```bash
sbatch --export=ALL,RELEASE_ROOT=$RELEASE,WORK=<previous work dir> ...
```

Scoring also runs standalone, outside SLURM — a few seconds on CPU once the npz files
exist:

```bash
python score_figure7.py --templates <tpl.npz> --database <db.npz> --out out/
python compare_metrics.py out/metrics.json reference/metrics_full.json
```

The SLURM header targets our cluster; adapt it for your scheduler, or run the two
Python steps directly.

## Two things that will trip you up

1. **`--max-length` must be 1024.** These numbers were measured with sequences
   truncated to 1024 residues, while `embed.py` encodes full length by default.
   Any other setting silently produces a different set of numbers. The run script
   hard-codes 1024.
2. **`RELEASE_ROOT` must be passed explicitly.** `sbatch` copies the script to a
   spool directory before executing it, so `${BASH_SOURCE[0]}` cannot locate the
   release.

## Metric definitions

From the STAR Methods ("Evaluation metrics") of Liu et al., *Cell Host & Microbe*
2026. For an EC family with *n* known bacterial isozymes, let `ranks` be their
1-based positions in that template's retrieval result:

| Metric | Definition |
|---|---|
| Top-k hit ratio (dataset-level) | (true positives ranked ≤ k) / 449 |
| Top-k absolute precision (per family) | (true positives ranked ≤ k) / k |
| Top-k relative precision (per family) | (true positives ranked ≤ k) / min(k, n) |
| Recall (per family) | (true positives ranked ≤ n) / n |

Ground truth comes straight from two boolean columns of `data/metadata.csv`:
`is_human_template` (90 rows) and `is_bacteria_target` (449 rows). A bacterial
protein is a true positive for a template if and only if their `ec_number` match.

Means and standard deviations are **equally weighted per family** (not per query),
and the standard deviation is the population one (divided by *n*).

## Comparison tolerance

`compare_metrics.py` applies two tiers:

- **Must match exactly:** `hit_ratio`, `hit_counts`, and all 90 values of
  `recall.per_family`. These are counts and per-family values, with no
  summation-order sensitivity.
- **Tolerance 1e-12:** the mean and standard deviation of recall and precision. These
  sum over 90 families, and a different summation order shifts the last bits of a
  float64 by 1e-16 to 1e-17. That is not a real difference.

## Verified

Run on 2026-10-14 (jobs 18929654 for encoding and 18929748 for scoring, on an L40S),
scored with the original implementation from the research codebase:

```
==== exact match required ====
  Top-10   hit_ratio 0.494432 vs 0.494432   hits 222 vs 222   OK
  Top-30   hit_ratio 0.719376 vs 0.719376   hits 323 vs 323   OK
  Top-100  hit_ratio 0.841871 vs 0.841871   hits 378 vs 378   OK
  recall.per_family  90 families  all equal  OK
==== tolerance 1e-12 ====
  max deviation 2.220e-16   OK
PASS
```

Outputs are in `verification/` (metrics for all three scopes).

`score_figure7.py` — the self-contained scorer in this directory — was verified
separately against the same embeddings: **max deviation 0.000e+00** from
`reference/metrics_full.json` (hit counts, all 90 per-family values, and every
summary statistic bit-identical), and 2.2e-16 from the original implementation, which
routes through an intermediate CSV. The two implementations are equivalent.

## Contents

| | |
|---|---|
| `run_figure7.sh` | SLURM entry point: encode → score → compare |
| `score_figure7.py` | Self-contained scoring: EC head, ranking, metrics. Pure numpy |
| `compare_metrics.py` | Comparison, usable standalone: `python compare_metrics.py got.json reference.json` |
| `data/templates.fasta` | 90 human template sequences |
| `data/database.fasta` | 4,213 bacterial database sequences |
| `data/metadata.csv` | Figure-7 ground truth (5,849 rows; 90 templates / 449 true positives) |
| `data/README.md` | Provenance, license and modification notice for the third-party data |
| `data/LICENSE.DEER` | Apache-2.0 text (DEER's license) |
| `reference/metrics_full.json` | The published numbers |
| `verification/` | Metrics from the 2026-10-14 run |

## Data provenance and license

The three files in `data/` come from **DEER's Figure-7 benchmark** (Liu et al.,
*Cell Host & Microbe* 2026) and carry the **Apache License 2.0**. Redistribution is
permitted; per Apache-2.0 §4 we include the license text, retain attribution, and
state our modifications (none to the contents — the files were only copied).

- Code: <https://github.com/WangJiuming/deer> (Apache-2.0, no NOTICE file)
- Data: <https://huggingface.co/datasets/cuhkaih/deer> (`license: apache-2.0`, not gated)
- Details, modification notice and citation: **`data/README.md`**; license text:
  **`data/LICENSE.DEER`**

Please cite the DEER paper if you use this data — BibTeX is in `data/README.md`.

> `score_figure7.py` is our own implementation, written from the published metric
> definitions. **No code was copied from the DEER repository.**
