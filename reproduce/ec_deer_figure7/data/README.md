# Third-party data: the DEER Figure-7 benchmark

The three data files in this directory are **not ours**. They come from the example
dataset released with DEER (Liu et al., *Cell Host & Microbe* 2026) under the
**Apache License 2.0** (full text in `LICENSE.DEER`).

## Files

| File | Contents |
|---|---|
| `metadata.csv` | Figure-7 ground truth. 5,849 rows; columns `uniprot_id, structure_id, ec_number, organism, is_human_template, is_bacteria_target`. 90 rows have `is_human_template=True` (the human templates) and 449 have `is_bacteria_target=True` (the true-positive bacterial isozymes) |
| `templates.fasta` | Amino-acid sequences of those 90 human templates |
| `database.fasta` | Amino-acid sequences of the 4,213 bacterial database proteins |
| `LICENSE.DEER` | Apache-2.0 text, taken from the DEER repository's `LICENSE` |

## Source

- Code: <https://github.com/WangJiuming/deer> — Apache-2.0, default branch `main`.
  The repository has **no** NOTICE file, so Apache-2.0 §4(d) does not apply.
- Data: <https://huggingface.co/datasets/cuhkaih/deer> (`data.zip`), dataset card
  declares `license: apache-2.0`, not gated. Its README states "This dataset contains
  5,849 enzyme structures and was used for benchmarking in our paper" — matching the
  5,849 rows of `metadata.csv`.
- A Google Drive mirror is also linked from the DEER repository's `readme.md`.

## Modifications (required by Apache-2.0 §4(b))

**No file contents were changed.** The only operations were copying and extraction:

1. `metadata.csv` — taken verbatim from DEER's `data.zip`
   (original path `deer/benchmark/data/example/metadata.csv`).
2. `templates.fasta` / `database.fasta` — the sequences are not at the top level of
   `data.zip`. They are the `seq.fasta` files DEER's own retrieval pipeline writes to
   its working directory, derived by DEER from the PDB structures. They were copied
   as-is and renamed. Nothing was re-exported, filtered, or reordered.

These three files are used **read-only**, as input to `../score_figure7.py`. No code
in this release writes to them.

## Citation

If you use this data, cite the DEER paper:

```bibtex
@article{liu2026exploring,
  title={Exploring functional insights into the human gut microbiome via the structural proteome},
  author={Liu, Hongbin and Shen, Juntao and Zhang, Zhiwei and Wang, Jiuming and
          Zhang, Chengxin and Zheng, Linggang and Ni, Haoran and Hong, Liang and
          Zhang, Jieqiong and Xue, Dongfang and others},
  journal={Cell Host \& Microbe},
  volume={34},
  number={1},
  pages={167--185},
  year={2026},
  publisher={Elsevier}
}
```

## The evaluation code is not DEER's

`../score_figure7.py` is our own implementation, written from the published metric
definitions in the paper's STAR Methods (reproduced in that file's docstring and in
the case README). No code was copied from the DEER repository.
