# Third-Party Notices

## CLAlign — vendored alignment kernel

`code/splm_tm/_clalign_core.py` contains source code copied from CLAlign so that
residue-level candidate alignment works from this repository alone, without a
second checkout.

Source: <https://github.com/yourh/CLAlign>, files `src/clalign/alignment.py` and
`src/clalign/metrics.py`.

Copyright (c) 2026 Ronghui You. Distributed under the MIT License, whose full
text is reproduced in the header of `code/splm_tm/_clalign_core.py`.

Copied verbatim: `ProteinSeq`, `AlignmentResult`, `get_coverage`, `align_core`
and `f1score`. No line of the retained code was altered; the modifications are
removals only, and the file's header lists them. When upstream CLAlign is
importable it takes precedence over this copy, and each run records which kernel
it used.

CLAlign in turn vendors ESM under `src/esm/` (MIT, Copyright (c) Meta Platforms,
Inc. and affiliates). None of that code is included here.

## CLAlign — benchmark data

`reproduce/malidup_malisam/data/malidup_hec.csv` and `malisam_hec.csv` come from
the same repository's `data/` directory, under the same MIT License, reproduced
at `reproduce/malidup_malisam/data/LICENSE.CLAlign`.

## DEER — benchmark data

`reproduce/ec_deer_figure7/data/` comes from the DEER release; its license is
reproduced at `reproduce/ec_deer_figure7/data/LICENSE.DEER`.
