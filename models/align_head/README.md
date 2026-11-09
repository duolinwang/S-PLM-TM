# Alignment head

`align_head.pt` — residue-level alignment head. 4.20M parameters, `hidden=2048`,
one residual block, trained at temperature 0.07.

## Input

This head consumes **per-residue** features, unlike the TM and EC heads which take
protein-level vectors.

```
                                           ┌─ TM head → emb_h [1024]        structure retrieval
sequence → backbone → pooled emb_raw [1024]┤
                                           └─ EC head (after emb_h)         functional retrieval

sequence → backbone → residue_feats [L,1024] → align head → [L,1024] → L2-normalise
                                                   → inner product → [L1,L2] score matrix
```

- **Do not** feed it `emb_raw` or `emb_h`. Those are single 1024-d vectors; a shape
  guard in the code will reject them.
- The entry point is `Backbone.encode_residues(seqs)`, which returns one `[L, 1024]`
  array per sequence (CLS and EOS removed).
- Sequences are encoded in full length by default. A positive `max_length`
  genuinely drops residues here.
- This head is **not** part of the `raw → tm → ec` chain. It branches off the residue
  features and is independent of the other two heads.

## Usage

```python
from splm_tm import Backbone, heads, paths

bb = Backbone.load()
res = bb.encode_residues(["MVLSEGEWQL...", "MKTAYIAK..."], max_length=None)

head, meta = heads.load_align_head(paths.default_align_head())
z = heads.apply_align_head(head, res)          # L2-normalised by default
S = heads.residue_similarity(z[0], z[1])       # [L1, L2] score matrix
```

`S` is the matrix an alignment algorithm consumes. **The alignment algorithm itself
is not bundled** — published numbers use `align_core` (global Needleman–Wunsch, zero
gap penalty) and `f1score` from [CLAlign](https://github.com/HeliXonProtein/CLAlign)
(MIT). If you only need head-transformed residue representations, stop at `z`.

> `apply_align_head` defaults to `normalize=True`, and that is part of the published
> setting: the numbers below come from head → L2-normalise along the last dimension →
> inner product. Disabling it means the inner product is no longer a cosine and the
> results are not comparable to the table.

To use the alternative head:

```bash
export SPLM_TM_ALIGN_HEAD=.../models/align_head/alternatives/align_head_published_table.pt
```

## The head

```
z = x + Linear(GELU(Linear(x)))
```

A single residual MLP block with a **zero-initialised last layer**, so an untrained
head is exactly the identity (the release includes a test asserting
`max|fresh(x) − x| = 0.0`). The "no head" row below is therefore this head's starting
point, not a different model.

Checkpoints are **self-describing** — they carry `dim`, `hidden`, `head_type` and
`mlp_blocks`, so the loader rebuilds the architecture without external configuration,
plus `val_top1`, `val_f1`, `epoch` and `temperature` for traceability.

One implementation constraint worth knowing: the head has **no** dropout,
LayerNorm or output scaling — `hidden` and `mlp_blocks` are the only structural
knobs, and passing a dropout argument is silently ignored.

## Results

Backbone fixed to this release's `backbone/splm_tm_backbone.pth`. MALIDUP and
MALISAM report **full-alignment F1**, computed with CLAlign's `align_core` and
`f1score`, matching the protocol of their Table 1.

| | Trainable params | val top-1 | MALIDUP F1 | MALISAM F1 |
|---|---|---|---|---|
| No head (= the identity starting point) | 0 | 0.5050 | 0.5780 | 0.2357 |
| **`align_head.pt`** | **4.20M** | **0.5579** | **0.6428** | **0.2822** |
| CLAlign-ProtT5 (same backbone, LoRA fine-tuning) | LoRA r=8 | 0.542 | 0.6456 | 0.2666 |
| CLAlign-ProstT5 (their strongest variant) | LoRA r=8 | 0.562 | 0.6766 | 0.2994 |

A frozen backbone with a 4.2M-parameter head is level with LoRA fine-tuning of the
same backbone on MALIDUP (0.6428 vs 0.6456) and ahead on MALISAM (+0.0156).

> MALISAM pairs are analogous — structurally similar but not homologous, median
> sequence identity 8.2%, all 130 pairs below 30%. MALIDUP pairs are homologous
> duplicated domains. **The two sets behave differently and often point in opposite
> directions; do not average them.**

## Files

| File | Contents |
|---|---|
| `align_head.pt` | The default head, 16.8 MB |
| `alternatives/align_head_published_table.pt` | `hidden=1024` variant (2.10M parameters, temperature 0.1), corresponding to the earlier published row 0.6341 / 0.2718. For reproducing those numbers only |
| `provenance/*.json` | Per-pair MALIDUP / MALISAM evaluation results for the default head, the earlier head, and the no-head baseline (`mean` holds precision / recall / f1) |

## Verification

The head is reimplemented in `splm_tm` rather than imported from the research
codebase. A mistranscription would not raise an error — it would quietly produce
slightly different vectors — so parity was checked explicitly against the research
implementation.

Verified for both the `hidden=1024` and `hidden=2048` heads at L = 37 / 120 / 512:
`max |diff| = 0.0` on both weights and forward pass.
