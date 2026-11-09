# S-PLM-TM — model card

One released model: the backbone checkpoint below, plus three task heads.

## Files

Weights are downloaded separately — see [DOWNLOAD.md](DOWNLOAD.md).

| Role | Path | Size | md5 |
|---|---|---|---|
| Backbone weights | `backbone/splm_tm_backbone.pth` | 2,343,711,180 B (2.2 GB) | `3656217c152aea153c0b4c4ea6c93b61` |
| Architecture definition | `backbone/splm_tm_backbone.yaml` | 8,290 B | `e02c562f691d63d2499fbf2ce4d482e6` |
| TM calibration head | `tm_head/tm_calib_head.pt` | 12,597,452 B | `62ed269464b624caceac55b7d7f2829e` |
| **EC retrieval head** | `ec_head/ec_head.pt` | 8,407,402 B | `e306df5929c804702a7fb5078ca267f6` |
| **Residue-level alignment head** | `align_head/align_head.pt` | 16,792,312 B | `ae84fa7ef77bcc8877606b3dd0ee9ee8` |
| Alignment head, earlier published setting | `align_head/alternatives/align_head_published_table.pt` | 8,399,616 B | `3fee506693f5b21885102d2cfee12c62` |

> The alignment head consumes something different from the other two. TM and EC heads
> take **protein-level** vectors (`emb_raw` / `emb_h`); the alignment head takes
> **per-residue** features (`Backbone.encode_residues`, one `[L, 1024]` array per
> sequence). It is a branch off the residue features, not a step in the
> `raw → tm → ec` chain. See [align_head/README.md](align_head/README.md).

## Pairing constraints

| Pair | What a mismatch does |
|---|---|
| config ↔ checkpoint | The config determines the architecture (adapter layers and scaling, flow parameters, projector width). A mismatch **does not raise an error**; the output is simply meaningless |
| TM head ↔ checkpoint | The head was fitted on this checkpoint's output distribution and does not transfer to a different backbone |
| EC head ↔ TM head ↔ checkpoint | The EC head consumes the **TM head's output** (`emb_h`), so it is bound to both. `in_dim` must be 1024 |

The config shares the checkpoint's basename (`splm_tm_backbone.yaml` /
`splm_tm_backbone.pth`) so that the pairing is visible at a glance. This release
ships exactly one backbone, one config and one TM head, so the three files under
`models/` *are* the pairing — never mix in a file from another build.

## The checkpoint is a slim archive

The 2.2 GB file **excludes the frozen ProtT5-XL trunk** (~3B parameters, frozen
throughout, so there is nothing to store). The trunk is fetched from
`Rostlab/prot_t5_xl_uniref50` at load time. Therefore:

- a log line reporting 196 skipped frozen tensors is **expected** and indicates a
  correct load;
- the first run needs either a warm `HF_HOME` cache or ~11 GB of download.

## Model description

Two towers. The sequence tower is ProtT5-XL (`Rostlab/prot_t5_xl_uniref50`,
**frozen**) with a 1.61M-parameter adapter in parallel on the last **12 of 24**
T5 blocks. `scale_to_input: True` with `gain_init: 0.44` scales the adapter output
in proportion to the input RMS — necessary because ProtT5 hidden states have a
much larger RMS than the adapter design assumed, which would otherwise leave the
adapter's relative influence roughly 30× too small. The structure tower is a GVP
encoder; it must be importable but takes no part in sequence-only encoding.

The protein vector is the 1024-d `features_seq`, pooled with `pool_mode: "2"`
(mean over real residues, excluding padding and EOS) and taken **before** the
projector. That vector is `emb_raw`.

**TM calibration head.** A small MLP (1024→1024→1024, zero-initialised residual,
identity at initialisation) on `emb_raw`. Its output `emb_h` is L2-normalised, so
the inner product of two vectors is the predicted TM-score.

**EC retrieval head.** Another small MLP on the **TM head's output** (`emb_h`):
LayerNorm → 1024 → GELU → 1024, residual with a zero-initialised last layer.
Outputs are L2-normalised and ranked by Euclidean distance.

> It is a **retrieval head, not a classifier** — there are no EC class logits, and
> this release does not perform EC classification. See
> [ec_head/README.md](ec_head/README.md).

### Key architecture settings

| Item | Value |
|---|---|
| Sequence encoder | `Rostlab/prot_t5_xl_uniref50` (frozen), `num_end_adapter_layers: 12`, `max_length: 512` |
| Adapter | 1.61M parameters, `scale_to_input: True`, `gain_init: 0.44` |
| Protein-level pooling | `pool_mode: "2"` — mean over real residues, excluding padding and EOS |
| Output vector | 1024-d `features_seq`, taken **before** the projector |

## Evaluation

> **How these were produced:** `code/scripts/embed.py` for vectors, then
> `code/scripts/search.py` for rankings. The **SCOPe remote-homology** and
> **bacterial EC** rows use `--prefilter pfamclan`; run `bash code/scripts/setup_pfam.sh`
> once to install what the prefilter needs (Pfam 36.0; `Pfam-A.clans.tsv` ships with
> the release). The **CATH and GO** rows use no prefilter — those protocols score the
> full library and compute F1max over the complete ranking, so a prefilter would
> exclude most true positives. Without the prefilter the first two rows are lower.

| Benchmark | This model | Standing |
|---|---|---|
| SCOPe remote homology, 6 bins (seqID < 30%) | fam[0,10) 0.4816 | Loses no bin to any of the 11 baselines |
| SCOPe, matched prefilter (vs PLMSearch, 20 bins) | — | Loses 1 bin (fam[90,100], a ceiling bin) |
| CATH F1max, mean over 4 levels | 0.5506 | Loses only 2 bins each to GTalign and TMalign |
| CATH HomSF F1max | 0.7034 | as above |
| GO-MF BMA (experimental) | 0.4572 | Above all 16 baselines (best baseline TMalign 0.4495) |
| Bacterial EC, mean recall | 0.2659 | Level with Dali (0.2631), the best structural aligner here |
| Residue alignment, MALIDUP / MALISAM | 0.6428 / 0.2822 | Level with CLAlign-ProtT5 (0.6456 / 0.2666) on MALIDUP, ahead by 0.0156 on MALISAM |

> The bacterial EC row is scored on the 90 EC-family templates released with
> DEER, one representative per family. An earlier internal run used all 200
> human enzymes carrying those same 90 EC numbers, which weights families with
> many human paralogues up to 12-fold and contradicts that study's per-family
> weighting; those numbers are superseded.

> The last row uses neither the TM head nor the Pfam prefilter — it is a separate
> pipeline (residue features → alignment head → L2 normalisation → inner product),
> listed here only to keep the results in one place.

Against the unmodified backbones, with the same head and prefilter settings:

| Comparison | SCOPe (20 bins) | Bacterial (7) | CATH-AUC (4) | F1max (4) | GO |
|---|---|---|---|---|---|
| vs ProstT5 | 13 win / 2 loss / 5 tie | 5 win | 4 win | 4 win | win |
| vs ESM2-650M | 19 win | 5 win | 4 win | 4 win | win |

> The two losses are fam[20,30) (−0.0040) and sup[20,30) (−0.0014); all five ties are
> ceiling bins at 0.9998–1.0000.

## Known limitations

1. **Bacterial EC retrieval is a weak point.** Dali leads on recall (0.3177),
   precision@10 (0.4243) and hit-ratio@30 (0.5568); this model is ahead only on
   precision@100.
2. **CATH sens-1FP loses at all four levels to structural alignment methods**
   (GTalign, Dali, Foldseek, TMalign). That is their home ground; this model's
   advantage is in F1max and in the remote-homology regime.
3. **In the SCOPe high-identity regime (≥40%) BLAST and HMMER are ahead**, though all
   methods sit between 0.99 and 1.00 there — a ceiling effect rather than a gap in
   capability.
4. **Sequences are encoded in full length by default.** `--max-length N` keeps only
   the first N residues; each benchmark's reproduction records the length it used.
5. **Do not use `emb_tm` to train your own downstream models** — use `emb_raw`. See
   the embedding table in [code/README.md](../code/README.md#model-outputs).

## Provenance

The heads are reimplemented in `splm_tm` rather than imported from the research
codebase, so that this release stands alone. Parity against the original
implementations is checked by `code/scripts/check_*_parity.py`; both the EC head and
the alignment head reproduce the originals with `max |diff| = 0.0` on weights and
forward pass.
