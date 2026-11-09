---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
library_name: pytorch
tags:
  - protein-language-model
  - protein
  - biology
  - remote-homology
  - structure-aware
  - retrieval
---

# S-PLM-TM — model weights

Structure-aware protein sequence embeddings. One sequence in, a 1024-d vector out,
in three flavours — plus a residue-level head for alignment.

**These are the weights only.** The code, model cards and reproduction scripts live in
the GitHub repository: `https://github.com/duolinwang/S-PLM-TM`

> ⚠️ **Noncommercial license.** Academic, teaching and nonprofit/educational/
> public-research/government use is granted; **commercial use is not**.
> Full text: **`LICENSE` in this repository** (PolyForm Noncommercial 1.0.0).
> Cite the paper if you use this.
>
> The weights are public, but **the code repository is currently private** while the
> paper is under preparation — the GitHub links below will 404 unless you have access.
> Ask the authors if you need it before then.

## What's here

| File | Size | What it is |
|---|---|---|
| `backbone/splm_tm_backbone.pth` | 2.2 GB | The encoder. **Slim archive** — the frozen ProtT5-XL trunk is *not* included; it is fetched by `from_pretrained` at run time |
| `tm_head/tm_calib_head.pt` | 12 MB | TM calibration head → `emb_tm`, L2-normalised, inner product ≈ predicted TM-score |
| `ec_head/ec_head.pt` | 8.4 MB | EC retrieval head → `emb_ec`. ⚠️ It consumes `emb_h` (TM head output), **not** `emb_raw` |
| `align_head/align_head.pt` | 17 MB | Residue-level alignment head. Consumes per-residue features, **not** a pooled vector |

The architecture YAML is **not** here — it is text, so it lives in git (`models/backbone/` in the GitHub repo) where it has a history and a visible diff. Keeping a second copy here would let the two drift apart silently.

Every head is **paired with this exact backbone checkpoint**. Swapping the checkpoint
without swapping the heads produces silently meaningless vectors.

## Quick start

```bash
git clone https://github.com/duolinwang/S-PLM-TM.git && cd S-PLM-TM
pip install -r requirements.txt
hf download deepduoduo/S-PLM-TM --local-dir models/

python code/scripts/embed.py --fasta my.fasta --modes raw tm ec --out out.npz
```

```python
d = np.load("out.npz", allow_pickle=True)
d["emb_raw"]   # [N,1024] pre-projector, unnormalised — use this to train downstream models
d["emb_tm"]    # [N,1024] L2-normalised — structure / remote-homology retrieval
d["emb_ec"]    # [N,1024] L2-normalised — EC / functional retrieval
```

The three are **chained, not parallel**:

```
sequence --backbone--> emb_raw --TM head--> emb_tm (= emb_h) --EC head--> emb_ec
```

so `--modes ec` pulls `tm` in automatically, and one forward pass gives you all three.

## Things that will bite you

1. **Output is 1024-d, not 1280.** A 1280-d vector means the backbone was built as
   ESM-2 by mistake.
2. **Embedding defaults to full sequences.** Set `--max-length N` to truncate;
   `0` also disables truncation. Some published numbers (the EC head's DEER
   Figure-7 results) were measured with **`--max-length 1024`**. Match that setting
   before comparing.
3. **"Skipped 196 frozen trunk tensors" on load is normal** — that is the slim archive
   working as intended; those tensors come from `from_pretrained`.
4. **`emb_ec` is not an EC-number prediction.** The EC head is a *retrieval* head
   there are no class logits and no probabilities.
5. ProtT5-XL (~11 GB) is downloaded on first use. Set `HF_HOME` to somewhere with
   space before running on a cluster.

## Verify your download

md5 checksums for every file are in `models/DOWNLOAD.md` in the GitHub repo.
The numbers reported in `models/MODEL_CARD.md` were measured with exactly these files.

## Reproducibility

`reproduce/ec_deer_figure7/` in the GitHub repo reproduces the EC head's published
DEER Figure-7 numbers end to end (re-encoding all 4,303 sequences from FASTA, not
reusing cached embeddings). Verified bit-for-bit: hit counts 222/323/378 and all 90
per-family recall values identical to the reference.

## Citation

See the GitHub repository's README for the paper citation.
