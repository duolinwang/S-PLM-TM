# EC head

`ec_head.pt` — the EC-number retrieval head.

```
sequence → backbone (frozen) → emb_raw → TM head (frozen) → emb_h → EC head → emb_ec
```

It is a **retrieval** head: it outputs a 1024-d vector for
ranking, not class logits or probabilities.

## Input

**The EC head consumes `emb_h` (the TM head's output), not `emb_raw`.** That is why
`--modes ec` pulls `tm` in automatically. Feeding it `emb_raw` will not raise an
error, it will just rank worse.

## Usage

```bash
python code/scripts/embed.py --fasta my.fasta --modes ec --out out.npz
python code/scripts/search.py --query q.npz --target t.npz --emb emb_ec --out hits.tsv
```

```python
import numpy as np
d = np.load("out.npz", allow_pickle=True)
emb_ec = d["emb_ec"]        # [N, 1024], L2-normalised
sim = emb_ec @ emb_ec.T     # cosine; equivalent to ranking by Euclidean distance
```

`out.npz` also carries `emb_ec_raw`, the un-normalised head output.

## Results — DEER Figure 7

90 templates / 4,213 database proteins / 449 true positives.

| Scope | Top-10 | Top-30 | Top-100 |
|---|---|---|---|
| Full | 0.4944 | 0.7194 | 0.8419 |
| ≤ 1024 residues | 0.5101 | 0.7146 | 0.8359 |
| ≤ 512 residues | 0.5217 | 0.7633 | 0.8261 |

Mean recall 0.3819 across the 90 EC families; relative precision @10/30/100 =
0.5162 / 0.6576 / 0.7358.

> Measured with `--max-length 1024`.

Reproduce: [`reproduce/ec_deer_figure7/`](../../reproduce/ec_deer_figure7/).

## Architecture

```
ECRetrievalHead(
  (ln):  LayerNorm(1024)
  (fc1): Linear(1024, 1024)
  (act): GELU
  (drop): Dropout(0.0)
  (fc2): Linear(1024, 1024)
)
forward: x + fc2(drop(act(fc1(ln(x)))))
```

2,101,248 parameters. The checkpoint stores `cfg` as a **delta against the defaults**
in `splm_tm.heads.MLP_DEFAULT_CFG`, so the full configuration is only recoverable
together with that table. `splm_tm.heads.ECRetrievalHead.load()` cross-checks the
stored `cfg` against the state-dict shapes and refuses to load on a mismatch.

## Files

| File | Contents |
|---|---|
| `ec_head.pt` | Weights, 8.4 MB — `{state_dict, cfg, family, seed, train}` |
