# Getting the model weights

The weights are **not in this git repository** — the backbone checkpoint alone is
2.2 GB, well past GitHub's 100 MB per-file limit. A fresh clone gives you this
directory's documentation and provenance; the `*.pt` / `*.pth` files are downloaded
separately.

They are hosted on Hugging Face: **[`deepduoduo/S-PLM-TM`](https://huggingface.co/deepduoduo/S-PLM-TM)**

## Download

```bash
pip install "huggingface_hub[cli]"

# All five weight files (~2.4 GB). The architecture YAML is not on Hugging Face —
# it is text, so it ships with the git repository and is already in models/backbone/.
hf download deepduoduo/S-PLM-TM --local-dir models/
```

A single head:

```bash
hf download deepduoduo/S-PLM-TM ec_head/ec_head.pt --local-dir models/
```

From Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("deepduoduo/S-PLM-TM", local_dir="models/")
```

The files do not have to live under `models/`. Point environment variables at them
instead:

```bash
export SPLM_TM_MODELS=/path/to/weights          # the whole models/ root
# or individually
export SPLM_TM_CKPT=/path/to/splm_tm_backbone.pth
export SPLM_TM_EC_HEAD=/path/to/ec_head.pt
```

To see which paths actually resolve — the first thing to check when something is not
found:

```bash
splmtm embed --show-paths
```

## Verify the download

```bash
md5sum models/backbone/splm_tm_backbone.pth \
       models/tm_head/tm_calib_head_*.pt \
       models/ec_head/ec_head.pt \
       models/align_head/align_head.pt
```

| File | Size | md5 |
|---|---|---|
| `backbone/splm_tm_backbone.pth` | 2,343,711,180 B | `3656217c152aea153c0b4c4ea6c93b61` |
| `tm_head/tm_calib_head.pt` | 12,597,452 B | `62ed269464b624caceac55b7d7f2829e` |
| `ec_head/ec_head.pt` | 8,407,402 B | `e306df5929c804702a7fb5078ca267f6` |
| `align_head/align_head.pt` | 16,792,312 B | `ae84fa7ef77bcc8877606b3dd0ee9ee8` |
| `align_head/alternatives/align_head_published_table.pt` | 8,399,616 B | `3fee506693f5b21885102d2cfee12c62` |

If a checksum does not match, do not use the file — the results reported in
[MODEL_CARD.md](MODEL_CARD.md) were measured with exactly these weights.

## What else is needed

- The **ProtT5-XL trunk** is not included here. `transformers` fetches it at run time
  (`Rostlab/prot_t5_xl_uniref50`, ~11 GB). Set `HF_HOME` to a location with enough
  space first, or it will fill your home directory.
- The backbone `.pth` is a **slim archive** and excludes the frozen ProtT5 trunk. A
  log line about skipped frozen tensors during loading is expected — those tensors
  come from `from_pretrained`.
