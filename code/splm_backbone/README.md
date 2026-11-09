# splm_backbone — model definition code

The architecture implementation for S-PLM-TM, as used for the published results.

**Do not edit these files.** Changing them breaks correspondence with the checkpoint
in `models/backbone/`.

The entry points are `../scripts/embed.py` and `../splm_tm/`. Do not call
scripts in this directory directly.

## What this release actually uses

| File | Role |
|---|---|
| `utils/generate_seq_embedding.py` | `SequenceRepresentModel()` — builds the model, loads the checkpoint, provides the batch converter. This is the ProtT5 variant, the same code path used for the published evaluations |
| `model_v6.py` | `prepare_models()` — assembles the two-tower model |
| `protT5_encoder.py` | ProtT5 sequence tower, plus `_T5Alphabet` / `T5BatchConverter`, which wrap the T5 tokenizer in an ESM-alphabet-shaped interface |
| `t5_adapter.py` | Adapter injection, including the `scale_to_input` proportional scaling |
| `ckpt_slim.py` | `load_slim_into()` — restores a slim archive and verifies that the missing keys are exactly the frozen trunk |
| `utils/tm_calib_head.py` | `TMCalibHead` — the TM calibration head and its `load()` |
| `utils/utils_v6.py` | `load_configs()` — the config parser for `models/backbone/*.yaml`, byte-for-byte the original implementation. This file holds nothing else |
| `gvp/`, `utils/SequenceSimGNN_pooling*.py` | Structure tower. It takes no part in sequence-only encoding, but must be importable |

`smoke_test.py` and `test_ckpt_slim.py` are self-checks rather than entry points.

## Module-name collisions

Generic names like `utils` and `model_v6` are easily shadowed by another package
earlier on `sys.path`. When that happens the loaded model is an **ESM-2**
architecture, and nothing raises — the only symptom is that the output dimension
changes from 1024 to 1280.

Two guards:

1. `splm_tm/backbone.py:ensure_backbone_importable()` inserts this directory at the
   front of `sys.path`, then verifies after import that
   `utils.generate_seq_embedding.__file__` really resolves inside this directory,
   raising if it does not.
2. `utils/generate_seq_embedding.py` additionally checks the backbone type and raises
   `RuntimeError` if the sequence tower is not ProtT5.

For the same reason, the batch scripts clear `PYTHONPATH` and related path variables
before running.
