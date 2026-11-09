"""Model adapters for the persistent search API. Models are loaded explicitly."""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np

from .index import file_sha256


def _fingerprint(files):
    return {name: file_sha256(path) for name, path in files.items()}


class T5SearchEncoder:
    """ProtT5 raw / TM / EC representations, with original head batching.

    Unlike SPLMTM.embed, head inference is split into chunks of 4,096 by default,
    matching the existing large-scale target encoder. This does not change the
    existing embedding API or its defaults.
    """

    def __init__(self, model, identity, mode="tm", head_batch_size=4096):
        if mode not in ("raw", "tm", "ec") or head_batch_size < 1:
            raise ValueError("Invalid mode or head batch size")
        self.model, self.identity, self.mode = model, identity, mode
        self.head_batch_size = head_batch_size

    @classmethod
    def load(cls, mode="tm", config_path=None, ckpt_path=None, tm_head_path=None,
             ec_head_path=None, device=None, head_batch_size=4096, verbose=True):
        from . import paths
        from .api import SPLMTM

        if mode not in ("raw", "tm", "ec") or head_batch_size < 1:
            raise ValueError("Invalid mode or head batch size")
        config_path = Path(config_path) if config_path else paths.default_config()
        ckpt_path = Path(ckpt_path) if ckpt_path else paths.default_ckpt()
        files = dict(config=config_path, checkpoint=ckpt_path)
        if mode in ("tm", "ec"):
            tm_head_path = Path(tm_head_path) if tm_head_path else paths.default_tm_head()
            files["tm_head"] = tm_head_path
        if mode == "ec":
            ec_head_path = Path(ec_head_path) if ec_head_path else paths.default_ec_head()
            files["ec_head"] = ec_head_path
        code_root = Path(__file__).resolve().parent
        source_files = [code_root / name for name in
                        ("backbone.py", "heads.py", "search_encoders.py")]
        source_files += sorted(paths.BACKBONE_CODE_DIR.rglob("*.py"))
        identity = dict(backend="prot_t5", mode=mode, files=_fingerprint(files),
                        source={str(path.relative_to(paths.CODE_DIR)): file_sha256(path)
                                for path in source_files},
                        head_batch_size=head_batch_size)
        model = SPLMTM.load(modes=(mode,), config_path=config_path, ckpt_path=ckpt_path,
                            tm_head_path=tm_head_path, ec_head_path=ec_head_path,
                            device=device, verbose=verbose)
        return cls(model, identity, mode, head_batch_size)

    def encode(self, seqs, ids, max_length=None, batch_size=8, max_tokens=8192):
        import torch
        from .heads import apply_ec_head

        raw = self.model.backbone.encode(seqs, ids, max_length=max_length,
                                         batch_size=batch_size, max_tokens=max_tokens,
                                         verbose=False)
        if raw.ndim != 2 or raw.shape[1] != 1024:
            raise RuntimeError("ProtT5 search requires 1,024-dimensional raw embeddings")
        if self.mode == "raw":
            return raw
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(raw), self.head_batch_size):
                value = torch.from_numpy(raw[start:start + self.head_batch_size]).float()
                chunks.append(self.model.tm_head(value.to(self.model.head_device)).float().cpu().numpy())
        tm = np.concatenate(chunks, axis=0)
        return apply_ec_head(self.model.ec_head, tm, device=self.model.head_device)[0] \
            if self.mode == "ec" else tm

    def synchronize(self):
        import torch

        device = torch.device(self.model.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


class LegacyESMSearchEncoder:
    """Compatibility with the original ESM2 SPLM research implementation.

    Requires the original ESM research checkout and its environment. It is kept
    separate from the bundled ProtT5 implementation: importing both architectures
    into one Python process is refused if their legacy module names collide.
    This adapter does not claim ESM/T5 head interchangeability or checkpoint parity.
    ESM sequences are encoded individually, matching the original benchmark.
    """

    @classmethod
    def load(cls, research_repo, config_path, ckpt_path, device=None):
        import torch

        root = Path(research_repo).expanduser().resolve()
        loader_path = root / "utils/generate_seq_embedding.py"
        for path in (loader_path, root / "model_v6.py", Path(config_path), Path(ckpt_path)):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not (root / "esm_adapterH").is_dir():
            raise ValueError("--research-repo must be the original ESM checkout, not the bundled T5 code")
        # These original implementations use top-level utils/model_v6/gvp imports.
        # Silently reusing the bundled T5 modules would produce incorrect vectors.
        for name in ("utils", "utils.generate_seq_embedding", "model_v6", "esm_adapterH", "gvp"):
            module = sys.modules.get(name)
            source = getattr(module, "__file__", None)
            if source and root not in Path(source).resolve().parents:
                raise RuntimeError("ESM and T5 legacy modules conflict; use a separate Python process")
        sys.path.insert(0, str(root))
        loader = importlib.import_module("utils.generate_seq_embedding")
        if Path(loader.__file__).resolve() != loader_path:
            raise RuntimeError("Wrong ESM loader imported")
        model, converter, dev = loader.SequenceRepresentModel(
            str(config_path), str(ckpt_path), truncate_inference=False, afterproject=False)
        if device is not None:
            dev = torch.device(device)
            model = model.to(dev)
        result = cls()
        result.model, result.converter, result.device = model.eval(), converter, dev
        result.use_keyword = "batch_tokens" in inspect.signature(model.forward).parameters
        source_files = [root / "model_v6.py", loader_path]
        for name in ("esm_adapterH", "gvp", "utils", "crossflow"):
            source_files.extend(sorted((root / name).rglob("*.py")))
        result.identity = dict(backend="esm2_legacy", mode="raw", encoding="individual",
                               files=_fingerprint(dict(config=Path(config_path), checkpoint=Path(ckpt_path))),
                               source={str(p.relative_to(root)): file_sha256(p) for p in source_files})
        return result

    def encode(self, seqs, ids, max_length=None, batch_size=1, max_tokens=8192):
        import torch

        out = []
        with torch.inference_mode():
            for identifier, sequence in zip(ids, seqs):
                sequence = sequence if max_length is None else sequence[:max_length]
                _, _, tokens = self.converter([(identifier, sequence)])
                tokens = tokens.to(self.device)
                if self.use_keyword:
                    _, _, features, _ = self.model(batch_tokens=tokens, mode="sequence", return_embedding=True)
                else:
                    _, _, features, _ = self.model(tokens, mode="sequence", return_embedding=True)
                out.append(features.float().cpu().numpy()[0])
        return np.stack(out).astype(np.float32)

    def synchronize(self):
        import torch

        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


__all__ = ["T5SearchEncoder", "LegacyESMSearchEncoder"]
