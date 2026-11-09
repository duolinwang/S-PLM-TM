# -*- coding: utf-8 -*-
"""S-PLM-TM 序列塔的加载与批量编码。

产出的是 **pre-projector 的蛋白级向量**（`features_seq`，1024 维，未归一化），
也就是 npz 里的 `emb_raw`。TM head / EC head 都架在它之上。

编码口径与发布的 benchmark 数字一致：分批规则、截断规则和 forward
调用方式都与产出那些数字时相同。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from . import paths


def ensure_backbone_importable() -> Path:
    """把 code/splm_backbone 放到 sys.path 最前面，并校验真的命中了它。

    为什么要校验：`utils` / `model_v6` 这种模块名很容易被别的仓库（或 PYTHONPATH /
    SCOPE_EXTRA_PATH 里的目录）抢先命中，装出来的是 **ESM2 架构**，而且一路不报错，
    只是输出维度变成 1280。宁可在这里炸掉。
    """
    root = paths.BACKBONE_CODE_DIR
    if not (root / "model_v6.py").is_file():
        raise FileNotFoundError(
            f"{root} 下找不到 model_v6.py —— release 的 code/splm_backbone 不完整。")
    s = str(root)
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    return root


def _check_module_origin(mod, root: Path) -> None:
    f = getattr(mod, "__file__", None)
    if f and root not in Path(f).resolve().parents and Path(f).resolve().parent != root:
        raise RuntimeError(
            f"模块 {mod.__name__} 解析到了 {f}，不在 {root} 下。\n"
            f"  几乎肯定是 PYTHONPATH / SCOPE_EXTRA_PATH 里有同名的 utils / model_v6 抢先命中，\n"
            f"  那会把 ProtT5 的 checkpoint 装进 ESM2 的架构（输出会变成 1280 维）。\n"
            f"  sys.path[:5]={sys.path[:5]}")


class Backbone:
    """加载好的序列塔 + 配套 batch_converter。只做推理，内部全程 no_grad。"""

    def __init__(self, model, batch_converter, device, impl: str,
                 config_path: Path, ckpt_path: Path, step=None):
        self.model = model
        self.batch_converter = batch_converter
        self.device = device
        self.impl = impl
        self.config_path = Path(config_path)
        self.ckpt_path = Path(ckpt_path)
        self.step = step
        self._dim: Optional[int] = None

    # ---------------------------------------------------------------- 加载
    @classmethod
    def load(cls, config_path=None, ckpt_path=None, device=None,
             verbose: bool = True) -> "Backbone":
        config_path = Path(config_path) if config_path else paths.default_config()
        ckpt_path = Path(ckpt_path) if ckpt_path else paths.default_ckpt()
        for p in (config_path, ckpt_path):
            if not p.is_file():
                raise FileNotFoundError(p)

        root = ensure_backbone_importable()
        # HF 主干（Rostlab/prot_t5_xl_uniref50，~11GB）走 transformers 的缓存；
        # 没设 HF_HOME 时 transformers 会用 ~/.cache/huggingface 并在首次运行下载。
        if verbose and not (os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")):
            print(">>> 未设 HF_HOME：ProtT5 主干将用默认缓存目录，首次运行需要联网下载 ~11GB")

        import utils.generate_seq_embedding as gse       # noqa: E402  (路径就绪后才能 import)
        _check_module_origin(gse, root)

        model, batch_converter, dev = gse.SequenceRepresentModel(
            config_path=str(config_path), checkpoint_path=str(ckpt_path),
            # None = 沿用 config 里的设置（本模型 config 不开 truncate_inference），
            # 与 SCOPE / CATH / GO 评测走的是同一条分支。实际截断由 encode(max_length=) 控制。
            truncate_inference=None, max_length_inference=None, afterproject=False,
        )
        impl = "utils.generate_seq_embedding (与发布评测一致)"
        # step 由 SequenceRepresentModel 自己打印（">>> 已装载 state_dict1 (step=15000, slim=True)"），
        # 这里不再二次 torch.load 那 2.2GB 的存档。
        step = None

        if device is not None:
            dev = torch.device(device)
            model = model.to(dev)
        model.eval()
        if verbose:
            print(f">>> 编码实现: {impl}")
            print(f">>> device: {dev}")
        return cls(model, batch_converter, dev, impl, config_path, ckpt_path, step)

    # ---------------------------------------------------------------- 编码
    @staticmethod
    def _plan_batches(lengths, batch_size: int, max_tokens: int):
        """按长度排序后做 token 预算分批：显存由注意力 O(b*L^2) 主导，固定 b 会在长序列上爆。
        保证 b * L_max <= max_tokens，长序列自动落到小 batch。"""
        order = np.argsort(lengths)
        batches, cur, cur_max = [], [], 0
        for i in order:
            L = int(lengths[i]) + 2                          # cls/eos
            new_max = max(cur_max, L)
            if cur and (len(cur) >= batch_size or new_max * (len(cur) + 1) > max_tokens):
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = L
            cur.append(int(i))
            cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    @torch.inference_mode()
    def encode(self, seqs: Sequence[str], ids: Optional[Sequence[str]] = None,
               batch_size: int = 8, max_length: Optional[int] = None,
               max_tokens: int = 8192, log_every: int = 50,
               verbose: bool = True) -> np.ndarray:
        """序列 -> [N, 1024] float32 的 pre-projector 蛋白向量（emb_raw）。

        max_length: 默认 None = 全长；正整数 N 时仅编码前 N 个残基。
        max_tokens: 单 batch 的 b*L_max 上限。ProtT5-XL 比 ESM2 吃显存，默认取保守值。
        """
        seqs = [str(s) if max_length is None else str(s)[:max_length] for s in seqs]
        if ids is None:
            ids = [str(i) for i in range(len(seqs))]
        lengths = np.array([len(s) for s in seqs])
        batches = self._plan_batches(lengths, batch_size, max_tokens)
        out: Optional[np.ndarray] = None
        t0 = time.time()
        n_batches = len(batches)
        if verbose:
            print(f"    {len(seqs)} 条 -> {n_batches} 个 batch (max_len={lengths.max()}, "
                  f"batch_size<={batch_size}, max_tokens={max_tokens})", flush=True)
        for bi, sel in enumerate(batches):
            batch = [(str(ids[i]), seqs[i]) for i in sel]
            _, _, toks = self.batch_converter(batch)
            toks = toks.to(self.device)
            # ⚠️ 必须用关键字传参：SimCLR.forward 的第一个位置参数是 graph 而不是
            # batch_tokens，位置传参会让 batch_tokens=None 并报
            # "'NoneType' object has no attribute 'ndim'"。保留位置回退以兼容其它模型类。
            try:
                _, _, feats, _ = self.model(batch_tokens=toks, mode="sequence",
                                            return_embedding=True)
            except TypeError:
                _, _, feats, _ = self.model(toks, mode="sequence", return_embedding=True)
            feats = feats.float().cpu().numpy()
            if out is None:
                out = np.zeros((len(seqs), feats.shape[1]), dtype=np.float32)
            out[sel] = feats
            if verbose and ((bi + 1) % log_every == 0 or bi + 1 == n_batches):
                el = time.time() - t0
                print(f"  [{bi+1}/{n_batches}] {el:.0f}s elapsed, "
                      f"ETA {el/(bi+1)*(n_batches-bi-1):.0f}s", flush=True)
        assert out is not None
        self._dim = int(out.shape[1])
        if self._dim != 1024:
            print("!" * 76)
            print(f"!! 警告：输出维度是 {self._dim}，S-PLM-TM 应为 1024。")
            print("!!       1280 = 装成了 ESM2 架构，说明 sys.path 上有别的仓库的 utils/model_v6。")
            print("!" * 76)
        return out

    @torch.inference_mode()
    def encode_residues(self, seqs: Sequence[str], ids: Optional[Sequence[str]] = None,
                        batch_size: int = 8, max_length: Optional[int] = None,
                        max_tokens: int = 8192, log_every: int = 50,
                        verbose: bool = True) -> List[np.ndarray]:
        """序列 -> 每条一个 [L, 1024] float32 的**残基级**特征数组（list，与输入同序）。

        这是比对头（`models/align_head/`）的输入。与 `encode` 的唯一区别是取 forward 的
        **第 4 个**返回值而不是第 3 个 —— 第 3 个是蛋白级 pooled 向量（emb_raw），
        第 4 个是 residue_feature_embedding。

        ⚠️ 第 4 个返回值的形状是 **[sum_L, D]**：batch 内所有序列的非特殊 token
        （不含 cls/eos）按行拼接成一条，**不是** [B, L, D]。所以必须按各自长度切回去，
        下面那条 assert 就是防止这个约定哪天变了还静默切错。
        口径与产出全部已发布比对数字时所用的残基特征缓存逐字一致。

        max_length: 默认 None = 全长，与 encode 一致。正整数 N 会**真的丢残基**
        （被截掉的位置不会出现在输出里）。
        """
        seqs = [str(s) if max_length is None else str(s)[:max_length] for s in seqs]
        if ids is None:
            ids = [str(i) for i in range(len(seqs))]
        lengths = np.array([len(s) for s in seqs])
        batches = self._plan_batches(lengths, batch_size, max_tokens)
        out: List[Optional[np.ndarray]] = [None] * len(seqs)
        t0 = time.time()
        n_batches = len(batches)
        if verbose:
            print(f"    {len(seqs)} 条 -> {n_batches} 个 batch（残基级，max_len="
                  f"{lengths.max()}）", flush=True)
        for bi, sel in enumerate(batches):
            batch = [(str(ids[i]), seqs[i]) for i in sel]
            _, _, toks = self.batch_converter(batch)
            toks = toks.to(self.device)
            # 关键字传参的原因同 encode()。
            try:
                _, _, _, res = self.model(batch_tokens=toks, mode="sequence",
                                          return_embedding=True)
            except TypeError:
                _, _, _, res = self.model(toks, mode="sequence", return_embedding=True)
            res = res.float().cpu().numpy()
            lens = [len(s) for _, s in batch]
            if res.shape[0] != sum(lens):
                raise RuntimeError(
                    f"残基数对不上：forward 返回 {res.shape[0]} 行，序列总长 {sum(lens)}。\n"
                    f"  说明 residue_feature_embedding 的拼接约定与预期不同"
                    f"（预期 = batch 内非特殊 token 按行拼接）。\n"
                    f"  别猜着切 —— 先核对 model_v6 的 forward。")
            off = 0
            for i, L in zip(sel, lens):
                out[i] = res[off:off + L]
                off += L
            if verbose and ((bi + 1) % log_every == 0 or bi + 1 == n_batches):
                el = time.time() - t0
                print(f"  [{bi+1}/{n_batches}] {el:.0f}s elapsed, "
                      f"ETA {el/(bi+1)*(n_batches-bi-1):.0f}s", flush=True)
        missing = [i for i, a in enumerate(out) if a is None]
        if missing:
            raise RuntimeError(f"有 {len(missing)} 条序列没被编码到（索引 {missing[:5]}…）")
        d = {a.shape[1] for a in out}
        if d != {1024}:
            print("!" * 76)
            print(f"!! 警告：残基特征维度是 {sorted(d)}，S-PLM-TM 应为 1024。")
            print("!!       1280 = 装成了 ESM2 架构（见 ensure_backbone_importable 的说明）。")
            print("!" * 76)
        return out                                              # type: ignore[return-value]

    @property
    def dim(self) -> int:
        """输出维度。未编码过时从 config 读 projector 输入维度。"""
        if self._dim is None:
            return 1024
        return self._dim

    def encode_fasta(self, fasta_path, **kw):
        from .fasta import read_fasta
        ids, seqs = read_fasta(fasta_path)
        return ids, self.encode(seqs, ids, **kw)


def summarize_lengths(seqs: Sequence[str], max_length: Optional[int]) -> str:
    L = [len(s) for s in seqs]
    msg = f"{len(seqs)} 条序列  长度 中位={int(np.median(L))} 最长={max(L)}"
    if max_length:
        frac = 100.0 * sum(x > max_length for x in L) / len(L)
        msg += f" (>{max_length} 的占 {frac:.1f}%，将被截断)"
    return msg


__all__ = ["Backbone", "ensure_backbone_importable", "summarize_lengths"]
