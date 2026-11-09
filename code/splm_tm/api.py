# -*- coding: utf-8 -*-
"""S-PLM-TM 的统一入口：给定任意序列，按需拿 raw / tm / ec 三种 embedding。

    from splm_tm import SPLMTM

    m = SPLMTM.load(modes=("raw", "tm"))          # 只装用得到的头
    r = m.embed(["MVLSEGEWQ...", "KVFGRCELAA..."])
    r["emb_raw"]   # [N,1024] pre-projector 原始向量 —— 迁移下游（分类/回归）用这个
    r["emb_tm"]    # [N,1024] 已 L2 归一化，内积即「预测 TM-score」—— 结构检索用这个
    r["emb_ec"]    # [N,1024] EC 检索头输出，已 L2 归一化 —— EC/功能检索用这个

依赖链是**串行**的：

    序列 --backbone--> emb_raw --TM head--> emb_tm(=emb_h) --EC head--> emb_ec

所以 `--modes ec` 会自动把 tm 带上（EC head 是在 `emb_h` 上训的），
而一次 forward 就能给出全部三种，backbone 不会跑两遍。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import heads, paths
from .backbone import Backbone, summarize_lengths
from .fasta import make_ids, read_fasta

MODES = ("raw", "tm", "ec")
# 每个 mode 需要哪些前置 mode（EC head 吃 emb_h = TM head 的输出）
MODE_REQUIRES = {"raw": (), "tm": ("raw",), "ec": ("raw", "tm")}


def _norm_modes(modes) -> Tuple[str, ...]:
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.replace(",", " ").split()]
    out = []
    for m in modes:
        m = str(m).lower()
        if m in ("h", "emb_h", "head", "tmhead"):      # 兼容旧叫法（npz 里的 emb_h）
            m = "tm"
        if m == "all":
            out = list(MODES)
            break
        if m not in MODES:
            raise ValueError(f"未知 mode: {m}；可选 {MODES} 或 'all'")
        out.append(m)
    want = set(out)
    for m in list(want):
        want.update(MODE_REQUIRES[m])
    return tuple(m for m in MODES if m in want)        # 固定成 raw -> tm -> ec 的顺序


class SPLMTM:
    """加载好的 backbone + 可选的两个头。构造一次、反复 `embed()`。"""

    def __init__(self, backbone: Backbone, tm_head=None, ec_head=None,
                 head_device: str = "cpu"):
        self.backbone = backbone
        self.tm_head = tm_head
        self.ec_head = ec_head
        self.head_device = head_device

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(cls, modes=("raw", "tm"), config_path=None, ckpt_path=None,
             tm_head_path=None, ec_head_path=None, device=None,
             verbose: bool = True) -> "SPLMTM":
        """只加载 `modes` 真正用到的头（含自动带上的前置 mode）。"""
        want = _norm_modes(modes)

        # ⚠️ 先把要用的头文件解析出来并确认存在，再去装那 2.2 GB 的 backbone ——
        # 否则 head 缺失要白等几分钟才报错。
        tm_p = ec_p = None
        if "tm" in want:
            tm_p = Path(tm_head_path) if tm_head_path else paths.default_tm_head()
            if not tm_p.is_file():
                raise FileNotFoundError(f"TM head 不存在：{tm_p}")
        if "ec" in want:
            ec_p = Path(ec_head_path) if ec_head_path else paths.default_ec_head(required=False)
            if ec_p is None or not Path(ec_p).is_file():
                raise FileNotFoundError(
                    heads.ec_head_missing_message(paths.models_dir() / "ec_head"))

        bb = Backbone.load(config_path, ckpt_path, device=device, verbose=verbose)
        head_dev = str(bb.device)

        tm = None
        if tm_p is not None:
            tm = heads.load_tm_head(tm_p, device=head_dev)
            if verbose:
                print(f">>> TM head: {tm_p}")

        ec = None
        if ec_p is not None:
            ec = heads.ECRetrievalHead.load(ec_p, map_location="cpu").to(head_dev).eval()
            if verbose:
                print(f">>> EC head: {ec_p}")
                print(f"    hidden={ec.cfg['hidden']}, out_dim={ec.out_dim}, "
                      f"n_blocks={ec.cfg['n_blocks']}, seed={ec.meta.get('seed')}, "
                      f"参数 {sum(p.numel() for p in ec.parameters()):,}")
        return cls(bb, tm, ec, head_dev)

    # ------------------------------------------------------------------ 编码
    def embed(self, seqs: Sequence[str], ids: Optional[Sequence[str]] = None,
              modes=("raw", "tm"), batch_size: int = 8, max_length: Optional[int] = None,
              max_tokens: int = 8192, verbose: bool = True) -> Dict[str, np.ndarray]:
        """序列 -> {"ids", "emb_raw"[, "emb_tm"/"emb_h"][, "emb_ec", "emb_ec_raw"]}。

        默认编码全长；显式传入正整数 max_length 时仅编码前 N 个残基。
        """
        want = _norm_modes(modes)
        seqs = [str(s) for s in seqs]
        if not seqs:
            raise ValueError("没有输入序列")
        ids = list(ids) if ids is not None else make_ids(len(seqs))
        if len(ids) != len(seqs):
            raise ValueError(f"ids 有 {len(ids)} 个，序列有 {len(seqs)} 条")
        if verbose:
            print(">>> " + summarize_lengths(seqs, max_length))

        raw = self.backbone.encode(seqs, ids, batch_size=batch_size,
                                   max_length=max_length, max_tokens=max_tokens,
                                   verbose=verbose)
        out: Dict[str, np.ndarray] = {"ids": np.array(ids, dtype=object), "emb_raw": raw}
        if verbose:
            print(f">>> emb_raw shape={raw.shape}")

        emb_tm = None
        if "tm" in want:
            if self.tm_head is None:
                raise RuntimeError("要 mode='tm' 但 TM head 没加载 —— "
                                   "SPLMTM.load(modes=(...,'tm')) 时才会装它")
            emb_tm, drift = heads.apply_tm_head(self.tm_head, raw, device=self.head_device)
            out["emb_tm"] = emb_tm
            out["emb_h"] = emb_tm          # 旧名字，保持与既有 npz / 下游脚本兼容
            if verbose:
                n = np.linalg.norm(emb_tm, axis=1)
                print(f">>> emb_tm shape={emb_tm.shape}  drift={drift:.4f}  "
                      f"L2: min={n.min():.4f} max={n.max():.4f} （应恒为 1）")

        if "ec" in want:
            if self.ec_head is None:
                raise RuntimeError("要 mode='ec' 但 EC head 没加载 —— "
                                   "SPLMTM.load(modes=(...,'ec')) 时才会装它")
            # ⚠️ EC head 的输入是 emb_h（TM head 的输出），不是 emb_raw
            feats, feats_raw = heads.apply_ec_head(self.ec_head, emb_tm,
                                                   device=self.head_device)
            out["emb_ec"] = feats
            out["emb_ec_raw"] = feats_raw
            if verbose:
                n = np.linalg.norm(feats, axis=1)
                print(f">>> emb_ec shape={feats.shape}  (输入=emb_h)  "
                      f"L2: min={n.min():.4f} max={n.max():.4f} （应恒为 1）")
        return out

    def embed_fasta(self, fasta_path, **kw) -> Dict[str, np.ndarray]:
        ids, seqs = read_fasta(fasta_path)
        return self.embed(seqs, ids, **kw)

    # ------------------------------------------------------------------ 便捷
    def predicted_tm_matrix(self, emb_tm) -> np.ndarray:
        """emb_tm 已 L2 归一化，内积即「预测 TM-score」。不要再 normalize。"""
        e = np.asarray(emb_tm)
        return e @ e.T

    def ec_similarity_matrix(self, emb_ec) -> np.ndarray:
        """EC 检索相似度：emb_ec 已归一化，内积 = 余弦（与欧氏距离排序等价）。

        ⚠️ 这个数值**不是** TM-score，也不是 EC 概率，只是用来排序的相似度。
        """
        e = np.asarray(emb_ec)
        return e @ e.T

    # ------------------------------------------------------------------ 其它
    @property
    def device(self):
        return self.backbone.device

    def info(self) -> str:
        return (f"S-PLM-TM\n"
                f"  config : {self.backbone.config_path}\n"
                f"  ckpt   : {self.backbone.ckpt_path}\n"
                f"  impl   : {self.backbone.impl}\n"
                f"  device : {self.backbone.device}\n"
                f"  tm head: {'已加载' if self.tm_head is not None else '未加载'}\n"
                f"  ec head: {'已加载' if self.ec_head is not None else '未加载'}")


def save_npz(path, result: Dict[str, np.ndarray]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)
    return path


__all__ = ["SPLMTM", "MODES", "MODE_REQUIRES", "save_npz"]
