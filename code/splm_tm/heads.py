# -*- coding: utf-8 -*-
"""架在 backbone 之上的两个头。

TM head
    `TMCalibHead`（定义在 splm_backbone/utils/tm_calib_head.py）。
    输入 `emb_raw` [N,1024] -> 输出 `emb_h` [N,1024]，**已 L2 归一化**，内积即「预测 TM-score」。
    用于结构检索 / 远同源搜索。与 backbone checkpoint **强配对**，换 ckpt 必须换 head。

EC head
    ⚠️ 它是**检索头（度量学习），不是分类器**。
    输出 1024 维向量，L2 归一化后按欧氏距离 / 内积排序。没有 EC 类别 logits。
    ⚠️ 它的输入是 **`emb_h`（TM head 之后、已 L2 归一化）**，不是 `emb_raw`。
    这是 exp_4-1.1 实际搜出来的口径（`heads_lib.EXP_INPUT['exp_4-1.1'] = 'emb_h'`），
    所以本 release 的依赖链是 raw -> tm -> ec，`--modes ec` 会自动把 tm 带上。

    结构 = `ECRetrievalHead`，**逐字复刻**原实现的
    `src/experiments/exp_3-1.1/heads_lib.py` 里的 `_MLPBlock` / `MLPHead`
    （LayerNorm -> Linear -> GELU -> Dropout -> Linear，残差相加）。
    复刻而非 import，是为了让 release 不依赖那个目录；
    数值等价性由 `scripts/check_ec_head_parity.py` 对着原实现逐位核验。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ensure_backbone_importable


# --------------------------------------------------------------------- TM head
def load_tm_head(path, device="cpu"):
    """加载 TM 校准头。返回的是 splm_backbone 里的 TMCalibHead（已 eval）。"""
    ensure_backbone_importable()
    from utils.tm_calib_head import TMCalibHead          # noqa: E402

    head = TMCalibHead.load(str(path), map_location="cpu").to(device).eval()
    return head


@torch.no_grad()
def apply_tm_head(head, raw, device="cpu", drift_n: int = 2000):
    """raw [N,D] (np 或 tensor) -> (emb_h [N,D] numpy, drift)。

    drift = 与原空间的平均余弦偏离（1.0 = 头是恒等变换），纯诊断量。
    """
    import numpy as np

    t = torch.as_tensor(np.asarray(raw)).float().to(device)
    emb = head(t)
    drift = float(head.drift(t[:drift_n]))
    return emb.cpu().numpy(), drift


# --------------------------------------------------------------------- EC head
# heads_lib.DEFAULT_CFG['mlp'] 原值。存档里的 `cfg` 只记**相对默认值的改动**
# （round0 是 `{}`，round19 是 `{"hidden": 512}`），所以必须有这份默认值才能复原结构。
MLP_DEFAULT_CFG = dict(n_blocks=1, hidden=1024, norm="ln_in", act="gelu", dropout=0.0,
                       residual=True, zero_init_last=True, out_dim=1024)

_ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class _MLPBlock(nn.Module):
    """逐字复刻 heads_lib._MLPBlock。"""

    def __init__(self, d: int, h: int, c: dict):
        super().__init__()
        self.ln = nn.LayerNorm(d) if c["norm"] in ("ln_in", "ln_both") else nn.Identity()
        self.fc1 = nn.Linear(d, h)
        self.act = _ACTS[c["act"]]()
        self.drop = nn.Dropout(c["dropout"])
        self.fc2 = nn.Linear(h, d)
        self.residual = c["residual"]
        if c["residual"] and c["zero_init_last"]:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.fc2(self.drop(self.act(self.fc1(self.ln(x)))))
        return x + y if self.residual else y


class ECRetrievalHead(_MLPBlock):
    """EC 检索头（度量学习头）。逐字复刻 heads_lib.MLPHead。

    block 0 就是本模块自己（`ln` / `fc1` / `fc2` 在顶层），n_blocks>1 时其余块在 `more` 里。
    `forward()` 返回**未归一化**的向量（与原实现一致，训练/评估代码在外面做 L2 归一化）；
    要可直接检索的向量用 `encode()`。
    """

    def __init__(self, c: Optional[dict] = None, in_dim: int = 1024,
                 meta: Optional[dict] = None):
        c = {**MLP_DEFAULT_CFG, **(c or {})}
        unknown = set(c) - set(MLP_DEFAULT_CFG)
        if unknown:
            raise ValueError(f"cfg 里有不认识的键 {sorted(unknown)}；"
                             f"可用的是 {sorted(MLP_DEFAULT_CFG)}")
        super().__init__(in_dim, int(c["hidden"]), c)
        self.cfg = c
        self.in_dim = int(in_dim)
        self.meta = dict(meta or {})
        if c["n_blocks"] > 1:
            self.more = nn.ModuleList([_MLPBlock(in_dim, int(c["hidden"]), c)
                                       for _ in range(int(c["n_blocks"]) - 1)])
        if c["norm"] in ("ln_out", "ln_both"):
            self.ln_final = nn.LayerNorm(in_dim)
        if int(c["out_dim"]) != in_dim:
            self.out = nn.Linear(in_dim, int(c["out_dim"]))

    @property
    def out_dim(self) -> int:
        return int(self.cfg["out_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_dim] -> [..., out_dim]，**未归一化**。"""
        x = _MLPBlock.forward(self, x.float())
        if hasattr(self, "more"):
            for b in self.more:
                x = b(x)
        if hasattr(self, "ln_final"):
            x = self.ln_final(x)
        if hasattr(self, "out"):
            x = self.out(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """检索用的向量：forward 之后 L2 归一化。"""
        return F.normalize(self.forward(x), dim=-1)

    # -- 存取 --------------------------------------------------------------
    def save(self, path) -> None:
        """与原存档格式兼容：cfg 只记相对默认值的改动。"""
        delta = {k: v for k, v in self.cfg.items() if MLP_DEFAULT_CFG[k] != v}
        torch.save({"state_dict": self.state_dict(), "cfg": delta,
                    "family": "mlp", "seed": self.meta.get("seed"),
                    "train": self.meta.get("train", {}), "meta": self.meta}, str(path))

    @staticmethod
    def load(path, map_location="cpu", strict: bool = True) -> "ECRetrievalHead":
        """加载原 exp_4-1.1 head.pt（或本类 save 的产物）。

        `cfg` 是相对 `MLP_DEFAULT_CFG` 的改动；这里再用 state_dict 的形状交叉校验一遍，
        不一致就报错 —— 结构配错不会抛异常、只会静默产出垃圾向量。
        """
        ck = torch.load(str(path), map_location=map_location)
        if not isinstance(ck, dict) or "state_dict" not in ck:
            raise TypeError(f"{path} 不是可识别的 EC head 存档"
                            f"（期望含 'state_dict' 的 dict）")
        fam = ck.get("family", "mlp")
        if fam != "mlp":
            raise NotImplementedError(
                f"{Path(path).name} 的 family 是 '{fam}'，本 release 只实现了 'mlp'。"
                f"（transformer/lstm/mamba 是残基级头，要的是 [B,L,1024] 输入，不是池化向量）")
        sd = ck["state_dict"]
        cfg = dict(ck.get("cfg") or {})

        # state_dict 形状 -> 结构，与 cfg 交叉校验
        if "fc1.weight" not in sd or "fc2.weight" not in sd:
            raise KeyError(f"{path} 的 state_dict 里没有 fc1/fc2，认不出 MLPHead")
        hidden, in_dim = (int(x) for x in sd["fc1.weight"].shape)
        out_dim = int(sd["out.weight"].shape[0]) if "out.weight" in sd else in_dim
        n_blocks = 1 + sum(1 for k in sd if k.startswith("more.") and k.endswith(".fc1.weight"))
        for key, inferred in (("hidden", hidden), ("out_dim", out_dim), ("n_blocks", n_blocks)):
            want = cfg.get(key, MLP_DEFAULT_CFG[key])
            if int(want) != int(inferred):
                raise RuntimeError(
                    f"{Path(path).name}: cfg 说 {key}={want}，但权重形状是 {inferred}。"
                    f"存档不自洽，拒绝加载。")
        norm = cfg.get("norm", MLP_DEFAULT_CFG["norm"])
        if ("ln.weight" in sd) != (norm in ("ln_in", "ln_both")):
            raise RuntimeError(
                f"{Path(path).name}: cfg 说 norm={norm!r}，但 state_dict 里"
                f"{'有' if 'ln.weight' in sd else '没有'} ln.weight —— 存档不自洽")

        head = ECRetrievalHead(cfg, in_dim=in_dim,
                               meta={"source": str(path), "seed": ck.get("seed"),
                                     "train": ck.get("train", {}),
                                     **(ck.get("meta") or {})})
        head.load_state_dict(sd, strict=strict)
        head.eval()
        return head


# 项目里说「EC head」指的就是这个检索头
ECHead = ECRetrievalHead


@torch.no_grad()
def apply_ec_head(head: ECRetrievalHead, emb_h, device="cpu", batch_size: int = 8192):
    """emb_h [N,1024]（TM head 的输出）-> (emb_ec 归一化后, emb_ec_raw 未归一化)，都是 numpy。

    ⚠️ 传进来的必须是 `emb_h` 而不是 `emb_raw` —— 这个头是在 emb_h 上训的。
    """
    import numpy as np

    t = torch.as_tensor(np.asarray(emb_h)).float()
    if t.ndim != 2 or t.shape[1] != head.in_dim:
        raise ValueError(
            f"EC head 期望输入 [N,{head.in_dim}]，拿到 {tuple(t.shape)}。"
            f"（S-PLM-TM 的 emb_h 是 1024 维；1280 维说明 backbone 装成了 ESM2）")
    outs = []
    for k in range(0, t.shape[0], batch_size):
        outs.append(head(t[k:k + batch_size].to(device)).cpu())
    raw = torch.cat(outs, 0)
    return F.normalize(raw, dim=-1).numpy(), raw.numpy()


def ec_head_missing_message(expected_dir) -> str:
    return (
        f"EC head checkpoint not found.\n"
        f"  expected: {expected_dir}/ec_head.pt\n"
        f"  download: hf download deepduoduo/S-PLM-TM --local-dir models/\n"
        f"  see models/DOWNLOAD.md, or set SPLM_TM_EC_HEAD to point elsewhere."
    )


# ------------------------------------------------------------------ align head
class AlignHead(nn.Module):
    """残基级比对头：`z = x + MLP(x)`，可堆叠 n_blocks 个残差块。

    逐字复刻研究仓库里的 AlignHead 实现。
    末层零初始化 => **未训练时等于恒等变换**，所以「不套头」就是这个头的起点，
    两者可以直接对比。
    第 1 块的键名是 `mlp`（与历史 checkpoint 兼容），额外块在 `extra` 里。

    ⚠️ 这个头**没有** dropout / LayerNorm / 输出缩放 —— 全部可调结构只有
    `hidden` 与 `n_blocks` 两个。2026-10-13 的 33 配置扫描正是基于这一点才敢说
    「参数空间已穷尽」（见 models/align_head/README.md）。
    """

    def __init__(self, dim: int = 1024, hidden: int = 1024, n_blocks: int = 1):
        super().__init__()

        def _make():
            m = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
            return m

        self.mlp = _make()
        self.extra = nn.ModuleList([_make() for _ in range(n_blocks - 1)])

    def forward(self, x):
        x = x + self.mlp(x)
        for m in self.extra:
            x = x + m(x)
        return x


def load_align_head(path, device="cpu"):
    """加载比对头。返回 (head, meta)；meta 是存档里除权重以外的全部字段。

    存档是自描述的（带 dim / hidden / head_type / mlp_blocks），不需要外部配置。
    只支持 head_type="mlp" —— 研究仓库里还有 conv / msconv / mixconv / transformer
    几族，但消融结论是纯 MLP 最好，本 release 不复刻其余几族；
    碰到就明确报错，而不是悄悄按 MLP 装（那会 load_state_dict 失败或前向用错感受野）。
    """
    payload = torch.load(str(path), map_location="cpu")
    ht = payload.get("head_type", "mlp")
    if ht != "mlp":
        raise ValueError(
            f"{path} 里的 head_type='{ht}'，本 release 只复刻了 'mlp' 一族。\n"
            f"  其余几族的实现留在研究仓库里。")
    head = AlignHead(dim=payload.get("dim", 1024),
                     hidden=payload.get("hidden", 1024),
                     n_blocks=payload.get("mlp_blocks", 1))
    head.load_state_dict(payload["state"])
    head = head.to(device).eval()
    return head, {k: v for k, v in payload.items() if k != "state"}


@torch.no_grad()
def apply_align_head(head, residues, device="cpu", normalize: bool = True):
    """把头套到残基级特征上。

    residues: `Backbone.encode_residues` 的返回值 —— 每条序列一个 [L, 1024] 数组。
    返回同样形状的 list。

    normalize=True（默认）沿最后一维做 L2 归一化。**这是发布口径的一部分**：
    全部已发布的 MALIDUP / MALISAM 数字都是「套头 → L2 归一化 → 内积」算出来的。
    关掉它内积就不再是余弦，与发布数字不可比。
    """
    out = []
    for a in residues:
        t = torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        if t.ndim != 2:
            raise ValueError(f"每条序列应是 [L, D] 二维，拿到 {tuple(t.shape)}。"
                             f"  比对头吃残基级特征，不是蛋白级 emb_raw/emb_h。")
        z = head(t)
        if normalize:
            z = F.normalize(z, dim=-1)
        out.append(z.cpu().numpy())
    return out


def residue_similarity(z1, z2):
    """两条序列的残基级相似度矩阵 [L1, L2]。

    输入应是 `apply_align_head(..., normalize=True)` 的输出，此时内积即余弦。
    这个矩阵就是喂给比对算法的打分矩阵（发布口径用 CLAlign 的 `align_core`：
    全局 Needleman-Wunsch、gap 罚分 0）。本 release **不内置**比对算法本身 ——
    它属于 CLAlign（MIT 许可），按需从其仓库取 `align_core` / `f1score`。
    """
    return np.asarray(z1, dtype=np.float32) @ np.asarray(z2, dtype=np.float32).T


def align_head_missing_message(expected_dir) -> str:
    return (
        f"比对头的 checkpoint 找不到。\n"
        f"  预期位置：{expected_dir}/align_head.pt\n"
        f"  本 release 的默认头 = 复现已发表表格那一个"
        f"（MALIDUP 0.6341 / MALISAM 0.2718、val top-1 0.5512）；\n"
        f"  `alternatives/` 里还有实测更好的温度 0.07 变体。\n"
        f"  也可以用 SPLM_TM_ALIGN_HEAD=<文件> 指到别处。\n"
        f"  ⚠️ 它吃**残基级**特征（Backbone.encode_residues），不是 emb_raw / emb_h。"
    )


__all__ = ["load_tm_head", "apply_tm_head",
           "ECRetrievalHead", "ECHead", "MLP_DEFAULT_CFG", "apply_ec_head",
           "ec_head_missing_message",
           "AlignHead", "load_align_head", "apply_align_head", "residue_similarity",
           "align_head_missing_message"]
