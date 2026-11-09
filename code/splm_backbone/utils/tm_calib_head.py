"""TM 校准头：架在**冻结** SPLM 检索向量之上的小 MLP，用真实 TM 分数训练。

用途有二，共用同一个产物：
  1) 证伪测试。SCOPE full 模式的失败取证已排除 hubness/各向异性（中心化≈raw、
     CSLS≈raw、target-z 更差；583 次失败对应 ≥557 个不同的抢跑者）。还剩最后一种
     可能：信息在 1280 维向量里，只是余弦这种固定读法读不出来。本头是**最强的
     后处理**——学出来的非线性重加权。它若仍失败，则"后处理"整类方法出局，
     必须重训 backbone。
  2) 若成功，它本身就是两段式检索的 stage-1 召回向量（纯 TM 预测器正好干召回），
     精排仍由原始 1280 维 embedding 负责 —— 原 checkpoint 一个字节不动。

设计要点：**零初始化残差 + 固定随机正交投影**

    out = normalize( P0 @ x + alpha * MLP(x) ),   MLP 末层权重零初始化

P0 是固定的随机正交矩阵（不训练）。由 Johnson–Lindenstrauss，随机正交投影近似
保持余弦，所以 **训练起点的检索性能 ≈ baseline**，之后只会变好；alpha 的大小可
直接读出"为了拟合 TM，表示被迫偏离了多远"。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TMCalibHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden: int = 1024, out_dim: int | None = None,
                 n_layers: int = 2, dropout: float = 0.0, seed: int = 0,
                 learn_affine: bool = True):
        super().__init__()
        # ⚠️ 默认 out_dim = in_dim（P0 = 单位阵）→ **起点与原 embedding 逐位等价**。
        # 降维版（out_dim < in_dim）用固定随机正交投影，但要小心：1280→512 的 JL 畸变
        # 实测约 0.027，而各向同性向量的余弦谱宽本身也只有 1/sqrt(1280)≈0.028 —— 畸变
        # 会淹掉信号（实测 spearman 掉到 0.61）。只有当真实余弦谱宽 >> 畸变时才可降维。
        # 做证伪测试时**务必用默认值**，否则分不清"头没救回来"和"投影搞坏了"。
        out_dim = int(in_dim if out_dim is None else out_dim)
        self.in_dim, self.out_dim = int(in_dim), out_dim

        if out_dim == in_dim:
            self.register_buffer("P0", torch.eye(in_dim))
        else:
            g = torch.Generator().manual_seed(int(seed))
            a = torch.randn(in_dim, out_dim, generator=g)
            q, _ = torch.linalg.qr(a)                   # [in_dim, out_dim] 列正交
            self.register_buffer("P0", q.contiguous())

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        last = nn.Linear(d, out_dim)
        nn.init.zeros_(last.weight)                     # 零初始化 → 起点 = P0 投影
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.ones(()))

        # pointwise 标定用的可学仿射：pred = scale * cos + bias
        if learn_affine:
            self.scale = nn.Parameter(torch.ones(()))
            self.bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("scale", torch.ones(()))
            self.register_buffer("bias", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] 冻结的 backbone 检索向量 → [..., out_dim] 已 L2 归一化。"""
        base = x.float() @ self.P0                      # [..., out_dim]
        return F.normalize(base + self.alpha * self.mlp(x.float()), dim=-1)

    @torch.no_grad()
    def drift(self, x: torch.Tensor) -> float:
        """诊断：新旧空间的平均余弦偏离（1.0 = 完全没动）。"""
        base = F.normalize(x.float() @ self.P0, dim=-1)
        return float((base * self.forward(x)).sum(-1).mean())

    def save(self, path) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "cfg": {"in_dim": self.in_dim, "out_dim": self.out_dim},
        }, str(path))

    @staticmethod
    def load(path, map_location="cpu") -> "TMCalibHead":
        ck = torch.load(str(path), map_location=map_location)
        sd = ck["state_dict"]
        in_dim, out_dim = ck["cfg"]["in_dim"], ck["cfg"]["out_dim"]
        # 从 state_dict 反推 hidden / n_layers，避免调用方重复记超参
        lin_w = [v for k, v in sd.items() if k.startswith("mlp.") and k.endswith(".weight")]
        n_layers = len(lin_w)
        hidden = lin_w[0].shape[0] if n_layers > 1 else out_dim
        head = TMCalibHead(in_dim=in_dim, hidden=hidden, out_dim=out_dim, n_layers=n_layers)
        head.load_state_dict(sd)
        head.eval()
        return head
