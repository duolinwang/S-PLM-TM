# -*- coding: utf-8 -*-
"""ProtT5 / ProstT5 序列编码器 —— 替换原来的 ESM2 分支。

本目录是 ProtT5 专用，与 ESM2 无关，所以这里**不保留** ESM2 的任何代码路径。

与 ESM2 版的关键差异（都是 T5 的架构事实，不是选择）：
  · 只用 encoder：`T5EncoderModel` 直接丢掉 decoder。做表征用不上 decoder，
    省一半显存 —— Rostlab 官方 embedding 示例也是这么做的。
  · d_model = **1024**（ESM2 650M 是 1280）。全仓的 1280 都要跟着改。
  · 分词是 sentencepiece，**没有 cls token**；序列末尾追加一个 `</s>`(eos)。
    所以 mask 只需要排掉 pad 与 eos，比 ESM 那套 cls/eos/pad 三件套简单。
  · 输入要求残基之间**加空格**（ProtT5 的既定用法），且 U/Z/O/B 统一映射成 X。
  · ProstT5 额外要一个方向前缀：氨基酸序列用 `<AA2fold>`。

池化口径（与 ESM2 版的一处已知问题不同，这里按正确的来）：
  ESM2 版 pool_mode="2" 走的是「只排 padding」的分支，**cls/eos 也参与平均**。
  本目录不继承那个行为 —— pool_mode="2" 就是在真实氨基酸上取平均（排 pad 与 eos）。
  这是新的一条模型线，本来就不与 ESM2 逐 token 可比，没有必要复制那个缺陷。
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

from t5_adapter import inject_adapters

# 非标准残基统一成 X，ProtT5 的既定预处理
_NONSTD = re.compile(r"[UZOB]")


class _T5Alphabet:
    """把 T5 tokenizer 包装成 ESM `alphabet` 的样子。

    只为一个目的：让 `MaskedLMDataCollator` 一行都不用改就能用 —— 它只从
    batch_converter.alphabet 取 cls/padding/eos/unk/mask 五个索引和词表大小。

    掩码符号用 **T5 自己的 span-corruption 哨兵 `<extra_id_0>`**，不另造 token：
    那个符号在 T5 预训练时被见过千万次，语义就是"这里被挖掉了"，比拿 <unk> 顶
    合适得多。（注意 T5 原生的 span corruption **目标函数**我们用不了 ——
    那是 seq2seq 的、要 decoder 还原；但这里掩码只是给序列侧 BYOL 造增强视图，
    不需要还原目标，所以只借它的符号。）

    T5 没有 cls token，cls_idx 指向 pad —— 效果是它落进 special 集合里被跳过，
    与"没有 cls"是一致的。
    """

    def __init__(self, tokenizer: T5Tokenizer, prefix: str | None = None):
        self.tok = tokenizer
        self.prefix = prefix
        self.padding_idx = tokenizer.pad_token_id
        self.eos_idx = tokenizer.eos_token_id
        self.unk_idx = tokenizer.unk_token_id
        self.cls_idx = tokenizer.pad_token_id          # T5 无 cls
        mask_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        if mask_id is None or mask_id == tokenizer.unk_token_id:
            raise RuntimeError(
                "这个 T5 词表里找不到 <extra_id_0>，无法用它当掩码符号。"
                "ProtT5 的 128 词表 = 25 氨基酸 + 3 特殊 + 100 哨兵，本该有；"
                "请先确认加载的是 Rostlab 的权重。")
        self.mask_idx = mask_id
        self.all_toks = list(tokenizer.get_vocab().keys())

    def get_batch_converter(self, truncation_seq_length=None):
        """与 ESM alphabet 同名同签名 —— 训练脚本里
        `alphabet.get_batch_converter(truncation_seq_length=...)` 因此不用改。
        """
        return T5BatchConverter(self.tok, max_length=truncation_seq_length,
                                prefix=self.prefix)


class T5BatchConverter:
    """ESM `alphabet.get_batch_converter()` 的等价替代（同签名，drop-in）。

    调用：converter([(label, seq), ...]) -> (labels, strs, tokens[B, L])
    训练循环里就是这么用的，所以签名必须一致。
    """

    def __init__(self, tokenizer: T5Tokenizer, max_length: int | None = None,
                 prefix: str | None = None):
        self.tok = tokenizer
        self.max_length = max_length          # None = 不截断
        self.prefix = prefix                  # ProstT5 用 "<AA2fold>"；ProtT5 为 None
        self.padding_idx = tokenizer.pad_token_id
        self.eos_idx = tokenizer.eos_token_id
        self.alphabet = _T5Alphabet(tokenizer, prefix)  # 供 MaskedLMDataCollator 使用

    def _prep(self, seq: str) -> str:
        s = _NONSTD.sub("X", str(seq).upper())
        if self.max_length is not None:
            s = s[: self.max_length]
        s = " ".join(list(s))                 # ProtT5 要求残基间加空格
        return f"{self.prefix} {s}" if self.prefix else s

    def __call__(self, raw_batch):
        labels = [b[0] for b in raw_batch]
        strs = [str(b[1]) for b in raw_batch]
        enc = self.tok.batch_encode_plus([self._prep(s) for s in strs],
                                         add_special_tokens=True, padding="longest",
                                         return_tensors="pt")
        return labels, strs, enc["input_ids"]


class MoBYMLP(nn.Module):
    """与 ESM2 版同款投影头（逐字照搬，避免两条线在投影结构上产生额外差异）。"""

    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super().__init__()
        linear_layers = []
        for _ in range(num_layers - 1):
            linear_layers.extend([nn.Linear(in_dim, inner_dim, bias=False),
                                  nn.BatchNorm1d(inner_dim), nn.ReLU(inplace=True)])
            in_dim = inner_dim
        if num_layers > 0:
            linear_layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*linear_layers) if linear_layers else nn.Identity()

    def forward(self, x):
        return self.mlp(x)


class ProtT5(nn.Module):
    """冻结的 T5 编码器 + 可训练 adapter + 残基/蛋白两个投影头。

    forward 返回 (graph_feature, residue_feature)，与原 ESM2 类一致，
    使得按名字实例化的调用点不用改。
    """

    def __init__(self, configs, logging=None,
                 residue_inner_dim=None, residue_out_dim=64,
                 protein_inner_dim=None, protein_out_dim=256,
                 residue_num_projector=2, protein_num_projector=2):
        super().__init__()
        enc_cfg = configs.model.seq_encoder
        name = enc_cfg.model_name
        # 两个 inner_dim 没传就从 config 读。
        # ⚠️ ESM2 那版的三个调用点都**没传** protein_inner_dim，靠的是函数默认值 4096，
        #    于是 config 里的 protein_inner_dim 实际是个摆设 —— 改了不生效。
        #    本目录不继承这个坑：以 config 为准。
        if residue_inner_dim is None:
            residue_inner_dim = enc_cfg.residue_inner_dim
        if protein_inner_dim is None:
            protein_inner_dim = enc_cfg.protein_inner_dim

        self.tokenizer = T5Tokenizer.from_pretrained(name, do_lower_case=False,
                                                     legacy=True)
        self.t5 = T5EncoderModel.from_pretrained(name)     # 只有 encoder
        self.embed_dim = self.t5.config.d_model            # 1024
        self.num_layers = self.t5.config.num_layers        # 24
        self.padding_idx = self.tokenizer.pad_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        self.pool_mode = str(enc_cfg.pool_mode)
        # ProstT5 要方向前缀 "<AA2fold>"；ProtT5 为 None
        self._prefix = enc_cfg.get("prefix") if hasattr(enc_cfg, "get") else None
        # 训练脚本用 esm2_net.alphabet.get_batch_converter(truncation_seq_length=...)
        # 拿分词器，这里提供同名接口，训练脚本因此不用改。
        self.alphabet = _T5Alphabet(self.tokenizer, self._prefix)

        # 主干全冻结；只有 adapter 参与训练（与 ESM2 版策略一致）
        for p in self.t5.parameters():
            p.requires_grad = False

        # 梯度检查点：T5 的注意力把 scores 转成 fp32 再 softmax
        # （transformers/models/t5/modeling_t5.py:552，即使开了 fp16 也这样），
        # 于是注意力矩阵按 fp32 存：B×heads×L×L×4 字节。
        # batch=20、32 头、L=512 时就是 671 MB/层，而编码器每步要前向多次
        # （干净视图 + 掩码视图 + byol 两个方向），80GB 的 A100 直接爆。
        # 开检查点后前向不留中间激活、反向时重算，显存大降、慢约 30%，
        # 但能保住 batch_size=20 —— 那个数决定 SimCLR 的负样本数，
        # 动了就没法和 ESM2 线单变量对比了。
        if enc_cfg.get("gradient_checkpointing", False):
            self.t5.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.t5.config.use_cache = False
            msg = "T5 编码器已开启梯度检查点（省显存，约慢 30%）"
            (logging.info if logging else print)(msg)

        self.adapters = None
        if enc_cfg.adapter_h.enable:
            self.adapters = inject_adapters(self.t5.encoder, enc_cfg.adapter_h,
                                            self.embed_dim, logging=logging)
            for p in self.adapters.parameters():
                p.requires_grad = True
        else:
            msg = "⚠️ adapter 未启用 —— 序列侧将完全没有可训练参数"
            (logging.warning if logging else print)(msg)

        self.projectors_residue = MoBYMLP(in_dim=self.embed_dim,
                                          inner_dim=residue_inner_dim,
                                          num_layers=residue_num_projector,
                                          out_dim=residue_out_dim)
        self.projectors_protein = MoBYMLP(in_dim=self.embed_dim,
                                          inner_dim=protein_inner_dim,
                                          num_layers=protein_num_projector,
                                          out_dim=protein_out_dim)

    def get_batch_converter(self, max_length=None):
        return self.alphabet.get_batch_converter(truncation_seq_length=max_length)

    def forward(self, x, return_embedding=False, return_logits=False):
        """x: [B, L] token id。返回 (protein级, 残基级[, 未投影的两份])。

        return_logits 只为兼容调用方签名（SimCLR.forward / crossflow 都会传）。
        我们只加载了 T5 的 encoder，没有 LM head，传 True 直接报错 ——
        本目录 MLM.enable 恒为 False，正常路径不会走到这里。
        """
        if return_logits:
            raise NotImplementedError(
                "ProtT5 只加载了 encoder，没有 LM head，无法返回 logits。"
                "本目录的 MLM.enable 应保持 False；掩码只用作 BYOL 的增强视图。")
        attn = (x != self.padding_idx).long()
        h = self.t5(input_ids=x, attention_mask=attn).last_hidden_state  # [B,L,1024]

        # 真实氨基酸的位置：排掉 pad 与 eos（T5 没有 cls）
        aa = (x != self.padding_idx) & (x != self.eos_idx)

        if self.pool_mode == "AA" or self.pool_mode == "2":
            denom = aa.sum(-1, keepdim=True).clamp(min=1)
            pooled = (h * aa.unsqueeze(-1)).sum(dim=1) / denom
        else:
            raise ValueError(
                f"本目录只支持 pool_mode '2'/'AA'（真实氨基酸上的均值池化）；"
                f"收到 {self.pool_mode!r}。SequenceSimGNN 那几条线属于 ESM2 的历史设置，"
                f"未移植。")

        graph_feature_embedding = pooled
        residue_feature_embedding = h[aa]                 # [N_res, 1024]
        graph_feature = self.projectors_protein(graph_feature_embedding)
        residue_feature = self.projectors_residue(residue_feature_embedding)
        if return_embedding:
            return (graph_feature, residue_feature,
                    graph_feature_embedding, residue_feature_embedding)
        return graph_feature, residue_feature
