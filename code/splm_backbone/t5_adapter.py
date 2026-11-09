# -*- coding: utf-8 -*-
"""给 HuggingFace T5 编码器注入 bottleneck adapter。

对应 ESM2 那边 `esm_adapterH` 的功能，但**不是移植** —— `esm_adapterH` 是 ESM 源码的
一个 fork（adapter 直接焊在它自己的 TransformerLayer 里），对 T5 完全不可复用。
这里改用 forward hook 的方式包住 T5 的模块，好处是不碰 transformers 的源码，
换 transformers 版本也不会碎。

两种注入位置（config 键 `inject`，**缺省 = "block_out" = 2026-09-14 之前的唯一行为**）：

  inject: "block_out"   （旧，默认）在最后 N 个 T5Block 的【输出 = 整条残差流】上并联：
        out = block(x)                     # T5 自己的 self-attn + FFN
        out = out + adapter(out)
      "每层 n_per_layer 个" = 同一位置的 n 个并联副本。

  inject: "sublayer"    （2026-09-14 新增）对齐 esm_adapterH/modules.py:186-226 的真实位置——
      adapter 作用在每个【子层的增量】上、残差相加之前：
        attn_delta = SelfAttention(LN(x));  attn_delta = attn_delta + adapter_a(attn_delta)
        x = x + attn_delta
        ffn_delta  = DenseReluDense(LN(x)); ffn_delta  = ffn_delta  + adapter_f(ffn_delta)
        x = x + ffn_delta
      "每层 n_per_layer 个" = **每个位置** n 个；n_per_layer=1 时每个 T5Block 共 2 个
      adapter（attn 后 + FFN 后），这才是 ESM2 版 `for _ in range(2)` 的本义。
      为什么要有这一档（2026-09-14 三条诊断）：① 两条 T5 臂学到的 gain 都把浅层压到
      0.13-0.24、深层留 0.3-0.5；② T5 残差流 RMS 在被 adapt 的 12 层内从 22.6 涨到 107.8
      （ESM2 同区间只涨 1.3 倍），按残差流 RMS 缩放的 adapter 扰动随深度膨胀；
      ③ ESM2 版从来不是改残差流，而是改子层增量。block_out 是"参数比例对齐、位置没对齐"。
      ⚠️ 2026-09-14 实跑（A 臂 17389664）：此档 scale_to_input 按【子层增量】的 RMS 缩放，而 T5 的
      子层增量相对残差流（22~108）很小 → adapter 绝对扰动极小，4k 步时检索 ≈ raw T5、gain 被模型往上推。
      位置对了、幅度参照系错了。

  inject: "sublayer_res" （2026-09-14 晚新增）位置同 sublayer（作用在子层增量上），但幅度参照 =
      该子层【输入的残差流】x 的 RMS（与 block_out 同一参照系，gain 0.44 才是 ESM2 那档的含义）。
      实现上包住整个 T5LayerSelfAttention / T5LayerFF：y = layer(x)，delta = y - x，
      y' = y + adapter(delta, ref=x)。

多路 adapter：ESM2 版支持 `num_end_adapter_layers` 传 list（如 [3,2,3]）来放置
多组独立 adapter。这里同样支持，第 i 组叫 `adapter_i`，各组的层范围独立。

只有 adapter 的参数 requires_grad=True，T5 主干全冻结 —— 与 ESM2 版一致。

⚠️ 参数命名约定（`ckpt_slim.is_frozen_backbone_key` 依赖它）：所有 adapter 参数的 key
都含 `.adapters.`，两种模式都如此；被包住的原模块改挂在 `.block` / `.inner` 下面。
"""
from __future__ import annotations

import torch
import torch.nn as nn

INJECT_MODES = ("block_out", "sublayer", "sublayer_res")


class Bottleneck(nn.Module):
    """down-project -> 非线性 -> up-project。两种初始化风格，由 esm2_style 选择。

    style="zero"（默认，旧行为）：up 零初始化，adapter 输出恒为 0，起点等价于原始 T5。
      ⚠️ 2026-09-09 实测：这一档在 ProtT5 线上**训不动**——26000 步后 up 层 |W|max 仅
      0.0013（默认初始化该是 0.125），adapter 开/关的 features_seq cosine = 1.000000，
      相对改变 0.04%，即 adapter 是 no-op、序列塔等于冻结 raw T5，CATH ARI 全程 0.02-0.04。
      机理：up=0 时 down/ln 的梯度被链式法则掐断（实测 0.000e+00），只有 up.W 本身有梯度；
      序列侧梯度一弱就永远爬不出零点。（原注释称"ESM2 版同样的做法"是错的，见下。）

    style="esm2"：对齐 esm_adapterH/ResMLP 的真实语义——
      Linear -> ReLU -> Linear -> (Dropout) -> LayerNorm，**默认初始化、不置零**，
      输出经 LayerNorm 后是 O(1) 量级，第一步起就实质参与表示。ESM2 线（CATH ARI
      0.067 -> 0.391）用的就是这一档。
    """

    def __init__(self, d_model: int, bottleneck: int, dropout: float = 0.1,
                 module_type: str = "MLP1", esm2_style: bool = False,
                 scale_to_input: bool = False, gain_init: float = 0.44):
        super().__init__()
        self.module_type = module_type
        self.esm2_style = esm2_style
        # scale_to_input：把 adapter 输出按**输入自身的 RMS**等比放大，使"每层相对影响力"
        # 与骨干无关。动机（2026-09-11 实测）：ESM2 隐状态 RMS=2.16、adapter 输出 RMS=0.95
        # → 相对影响力 0.441；同一套代码搬到 T5 上，隐状态 RMS=68.4 而 adapter 输出仍被
        # LayerNorm 钉在 0.99 → 相对影响力只有 0.0144，**弱 31 倍**。这正是 adapter 换成
        # esm2 初始化、容量从 1.6M 提到 25M 后 CATH ARI 仍无起色的原因：瓶颈是幅度不是容量。
        # gain 可学，初值 0.44 = ESM2 线的实测比值。
        # 注意"输入"指 adapter 实际读到的张量：block_out 模式是残差流，sublayer 模式是子层增量。
        self.scale_to_input = scale_to_input
        if scale_to_input:
            self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        # esm2 档不做输入 LayerNorm（ESM2 版 ResMLP 没有），用 Identity 保持 forward 统一
        self.ln = nn.Identity() if esm2_style else nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.ReLU() if esm2_style else nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, d_model)
        if esm2_style:
            self.out_ln = nn.LayerNorm(d_model)
        else:
            self.out_ln = nn.Identity()
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        if module_type == "MLP2":          # 更深的一档，与 ESM2 版的 MLP2 对应
            self.mid = nn.Linear(bottleneck, bottleneck)
        else:
            self.mid = None

    def forward(self, x, ref=None):
        """ref：scale_to_input 的幅度参照张量；缺省 None = 用 x 自己（2026-09-14 前的唯一行为）。
        sublayer_res 模式传入残差流 x，让"相对影响力"与 block_out 模式同一口径。"""
        h = self.down(self.ln(x))
        h = self.act(h)
        if self.mid is not None:
            h = self.act(self.mid(h))
        out = self.out_ln(self.up(self.drop(h)))
        if self.scale_to_input:
            # 逐 token 的 RMS：T5 各层幅度差异很大，用每处的实际尺度而不是全局常数
            src = x if ref is None else ref
            x_rms = src.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(out.dtype)
            out = out * (x_rms * self.gain)
        return out


class _BlockWithAdapters(nn.Module):
    """【block_out 模式】包住一个 T5Block，在它输出上并联若干 adapter。

    T5Block.forward 返回的是 tuple，第 0 项才是 hidden_states，其余是 attention
    缓存等；只改第 0 项，其余原样透传 —— 否则 T5 的 position_bias 传递会断。
    """

    def __init__(self, block: nn.Module, adapters: nn.ModuleList):
        super().__init__()
        self.block = block
        self.adapters = adapters

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            h = out[0]
            for ad in self.adapters:
                h = h + ad(h)
            return (h,) + out[1:]
        for ad in self.adapters:
            out = out + ad(out)
        return out


class _SublayerWithAdapters(nn.Module):
    """【sublayer 模式】包住 T5 的一个子层核心（SelfAttention 或 DenseReluDense），
    在它的输出（= 该子层对残差流的增量，尚未与残差相加）上并联若干 adapter。

    T5LayerSelfAttention.forward（transformers 4.48）：
        attention_output = self.SelfAttention(LN(x), ...)      # tuple，[0] 是增量
        x = x + dropout(attention_output[0])
    T5LayerFF.forward：
        forwarded = self.DenseReluDense(LN(x))                 # tensor，就是增量
        x = x + dropout(forwarded)
    所以包住 `SelfAttention` / `DenseReluDense` 就恰好落在 ESM2 版 adapter 的位置上。
    tuple 输出只改第 0 项，其余（position_bias / attention weights）原样透传。
    """

    def __init__(self, inner: nn.Module, adapters: nn.ModuleList):
        super().__init__()
        self.inner = inner
        self.adapters = adapters

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            h = out[0]
            for ad in self.adapters:
                h = h + ad(h)
            return (h,) + out[1:]
        for ad in self.adapters:
            out = out + ad(out)
        return out


class _LayerWithDeltaAdapters(nn.Module):
    """【sublayer_res 模式】包住整个 T5LayerSelfAttention / T5LayerFF。

    这两个模块的 forward 都是  y = x + dropout(sublayer(LN(x)))，返回 tensor 或 (y, ...) tuple。
    这里取 delta = y - x（= 该子层对残差流的增量），adapter 作用在 delta 上、
    幅度参照 = x（残差流）的 RMS：  y' = y + adapter(delta, ref=x)。
    与 ESM2 版位置一致（改增量、在残差相加之后等价于相加之前），幅度口径与 block_out 一致。
    """

    def __init__(self, inner: nn.Module, adapters: nn.ModuleList):
        super().__init__()
        self.inner = inner
        self.adapters = adapters

    def forward(self, hidden_states, *args, **kwargs):
        out = self.inner(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            y, rest = out[0], out[1:]
        else:
            y, rest = out, None
        delta = y - hidden_states
        for ad in self.adapters:
            y = y + ad(delta, ref=hidden_states)
        return (y,) + rest if rest is not None else y


def _new_adapters(n: int, d_model: int, bottleneck: int, dropout: float,
                  module_type: str, esm2_style: bool, scale_to_input: bool,
                  gain_init: float) -> list:
    return [Bottleneck(d_model, bottleneck, dropout, module_type,
                       esm2_style=esm2_style, scale_to_input=scale_to_input,
                       gain_init=gain_init) for _ in range(n)]


def inject_adapters(t5_encoder: nn.Module, adapter_args, d_model: int,
                    logging=None) -> nn.ModuleList:
    """把 adapter 注入 T5 编码器的后若干层，返回所有 adapter 的 ModuleList。

    adapter_args 需要的字段（与 ESM2 版 config 同名，便于对照）：
      num_end_adapter_layers : int 或 list[int]，每组 adapter 覆盖最后多少层
      module_type            : "MLP1" | "MLP2"
      bottleneck             : 中间维度（ESM2 版没有这个键、恒为 d_model//2；T5 侧显式给出）
      dropout                : adapter 内部 dropout
      esm2_style             : 可选，默认 False=旧行为（up 零初始化）。True 时对齐
                               esm_adapterH 的真实语义（默认初始化 + 输出 LayerNorm），
                               修 2026-09-09 查实的"adapter 恒为 no-op"问题
      n_per_layer            : 可选，默认 1。block_out 模式=每层几个并联副本；
                               sublayer 模式=每个位置（attn 后 / FFN 后）几个
      scale_to_input         : 可选，默认 False=旧行为。True 时 adapter 输出按输入 RMS
                               等比缩放（gain 可学），修 T5 隐状态 RMS 68 vs ESM2 2.2
                               造成的"相对影响力弱 31 倍"问题
      gain_init              : 可选，默认 0.44（= ESM2 线实测的 adapter/hidden 比值）
      inject                 : 可选，默认 "block_out"（旧行为，逐字不变）。
                               "sublayer" = ESM2 对齐位置，见文件头
    """
    blocks = t5_encoder.block                      # nn.ModuleList[T5Block]
    n_layer = len(blocks)

    spec = adapter_args.num_end_adapter_layers
    if not isinstance(spec, (list, tuple)):
        spec = [spec]
    bottleneck = getattr(adapter_args, "bottleneck", 64)
    module_type = getattr(adapter_args, "module_type", "MLP1")
    dropout = getattr(adapter_args, "dropout", 0.1)
    esm2_style = bool(getattr(adapter_args, "esm2_style", False))
    n_per_layer = int(getattr(adapter_args, "n_per_layer", 1))
    scale_to_input = bool(getattr(adapter_args, "scale_to_input", False))
    gain_init = float(getattr(adapter_args, "gain_init", 0.44))
    inject = str(getattr(adapter_args, "inject", "block_out"))
    if inject not in INJECT_MODES:
        raise ValueError(f"adapter_h.inject 只能是 {INJECT_MODES}，收到 {inject!r}")
    kw = dict(d_model=d_model, bottleneck=bottleneck, dropout=dropout,
              module_type=module_type, esm2_style=esm2_style,
              scale_to_input=scale_to_input, gain_init=gain_init)

    # layer_idx -> [该层要挂的 adapter]（block_out）
    # layer_idx -> ([attn 后的], [FFN 后的])（sublayer）
    per_layer = {i: [] for i in range(n_layer)}
    per_layer_sub = {i: ([], []) for i in range(n_layer)}
    all_adapters = nn.ModuleList()
    for gi, n_end in enumerate(spec):
        n_end = int(n_end)
        if n_end <= 0:
            continue
        start = max(0, n_layer - n_end)
        for li in range(start, n_layer):
            if inject == "block_out":
                ads = _new_adapters(n_per_layer, **kw)
                per_layer[li].extend(ads)
                all_adapters.extend(ads)
            else:   # sublayer / sublayer_res：每个 block 两个位置各 n 个
                ads_a = _new_adapters(n_per_layer, **kw)
                ads_f = _new_adapters(n_per_layer, **kw)
                per_layer_sub[li][0].extend(ads_a)
                per_layer_sub[li][1].extend(ads_f)
                all_adapters.extend(ads_a)
                all_adapters.extend(ads_f)
        if logging is not None:
            logging.info(f"adapter_{gi}: 覆盖第 {start}~{n_layer-1} 层（共 {n_end} 层）")

    n_layers_with = 0
    if inject == "block_out":
        for li, ads in per_layer.items():
            if ads:
                blocks[li] = _BlockWithAdapters(blocks[li], nn.ModuleList(ads))
                n_layers_with += 1
    else:
        for li, (ads_a, ads_f) in per_layer_sub.items():
            if not (ads_a or ads_f):
                continue
            blk = blocks[li]
            # T5Block.layer[0] = T5LayerSelfAttention（子模块 SelfAttention）
            # T5Block.layer[1] = T5LayerFF（子模块 DenseReluDense）；解码器才有 layer[2]
            attn_layer, ff_layer = blk.layer[0], blk.layer[-1]
            if not hasattr(attn_layer, "SelfAttention") or not hasattr(ff_layer, "DenseReluDense"):
                raise RuntimeError(
                    f"T5Block[{li}] 的子层结构不是预期的 SelfAttention/DenseReluDense："
                    f"{[type(l).__name__ for l in blk.layer]} —— transformers 版本变了？")
            if inject == "sublayer":
                attn_layer.SelfAttention = _SublayerWithAdapters(attn_layer.SelfAttention,
                                                                 nn.ModuleList(ads_a))
                ff_layer.DenseReluDense = _SublayerWithAdapters(ff_layer.DenseReluDense,
                                                                nn.ModuleList(ads_f))
            else:   # sublayer_res：包整个子层，拿得到残差流 x 与增量 delta
                blk.layer[0] = _LayerWithDeltaAdapters(attn_layer, nn.ModuleList(ads_a))
                blk.layer[-1] = _LayerWithDeltaAdapters(ff_layer, nn.ModuleList(ads_f))
            n_layers_with += 1

    n_param = sum(p.numel() for p in all_adapters.parameters())
    per_layer_txt = (f"每层 {n_per_layer} 个" if inject == "block_out"
                     else f"每层 {2 * n_per_layer} 个（attn 后 {n_per_layer} + FFN 后 {n_per_layer}）")
    msg = (f"T5 adapter 注入完成：{n_layer} 层里有 {n_layers_with} 层带 adapter，"
           f"{per_layer_txt}，bottleneck={bottleneck}，"
           f"init={'esm2(默认初始化+输出LN)' if esm2_style else 'zero(旧行为)'}，"
           f"scale_to_input={scale_to_input}"
           + (f"(gain_init={gain_init})" if scale_to_input else "")
           + (f"，inject={inject}" if inject != "block_out" else "")
           + f"，可训练参数 {n_param/1e6:.2f}M")
    if logging is not None:
        logging.info(msg)
    else:
        print(msg, flush=True)
    return all_adapters
