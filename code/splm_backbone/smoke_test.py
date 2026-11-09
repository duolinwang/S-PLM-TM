# -*- coding: utf-8 -*-
"""ProtT5 线的冒烟测试：能不能 import、能不能前向、维度对不对、adapter 是否真在训。

跑法（集群上）：
    module load miniconda3 && source activate <env>
    export HF_HOME=<your huggingface cache>
    python smoke_test.py

不碰数据集、不训练，只验证模型侧接得通。逐项打印通过/失败，任何一项失败都直接抛。
"""
import os
import sys

import torch
import yaml
from box import Box

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CFG = os.environ.get("SPLM_TM_CONFIG") or os.path.join(
    HERE, os.pardir, os.pardir, "models", "backbone", "splm_tm_backbone.yaml")


def ok(msg):
    print(f"  [OK]   {msg}", flush=True)


def main():
    print("=" * 72)
    print("1) 读 config")
    cfg = Box(yaml.full_load(open(CFG, encoding="utf-8")))
    enc = cfg.model.seq_encoder
    ok(f"model_name={enc.model_name}  pool_mode={enc.pool_mode}  "
       f"adapter={enc.adapter_h.enable}/{enc.adapter_h.num_end_adapter_layers} 层")

    print("2) import 模型模块（会暴露残留的 ESM 依赖）")
    from protT5_encoder import ProtT5, T5BatchConverter   # noqa: F401
    import model_v6                                        # noqa: F401
    ok("protT5_encoder / model_v6 均可导入")
    assert model_v6.ESM2 is ProtT5, "model_v6.ESM2 应当是 ProtT5 的别名"
    ok("model_v6.ESM2 已指向 ProtT5")

    print("3) 建编码器（首次会从 HF 缓存加载 1.2B 权重，约 1 分钟）")
    net = ProtT5(cfg,
                 residue_inner_dim=enc.residue_inner_dim,
                 residue_out_dim=cfg.model.residue_out_dim,
                 protein_inner_dim=enc.protein_inner_dim,
                 protein_out_dim=cfg.model.protein_out_dim,
                 residue_num_projector=cfg.model.residue_num_projector,
                 protein_num_projector=cfg.model.protein_num_projector)
    ok(f"embed_dim={net.embed_dim}  层数={net.num_layers}  "
       f"pad={net.padding_idx} eos={net.eos_idx}")
    assert net.embed_dim == 1024, f"ProtT5 应当是 1024 维，实际 {net.embed_dim}"

    print("4) 参数冻结检查：只有 adapter + 投影头可训")
    total = sum(p.numel() for p in net.parameters())
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    t5_train = sum(p.numel() for n, p in net.named_parameters()
                   if p.requires_grad and n.startswith("t5.") and ".adapters." not in n)
    ad = sum(p.numel() for n, p in net.named_parameters() if ".adapters." in n)
    print(f"       总参数 {total/1e6:.1f}M / 可训 {train/1e6:.2f}M "
          f"(adapter {ad/1e6:.2f}M)")
    assert t5_train == 0, f"T5 主干应当全冻结，却有 {t5_train} 个可训参数"
    assert ad > 0, "adapter 没有可训练参数 —— 注入失败"
    ok("T5 主干全冻结；adapter 可训")

    print("5) 前向：三条长短不一的序列")
    bc = T5BatchConverter(net.tokenizer, max_length=enc.max_length,
                          prefix=enc.get("prefix"))
    seqs = [("p1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"),
            ("p2", "MSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS"),
            ("p3", "MKV")]
    labels, strs, toks = bc(seqs)
    ok(f"tokens shape={tuple(toks.shape)}  pad_id={bc.padding_idx} eos_id={bc.eos_idx}")
    with torch.no_grad():
        g, r = net(toks)
    ok(f"protein 级 {tuple(g.shape)}（期望 [3,{cfg.model.protein_out_dim}]）")
    ok(f"残基级   {tuple(r.shape)}（期望 [氨基酸总数,{cfg.model.residue_out_dim}]）")
    assert g.shape == (3, cfg.model.protein_out_dim)
    n_aa = sum(len(s) for _, s in seqs)
    assert r.shape[0] == n_aa, f"残基数应为 {n_aa}，实际 {r.shape[0]}（说明 mask 排错了）"
    assert r.shape[1] == cfg.model.residue_out_dim
    ok(f"残基数 {r.shape[0]} == 输入氨基酸总数 {n_aa}，pad/eos 已正确排除")

    print("6) 掩码增强（seq2seq BYOL 用）")
    from model_v6 import MaskedLMDataCollator
    coll = MaskedLMDataCollator(bc, mlm_probability=enc.MLM.mask_ratio)
    mi, ml = coll.mask_tokens(toks)
    changed = int((mi != toks).sum())
    ok(f"掩码 token 用的是 <extra_id_0>(id={bc.alphabet.mask_idx})，本批改动 {changed} 个位置")
    assert changed > 0, "一个位置都没被掩掉，掩码逻辑没生效"

    print("7) 反向：确认梯度真的流到 adapter")
    g, r = net(toks)
    (g.square().mean() + r.square().mean()).backward()
    with_grad = [n for n, p in net.named_parameters()
                 if ".adapters." in n and p.grad is not None and p.grad.abs().sum() > 0]
    ok(f"{len(with_grad)} 个 adapter 参数拿到非零梯度，例如 {with_grad[0]}")
    assert with_grad, "adapter 没有梯度 —— forward hook 没接上"

    print("=" * 72)
    print("冒烟测试全部通过。下一步可以接数据集跑训练。")


if __name__ == "__main__":
    main()
