# -*- coding: utf-8 -*-
"""ckpt_slim 的本地 CPU 验证（不需要集群、不需要真权重）。

用 2 层随机小 T5 走一遍「保存瘦身存档 -> 新建模型 -> 加载 -> 比对」，验证四件事：
  1. 过滤规则没有误伤 adapter（adapter 名字里也带 t5.，是最容易写错的地方）
  2. 瘦身后加载，**所有可训练参数逐位一致**
  3. 冻结主干仍是 from_pretrained 的原值（没被残缺存档污染）
  4. 缺了可训练参数时会**报错**，而不是静默加载出半个模型

    python test_ckpt_slim.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ok(m):
    print(f"  [OK]   {m}", flush=True)


class _FakeTok:
    """最小分词器替身，让测试完全离线。

    只提供 _T5Alphabet / T5BatchConverter 用到的那几样：pad/eos/unk id、
    `<extra_id_0>` 的 id、词表、以及批量编码。本测试验的是**参数名过滤与加载**，
    与真实分词无关；用替身可以避开 HF 下载和 tiktoken 依赖。
    真实词表的验证在集群的 smoke_test.py 里做过（pad=0 eos=1 extra_id_0=127）。
    """
    pad_token_id, eos_token_id, unk_token_id = 0, 1, 2
    _V = {"<pad>": 0, "</s>": 1, "<unk>": 2, "<extra_id_0>": 127}

    def convert_tokens_to_ids(self, t):
        return self._V.get(t)

    def get_vocab(self):
        return dict(self._V)

    def batch_encode_plus(self, texts, **kw):
        seqs = [t.replace(" ", "") for t in texts]
        L = max(len(s) for s in seqs) + 1
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            for j in range(len(s)):
                ids[i, j] = 3 + (j % 20)          # 任意有效 token id
            ids[i, len(s)] = self.eos_token_id
        return {"input_ids": ids}


def build(cfg):
    """建一个用小 T5 的 ProtT5（固定种子，使两次构造的主干初值一致）。"""
    import transformers
    import protT5_encoder
    from transformers import T5Config, T5EncoderModel

    tiny = T5Config(d_model=64, d_ff=128, num_layers=4, num_heads=2,
                    num_decoder_layers=0, vocab_size=128)
    torch.manual_seed(1234)
    transformers.T5EncoderModel.from_pretrained = staticmethod(
        lambda n, *a, **k: T5EncoderModel(tiny))
    protT5_encoder.T5EncoderModel = transformers.T5EncoderModel
    protT5_encoder.T5Tokenizer = type(
        "T5TokenizerStub", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeTok())})
    return protT5_encoder.ProtT5(
        cfg, residue_out_dim=cfg.model.residue_out_dim,
        protein_out_dim=cfg.model.protein_out_dim,
        residue_num_projector=cfg.model.residue_num_projector,
        protein_num_projector=cfg.model.protein_num_projector)


def main():
    import yaml
    from box import Box
    from ckpt_slim import (is_frozen_backbone_key, strip_frozen, slim_state,
                           load_slim_into)

    # 直接 yaml + Box，不走 utils_v6.load_configs —— 那条路会拉进 timm/accelerate 等
    # 一堆重依赖，而本测试只需要 configs.model.seq_encoder.* 这几个字段。
    cfg_path = os.environ.get("SPLM_TM_CONFIG") or os.path.join(
        HERE, os.pardir, os.pardir, "models", "backbone", "splm_tm_backbone.yaml")
    cfg = Box(yaml.full_load(open(cfg_path, encoding="utf-8")))

    print("=" * 72)
    print("1) 建模型并把 adapter 训一步（让可训练参数不再是初值）")
    net = build(cfg)
    bc = net.alphabet.get_batch_converter(truncation_seq_length=512)
    _, _, toks = bc([("a", "MKTAYIAKQRQISFVKSHF"), ("b", "MSSHHHHHHSSG")])
    g, r = net(toks)
    (g.square().mean() + r.square().mean()).backward()
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad, alpha=0.1)          # 人为改动，便于验证真的被存下来
    ok("adapter/投影头已偏离初值")

    print("2) 过滤规则：adapter 不能被误判为冻结主干")
    sd = net.state_dict()
    ad_keys = [k for k in sd if ".adapters." in k]
    assert ad_keys, "没有 adapter 参数，测试无意义"
    mis = [k for k in ad_keys if is_frozen_backbone_key(k)]
    assert not mis, f"adapter 被误判成冻结主干：{mis[:3]}"
    ok(f"{len(ad_keys)} 个 adapter 参数全部保留（名字里也带 t5. 但没被误伤）")

    trainable = {k for k, p in net.named_parameters() if p.requires_grad}
    wrongly_dropped = [k for k in trainable if is_frozen_backbone_key(k)]
    assert not wrongly_dropped, f"可训练参数被判为可丢弃：{wrongly_dropped[:3]}"
    ok(f"{len(trainable)} 个可训练参数无一被判为可丢弃")

    print("3) 瘦身效果")
    slim, n_drop, numel = strip_frozen(sd)
    tot = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"       原 {len(sd)} 个张量 / {tot/1e6:.3f}M 元素")
    print(f"       瘦身后 {len(slim)} 个 / {(tot-numel)/1e6:.3f}M 元素"
          f"（丢弃 {n_drop} 个、{numel/1e6:.3f}M）")
    assert numel > 0 and len(slim) < len(sd)
    ok(f"体积降到 {100*(tot-numel)/tot:.1f}%")

    print("4) 存 -> 新建 -> 加载，可训练参数必须逐位一致")
    state = slim_state({"state_dict1": dict(sd)}, keys=("state_dict1",))
    assert state["seq_state_is_slim"] is True
    net2 = build(cfg)                                   # 全新模型，主干同初值
    load_slim_into(net2, state["state_dict1"], is_slim=True, tag=" seq")
    sd2 = net2.state_dict()
    bad = [k for k in trainable if not torch.equal(sd[k], sd2[k])]
    assert not bad, f"可训练参数没恢复：{bad[:3]}"
    ok(f"{len(trainable)} 个可训练参数逐位一致")

    frozen = [k for k in sd if is_frozen_backbone_key(k)]
    bad_f = [k for k in frozen if not torch.equal(sd[k], sd2[k])]
    assert not bad_f, f"冻结主干被改动：{bad_f[:3]}"
    ok(f"{len(frozen)} 个冻结主干张量保持 from_pretrained 原值")

    print("5) 前向输出必须一致（端到端确认）")
    net.eval(); net2.eval()
    with torch.no_grad():
        a1, b1 = net(toks)
        a2, b2 = net2(toks)
    assert torch.allclose(a1, a2, atol=1e-6) and torch.allclose(b1, b2, atol=1e-6)
    ok("两个模型的 protein/残基级输出完全相同")

    print("6) 缺了可训练参数时必须报错，而不是静默加载")
    broken = {k: v for k, v in state["state_dict1"].items() if ".adapters." not in k}
    try:
        load_slim_into(build(cfg), broken, is_slim=True, tag=" 故意残缺")
        raise AssertionError("残缺存档竟然加载成功了 —— 校验没起作用")
    except RuntimeError as e:
        assert "非冻结主干" in str(e)
    ok("残缺存档如期被拒绝")

    print("7) 旧的完整存档仍能加载（三天后续跑要靠它）")
    net3 = build(cfg)
    load_slim_into(net3, dict(sd), is_slim=False, tag=" 旧格式")
    bad3 = [k for k in trainable if not torch.equal(sd[k], net3.state_dict()[k])]
    assert not bad3
    ok("旧格式（完整）存档照常加载，行为不变")

    print("=" * 72)
    print("全部通过。可以接进 save_flow_checkpoints / 加载逻辑。")


if __name__ == "__main__":
    main()
