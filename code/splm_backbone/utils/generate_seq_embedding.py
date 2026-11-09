# -*- coding: utf-8 -*-
"""ProtT5 版 SequenceRepresentModel —— 供 `splm_tm.Backbone` 加载序列塔用。

两件事别改：
  ① 顶部的骸干类型硬校验 —— 装错骸干直接报错，而不是静默产出垃圾向量；
  ② `ckpt_slim.load_slim_into(...)` 会校验「缺失的键恰好是冻结的 T5 主干」，
     少一个可训练参数就报错；主干由 `from_pretrained` 复原。

调用约定：返回 `(model, batch_converter, device)`，其中
`model(batch_tokens=toks, mode="sequence", return_embedding=True)` 的**第 3 个**返回值
= pre-projector 的蛋白向量（1024 维）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator

# 本仓库必须在 sys.path 最前，否则 utils / model_v6 会被外部同名包遮蔽
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) in sys.path:
    sys.path.remove(str(_REPO))
sys.path.insert(0, str(_REPO))


def SequenceRepresentModel(config_path: str, checkpoint_path: str | None = None,
                           truncate_inference=None,
                           max_length_inference=None,
                           afterproject: bool = False):
    """构建冻结的 ProtT5 序列塔并装载 checkpoint。afterproject 保留只为签名兼容（未使用）。"""
    from utils.utils_v6 import load_configs          # 与训练同一条 config 解析
    from model_v6 import prepare_models              # 里面 ESM2 = ProtT5

    with open(config_path) as fh:
        configs = load_configs(yaml.full_load(fh), args=None)

    accelerator = Accelerator(mixed_precision=configs.train_settings.mixed_precision)
    model = prepare_models(configs=configs)

    # ---- 骨干类型硬校验：装错了立刻炸，别静默产出垃圾向量 ----
    from protT5_encoder import ProtT5
    if not isinstance(model.model_seq, ProtT5):
        raise RuntimeError(
            f"序列塔不是 ProtT5 而是 {type(model.model_seq).__module__}."
            f"{type(model.model_seq).__name__} —— 几乎肯定是 sys.path 把别的仓库的 "
            f"model/utils 抢先命中了。检查 PYTHONPATH 里有没有同名的 utils / model 包。\n"
            f"sys.path[:5]={sys.path[:5]}")

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict1" not in ckpt:
            raise RuntimeError(f"{checkpoint_path} 里没有 state_dict1（序列塔权重）"
                               f"；顶层键={sorted(ckpt.keys())[:12]}")
        from ckpt_slim import load_slim_into
        load_slim_into(model.model_seq, ckpt["state_dict1"],
                       is_slim=bool(ckpt.get("seq_state_is_slim", False)),
                       tag="model_seq(eval)")
        print(f">>> 已装载 state_dict1 (step={ckpt.get('step')}, "
              f"slim={bool(ckpt.get('seq_state_is_slim', False))}) <- {checkpoint_path}")
        del ckpt

    model = accelerator.prepare(model)
    model.eval()

    # batch_converter：ProtT5 的 _T5Alphabet 提供了与 ESM alphabet 同签名的接口
    alphabet = model.model_seq.alphabet if hasattr(model, "model_seq") else model.module.model_seq.alphabet
    seq_cfg = configs.model.seq_encoder
    if truncate_inference is not None:
        if truncate_inference:
            assert max_length_inference is not None, \
                "truncate_inference=True 时 max_length_inference 不能为 None"
            batch_converter = alphabet.get_batch_converter(truncation_seq_length=max_length_inference)
            print(f"truncate_inference with max_length={max_length_inference}")
        else:
            batch_converter = alphabet.get_batch_converter()
    elif getattr(seq_cfg, "truncate_inference", False):
        _ml = getattr(seq_cfg, "max_length_inference", None)
        batch_converter = alphabet.get_batch_converter(truncation_seq_length=_ml)
        print(f"truncate_inference with max_length={_ml} (来自 config)")
    else:
        # 默认不截断 —— 与既有 SCOPE 缓存的 TRUNCATE_INFERENCE=False 口径一致；
        # 长序列由调用方的 max_tokens 预算分批控制显存。
        batch_converter = alphabet.get_batch_converter()

    return model, batch_converter, accelerator.device
