# -*- coding: utf-8 -*-
"""存档瘦身：不保存冻结的 T5 主干。

为什么值得做
------------
现在每个存档 12 GB，到 28000 步会有 14 个周期性存档 ≈ 168 GB。
里面的大头是**冻结不变的 ProtT5 主干**（1.2B × fp32 ≈ 4.8 GB），而且存了不止一份：
  · `state_dict1` = 序列编码器本体
  · `flow_all`    = flowmodels.state_dict()，里面还含 online/target 两套编码器
主干在整个训练里一个字节都不会变，与 `from_pretrained` 拿到的完全相同 —— 存它纯属浪费。

过滤规则（这一条最容易写错）
--------------------------
adapter 是用 forward hook 注入到 T5Block 里的，所以它们的参数名**也带 `t5.` 前缀**：
    t5.encoder.block.12.adapters.0.up.weight     <- 必须保留！
    t5.encoder.block.12.layer.0.SelfAttention.q.weight   <- 冻结，丢掉
因此不能简单地按 `t5.` 过滤，必须**排除掉 adapters 那一支**。
本模块统一用 `is_frozen_backbone_key()` 判断，保存与校验共用同一个函数。

加载
----
`load_slim_into()` 用 `strict=False`，并**校验缺失的 key 恰好是冻结主干那一批** ——
少一个可训练参数就报错，而不是静默加载出一个半吊子模型。

⚠️ 必须兼容旧的完整存档：当前这轮（job 16985133）产出的是旧格式，三天后续跑要能接上。
`load_slim_into()` 对两种格式都能处理（靠 checkpoint 里的 `seq_state_is_slim` 标记）。
"""
from __future__ import annotations

# 冻结主干的标志：带 t5. 前缀但不属于 adapters 那一支
_ADAPTER_MARK = ".adapters."


def is_frozen_backbone_key(key: str) -> bool:
    """这个参数名是否属于**冻结的 T5 主干**（= 存档里可以省掉的部分）。

    带 `t5.` 前缀 且 不在 adapters 支路上 -> 冻结主干。
    adapter 虽然物理上挂在 T5Block 里、名字也带 t5.，但它是可训练的，必须留。
    """
    k = str(key)
    if "t5." not in k:
        return False
    return _ADAPTER_MARK not in k


def strip_frozen(state_dict: dict) -> tuple[dict, int, int]:
    """去掉冻结主干的键。返回 (瘦身后的 dict, 丢弃数, 丢弃的元素总量)。"""
    out, n_drop, numel = {}, 0, 0
    for k, v in state_dict.items():
        if is_frozen_backbone_key(k):
            n_drop += 1
            try:
                numel += v.numel()
            except AttributeError:
                pass
        else:
            out[k] = v
    return out, n_drop, numel


def slim_state(state: dict, keys=("state_dict1", "flow_all"), logging=None) -> dict:
    """就地瘦身一个 checkpoint 状态字典，并打上 `seq_state_is_slim` 标记。

    只处理会含 T5 主干的那几个键；`state_dict2`（GVP）本来就小，不动。
    """
    total_numel = 0
    for k in keys:
        if k in state and isinstance(state[k], dict):
            state[k], n, numel = strip_frozen(state[k])
            total_numel += numel
            msg = f"[ckpt_slim] {k}: 丢弃 {n} 个冻结主干张量（{numel/1e6:.0f}M 元素）"
            (logging.info if logging else print)(msg)
    state["seq_state_is_slim"] = True
    saved_gb = total_numel * 4 / 1e9
    msg = f"[ckpt_slim] 本次省下约 {saved_gb:.2f} GB（按 fp32 计）"
    (logging.info if logging else print)(msg)
    return state


def load_slim_into(module, state_dict: dict, is_slim: bool, tag: str = "", logging=None):
    """把（可能瘦身过的）state_dict 灌进 module。

    is_slim=True 时用 strict=False，并校验缺失的 key **恰好**是冻结主干那批；
    若有任何可训练参数缺失就抛错 —— 宁可失败也不要静默加载出半个模型。
    is_slim=False（旧的完整存档）走 strict=True，行为与改造前一致。
    """
    if not is_slim:
        module.load_state_dict(state_dict)
        return

    res = module.load_state_dict(state_dict, strict=False)
    missing = list(getattr(res, "missing_keys", []))
    unexpected = list(getattr(res, "unexpected_keys", []))

    bad_missing = [k for k in missing if not is_frozen_backbone_key(k)]
    if bad_missing:
        raise RuntimeError(
            f"[ckpt_slim]{tag} 存档里缺了 {len(bad_missing)} 个**非冻结主干**的参数，"
            f"不能当作瘦身存档加载。前几个：{bad_missing[:5]}")
    if unexpected:
        raise RuntimeError(
            f"[ckpt_slim]{tag} 存档里有 {len(unexpected)} 个模型上不存在的键：{unexpected[:5]}")

    msg = (f"[ckpt_slim]{tag} 已加载瘦身存档：跳过 {len(missing)} 个冻结主干张量"
           f"（它们由 from_pretrained 提供，本来就不变）")
    (logging.info if logging else print)(msg)
