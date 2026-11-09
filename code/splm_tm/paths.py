# -*- coding: utf-8 -*-
"""默认文件定位：把 release 包里的 models/ 目录解析成具体路径。

目录约定（相对本 release 根目录）::

    models/backbone/splm_tm_backbone.pth      编码器权重（slim 存档，不含冻结 ProtT5 主干）
    models/backbone/<模型全名>.yaml             架构定义，必须与上面成对
    models/tm_head/tm_calib_head_*.pt           TM 校准头，必须与上面的 ckpt 成对
    models/ec_head/ec_head_*.pt                 EC head（可选；没有时 mode="ec" 会给出明确报错）
    models/align_head/align_head.pt             残基级比对头（可选，mode="align"）

⚠️ 比对头与其他两个头**吃的东西不一样**：TM / EC head 吃**蛋白级**向量，
而比对头吃**残基级**特征（`Backbone.encode_residues`，每条序列 [L,1024]）。
别把它接到 `emb_raw` / `emb_h` 上 —— 那是蛋白级，形状对不上也没有意义。

每一项都能用环境变量覆盖，便于在集群上直接指到共享盘而不必拷贝：

    SPLM_TM_MODELS      models/ 根目录
    SPLM_TM_CONFIG      config yaml
    SPLM_TM_CKPT        backbone checkpoint
    SPLM_TM_TM_HEAD     TM 校准头
    SPLM_TM_EC_HEAD     EC head
    SPLM_TM_ALIGN_HEAD  比对头
"""
from __future__ import annotations

import os
from pathlib import Path

# code/splm_tm/paths.py -> code/splm_tm -> code -> <release root>
PKG_DIR = Path(__file__).resolve().parent
CODE_DIR = PKG_DIR.parent
RELEASE_ROOT = CODE_DIR.parent
BACKBONE_CODE_DIR = CODE_DIR / "splm_backbone"


def models_dir() -> Path:
    env = os.environ.get("SPLM_TM_MODELS")
    return Path(env).expanduser().resolve() if env else RELEASE_ROOT / "models"


# 权重不随 git 分发（backbone 单文件 2.2 GB，超 GitHub 上限），所以「文件不在」是
# 外部用户最先撞上的错。找不到文件的报错一律带上这段，别让人对着空目录猜。
_GET_WEIGHTS = "\n".join([
    "  权重不在 git 仓库里，需要另外下载：",
    "    pip install 'huggingface_hub[cli]'",
    "    hf download deepduoduo/S-PLM-TM --local-dir models/",
    "  详见 models/DOWNLOAD.md（含 md5 校验表）。",
    "  已下到别处的话，用 SPLM_TM_MODELS 指向 models/ 根目录。",
    "  想看当前解析到哪些路径：splmtm embed --show-paths",
])


def _pick(d: Path, patterns, what: str, env_var: str, required: bool = True):
    """在目录里按 glob 找唯一文件。找到多个时报错而不是随便挑一个 —— 配错 ckpt/head
    不会抛异常、只会静默产出无意义的向量，所以这里宁可早失败。"""
    if not d.is_dir():
        if required:
            raise FileNotFoundError(
                f"找不到 {what} 所在目录：{d}\n"
                f"  要么把文件放进该目录，要么设环境变量 {env_var} 指向具体文件。\n"
                f"{_GET_WEIGHTS}")
        return None
    hits = []
    for pat in patterns:
        hits += sorted(p for p in d.glob(pat) if p.is_file())
    # 去重并保持顺序
    seen, uniq = set(), []
    for p in hits:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if not uniq:
        if required:
            raise FileNotFoundError(
                f"{d} 里没有 {what}（找的是 {', '.join(patterns)}）\n"
                f"  可用环境变量 {env_var} 指向具体文件。\n"
                f"{_GET_WEIGHTS}")
        return None
    if len(uniq) > 1:
        raise RuntimeError(
            f"{d} 里有多个 {what}，无法判断用哪个：\n  "
            + "\n  ".join(str(p.name) for p in uniq)
            + f"\n  请只保留一个，或用环境变量 {env_var} 显式指定。")
    return uniq[0]


def default_config() -> Path:
    env = os.environ.get("SPLM_TM_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return _pick(models_dir() / "backbone", ["*.yaml", "*.yml"],
                 "config yaml", "SPLM_TM_CONFIG")


def default_ckpt() -> Path:
    env = os.environ.get("SPLM_TM_CKPT")
    if env:
        return Path(env).expanduser().resolve()
    return _pick(models_dir() / "backbone", ["*.pth", "*.pt"],
                 "backbone checkpoint", "SPLM_TM_CKPT")


def default_tm_head() -> Path:
    env = os.environ.get("SPLM_TM_TM_HEAD")
    if env:
        return Path(env).expanduser().resolve()
    return _pick(models_dir() / "tm_head", ["tm_calib_head_*.pt", "*.pt"],
                 "TM 校准头", "SPLM_TM_TM_HEAD")


def default_ec_head(required: bool = True):
    env = os.environ.get("SPLM_TM_EC_HEAD")
    if env:
        return Path(env).expanduser().resolve()
    return _pick(models_dir() / "ec_head", ["ec_head_*.pt", "*.pt"],
                 "EC head", "SPLM_TM_EC_HEAD", required=required)


def default_align_head(required: bool = True):
    """残基级比对头。默认只认 align_head/ 顶层的 .pt ——
    `alternatives/` 里的变体不会被自动挑中（glob 不递归），要用就显式传路径
    或设 SPLM_TM_ALIGN_HEAD。"""
    env = os.environ.get("SPLM_TM_ALIGN_HEAD")
    if env:
        return Path(env).expanduser().resolve()
    return _pick(models_dir() / "align_head", ["align_head*.pt", "*.pt"],
                 "比对头", "SPLM_TM_ALIGN_HEAD", required=required)


def describe() -> str:
    """把当前解析到的路径打成一段可读文本，供脚本启动时打印/排错。"""
    lines = [f"release root : {RELEASE_ROOT}", f"models dir   : {models_dir()}"]
    for name, fn in (("config", default_config), ("ckpt", default_ckpt),
                     ("tm head", default_tm_head)):
        try:
            lines.append(f"{name:<13}: {fn()}")
        except Exception as e:                                  # noqa: BLE001
            lines.append(f"{name:<13}: <未解析> {e.__class__.__name__}: {e}")
    for name, fn in (("ec head", default_ec_head), ("align head", default_align_head)):
        try:
            p = fn(required=False)
        except Exception as e:                                  # noqa: BLE001
            p = f"<多个候选> {e}"
        lines.append(f"{name:<13}: {p if p else '<未提供>'}")
    return "\n".join(lines)
