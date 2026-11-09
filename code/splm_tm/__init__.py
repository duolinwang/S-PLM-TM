# -*- coding: utf-8 -*-
"""S-PLM-TM 推理包 —— 序列进，embedding 出。

    from splm_tm import SPLMTM
    m = SPLMTM.load(modes=("raw", "tm", "ec"))
    r = m.embed(["MVLSEGEWQLVLHVWAKVEAD..."])

三种**蛋白级** embedding，依赖链是串行的：

    序列 --backbone--> emb_raw --TM head--> emb_tm(=emb_h) --EC head--> emb_ec

    raw  pre-projector 原始向量 [N,1024]，未归一化 —— 迁移下游（自己训分类/回归）用这个
    tm   TM 校准头输出 [N,1024]，已 L2 归一化，内积 = 预测 TM-score —— 结构检索用这个
    ec   EC 检索头输出 [N,1024]，已 L2 归一化 —— EC / 功能检索用这个（**不是分类概率**）

命令行：`splmtm embed --fasta x.fasta --modes raw tm ec --out out.npz`

检索（embedding -> 排名）：`python code/scripts/search.py --query q.npz --target t.npz`
加 `--prefilter pfamclan` 就是 MODEL_CARD 里「+TM head +Pfam 预筛」那一档口径。
Pfam 注释与 embedding **互相独立**，只在打分后以逐行 mask 的形式汇合；细节与
「无 Pfam 命中的 query 退回全库」这条行为见 `splm_tm/pfam.py`。

另有一条**残基级**旁支，不在上面那条链上：

    序列 --backbone--> residue_feats [L,1024] --align head--> [L,1024]（L2 归一化）

用于残基级比对（MALIDUP / MALISAM 式的逐残基对应）。入口 =
`Backbone.encode_residues` + `heads.load_align_head` / `apply_align_head`。
⚠️ 比对头吃残基级特征，**不能**接到 emb_raw / emb_h 上（蛋白级，形状和含义都不对）。
见 `models/align_head/README.md`。
"""
from importlib import import_module as _import_module

from .api import MODE_REQUIRES, MODES, SPLMTM, save_npz
from .backbone import Backbone
from .fasta import clean_sequences, make_ids, read_fasta, write_fasta
from .pfam import (apply_pfam_mask, apply_pfam_mask_row, build_pfam_overlap_map,
                   load_clan_map, load_pfam_sets, pfam_missing_message,
                   prepare_pfam_overlap)
from .heads import (MLP_DEFAULT_CFG, AlignHead, ECHead, ECRetrievalHead,
                    align_head_missing_message, apply_align_head, apply_ec_head,
                    apply_tm_head, load_align_head, load_tm_head,
                    residue_similarity)

__version__ = "3.3.0"

# Reproduction runners share the low-level model and data utilities above, but
# must remain importable independently of the evolving search/alignment APIs.
_LAZY_EXPORTS = {
    "Index": ".index",
    "QueryEmbeddings": ".index",
    "SearchIndex": ".index",
    "SearchResult": ".index",
    "T5SearchEncoder": ".search_encoders",
    "LegacyESMSearchEncoder": ".search_encoders",
    "CandidateAligner": ".alignment",
    "CandidateAlignmentResult": ".alignment",
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "SPLMTM", "MODES", "MODE_REQUIRES", "save_npz", "Backbone",
    "Index", "SearchIndex", "SearchResult", "QueryEmbeddings", "T5SearchEncoder", "LegacyESMSearchEncoder",
    "CandidateAligner", "CandidateAlignmentResult",
    "read_fasta", "write_fasta", "clean_sequences", "make_ids",
    "ECRetrievalHead", "ECHead", "MLP_DEFAULT_CFG", "apply_ec_head",
    "load_tm_head", "apply_tm_head",
    "AlignHead", "load_align_head", "apply_align_head", "residue_similarity",
    "align_head_missing_message",
    "prepare_pfam_overlap", "apply_pfam_mask", "apply_pfam_mask_row",
    "build_pfam_overlap_map", "load_clan_map", "load_pfam_sets",
    "pfam_missing_message",
    "__version__",
]
