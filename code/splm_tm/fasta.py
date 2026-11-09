# -*- coding: utf-8 -*-
"""极简 FASTA 读写。id 取 header 第一个空格前的部分。"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

# ProtT5 的 tokenizer 不认识 U/Z/O/B，官方用法是统一映射成 X。
_NONSTD = re.compile(r"[UZOB]")


def read_fasta(path) -> Tuple[List[str], List[str]]:
    ids, seqs, cur, buf = [], [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    ids.append(cur)
                    seqs.append("".join(buf))
                cur, buf = (line[1:].split() or [""])[0], []
            else:
                buf.append(line)
    if cur is not None:
        ids.append(cur)
        seqs.append("".join(buf))
    if not ids:
        raise ValueError(f"{path} 里没有读到任何序列（FASTA 格式？）")
    return ids, seqs


def write_fasta(path, ids: Sequence[str], seqs: Sequence[str], width: int = 60) -> None:
    with open(path, "w") as f:
        for i, s in zip(ids, seqs):
            f.write(f">{i}\n")
            for k in range(0, len(s), width):
                f.write(s[k:k + width] + "\n")


def clean_sequences(seqs: Iterable[str]) -> List[str]:
    """大写、去空白、非标准残基 U/Z/O/B -> X。不做截断（截断在编码时按 max_length 做）。

    ⚠️ 编码路径上**不需要**调用它：`T5BatchConverter._prep`（splm_backbone/protT5_encoder.py）
    已经做了同样的大写 + U/Z/O/B->X，这里只是幂等的重复，留着给上游做数据体检用。
    默认不调用，是为了让 release 的编码口径与发布的 benchmark 数字逐位一致。
    """
    out = []
    for s in seqs:
        s = "".join(str(s).split()).upper()
        out.append(_NONSTD.sub("X", s))
    return out


def make_ids(n: int, prefix: str = "seq") -> List[str]:
    w = len(str(max(1, n - 1)))
    return [f"{prefix}{i:0{w}d}" for i in range(n)]
