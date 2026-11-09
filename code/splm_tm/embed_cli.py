#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S-PLM-TM 命令行编码器：序列 -> embedding（raw / tm / ec 任选）。

`splmtm embed` 与 `python code/scripts/embed.py` 是同一实现，参数和默认值完全相同。

典型用法
--------
    # 检索用（raw + TM head）
    splmtm embed --fasta my.fasta --modes raw tm --out out.npz

    # 只要迁移下游用的原始向量
    splmtm embed --fasta my.fasta --modes raw --out raw.npz

    # 带 EC head（EC head 吃 emb_h，所以 tm 会被自动带上）
    splmtm embed --fasta my.fasta --modes raw ec --out ec.npz --print-ec

    # 不写文件、直接看几条序列的预测 TM-score 矩阵
    splmtm embed --seq MVLSEGEWQ... --seq KVFGRCELAA... --modes tm --print-tm

输出 npz 的 key
---------------
    ids        [N]            序列 id（FASTA header 第一个空格前的部分）
    emb_raw    [N,1024]       pre-projector 原始向量，未归一化 —— 迁移下游用这个
    emb_tm     [N,1024]       TM 校准头输出，已 L2 归一化，内积 = 预测 TM-score
    emb_h      [N,1024]       emb_tm 的别名（兼容既有下游脚本里的旧 key）
    emb_ec     [N,1024]       EC 检索头输出，**已 L2 归一化** —— EC/功能检索用这个
    emb_ec_raw [N,1024]       同上但未归一化（对照用）

    d = np.load("out.npz", allow_pickle=True)
    sim = d["emb_tm"] @ d["emb_tm"].T       # 已归一化，不要再 normalize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    from .api import MODES
    src = p.add_argument_group("输入（--fasta / --seq / --seq-file 三者取一）")
    src.add_argument("--fasta", help="输入 FASTA")
    src.add_argument("--seq", action="append", default=[],
                     help="直接给一条序列，可重复多次")
    src.add_argument("--seq-file", help="纯序列文本，一行一条（没有 header）")
    src.add_argument("--ids", help="与 --seq/--seq-file 配套的 id 文件，一行一个")

    out = p.add_argument_group("输出")
    out.add_argument("--out", help="输出 .npz（不给就只打印，不落盘）")
    out.add_argument("--modes", nargs="+", default=["raw", "tm"],
                     help=f"要哪些 embedding，可选 {list(MODES)} 或 all（默认 raw tm）。"
                          f"raw 总会算，因为 tm/ec 都由它派生")
    out.add_argument("--print-tm", action="store_true",
                     help="打印预测 TM-score 矩阵（序列条数少时用）")
    out.add_argument("--print-ec", action="store_true",
                     help="打印 EC 检索相似度矩阵（序列条数少时用）")
    out.add_argument("--save-json",
                     help="把 ids / 预测 TM 矩阵 / EC 相似度矩阵另存一份 json")

    mdl = p.add_argument_group("模型文件（都可省，默认用 release 里的 models/）")
    mdl.add_argument("--config", help="backbone config yaml")
    mdl.add_argument("--ckpt", help="backbone checkpoint .pth")
    mdl.add_argument("--tm-head", help="TM 校准头 .pt")
    mdl.add_argument("--ec-head", help="EC head .pt")
    mdl.add_argument("--show-paths", action="store_true",
                     help="只打印解析到的模型路径然后退出")

    run_ = p.add_argument_group("运行")
    run_.add_argument("--batch-size", type=int, default=8,
                      help="batch 内条数上限（受 --max-tokens 约束）。ProtT5-XL 吃显存，默认 8")
    run_.add_argument("--max-tokens", type=int, default=8192,
                      help="单 batch 的 b*L_max 预算，长序列自动落到小 batch")
    run_.add_argument("--max-length", type=int, default=None,
                      help="默认全长、不截断；正整数仅保留前 N 个残基，0 也表示不截断（显存 O(b*L^2)）")
    run_.add_argument("--device", default=None, help="cuda / cuda:0 / cpu（默认自动）")
    run_.add_argument("--quiet", action="store_true")
    return p


def build_parser() -> argparse.ArgumentParser:
    return add_arguments(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))


def load_inputs(args):
    from .fasta import read_fasta
    if args.fasta:
        return read_fasta(args.fasta)
    seqs = list(args.seq)
    if args.seq_file:
        with open(args.seq_file) as f:
            seqs += [ln.strip() for ln in f if ln.strip()]
    if not seqs:
        raise SystemExit("没有输入：用 --fasta、--seq 或 --seq-file 给序列")
    if args.ids:
        with open(args.ids) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        if len(ids) != len(seqs):
            raise SystemExit(f"--ids 有 {len(ids)} 个，序列有 {len(seqs)} 条")
    else:
        ids = None
    return ids, seqs


def run(args) -> int:
    from . import paths as splm_paths
    if args.show_paths:
        print(splm_paths.describe())
        return 0

    import numpy as np
    from .api import SPLMTM, save_npz

    ids, seqs = load_inputs(args)
    verbose = not args.quiet
    modes = args.modes
    if args.print_tm and "tm" not in modes and "all" not in modes:
        modes = list(modes) + ["tm"]
    if args.print_ec and "ec" not in modes and "all" not in modes:
        modes = list(modes) + ["ec"]

    model = SPLMTM.load(modes=modes, config_path=args.config, ckpt_path=args.ckpt,
                        tm_head_path=args.tm_head, ec_head_path=args.ec_head,
                        device=args.device, verbose=verbose)
    res = model.embed(seqs, ids, modes=modes, batch_size=args.batch_size,
                      max_length=args.max_length or None, max_tokens=args.max_tokens,
                      verbose=verbose)

    extra = {"ids": [str(i) for i in res["ids"]]}
    if args.print_tm and "emb_tm" in res:
        sim = model.predicted_tm_matrix(res["emb_tm"])
        extra["predicted_tm"] = sim.tolist()
        print("\n>>> 预测 TM-score 矩阵（行列顺序 = ids）")
        w = max(12, *(len(str(i)) for i in res["ids"]))
        print(" " * w + "".join(f"{str(i)[:9]:>10}" for i in res["ids"]))
        for i, row in zip(res["ids"], sim):
            print(f"{str(i)[:w]:<{w}}" + "".join(f"{v:>10.3f}" for v in row))

    if args.print_ec and "emb_ec" in res:
        sim = model.ec_similarity_matrix(res["emb_ec"])
        extra["ec_similarity"] = sim.tolist()
        print("\n>>> EC 检索相似度矩阵（行列顺序 = ids）")
        print("    注意：这不是 TM-score，也不是 EC 概率 —— EC head 是检索头，不是分类器")
        w = max(12, *(len(str(i)) for i in res["ids"]))
        print(" " * w + "".join(f"{str(i)[:9]:>10}" for i in res["ids"]))
        for i, row in zip(res["ids"], sim):
            print(f"{str(i)[:w]:<{w}}" + "".join(f"{v:>10.3f}" for v in row))

    if args.out:
        p = save_npz(args.out, res)
        print(f"\n>>> 写出 {p}")
        print("    读取: d = np.load(path, allow_pickle=True); d['ids'], d['emb_raw']")
        if "emb_tm" in res:
            print("    检索: sim = d['emb_tm'] @ d['emb_tm'].T   # 已归一化，内积即余弦")
    elif not args.print_tm and not args.print_ec:
        print("\n>>> 没给 --out，结果没有落盘。各 key 的形状：")
        for k, v in res.items():
            print(f"    {k:<10} {np.asarray(v).shape}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(extra, f, indent=2, ensure_ascii=False)
        print(f">>> 写出 {args.save_json}")
    return 0


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))
