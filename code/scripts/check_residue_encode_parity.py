#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端核验：release 的 `Backbone.encode_residues()` 与已发布数字所依赖的
残基特征缓存逐位一致。

为什么需要这个：`encode_residues` 是 release 新加的出口。它的 forward 调用与
产出那份缓存的脚本相同（都取第 4 个返回值），但分批与
切分是新写的代码。切错的后果不是报错，而是**残基顺序错位**——比对 F1 会塌，
却没有任何异常。所以这里直接对着那份"产出了全部已发布 MALIDUP/MALISAM 数字"的缓存比。

在**集群**上跑（需要 backbone checkpoint 与 ProtT5 主干缓存）::

    ROOT=<the source root holding results/ and tm_calib/>
    M=<研究仓库里的 model 目录名>
    export SPLM_TM_CONFIG=$ROOT/splm_protT5/results/$M/$M.yaml
    export SPLM_TM_CKPT=$ROOT/splm_protT5/results/$M/checkpoints/<checkpoint>.pth
    python check_residue_encode_parity.py \
        --h5 $ROOT/tm_calib/<residue feature cache>.h5 \
        --valid $ROOT/CLAlign/data/valid300.csv

缓存是 **float16** 存的，所以比较时把本次编码也降到 float16 再比 —— 期望 max|diff| = 0。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


def seq_key(seq: str) -> str:
    """与研究仓库里那份残基特征缓存完全一致的键。"""
    return hashlib.md5(seq.encode()).hexdigest()


def collect_from_valid(path: Path):
    """从 CLAlign 的 valid300.csv 里取序列。列名不固定，所以按"看起来像蛋白序列"挑。"""
    out = []
    with open(path, newline="") as fp:
        for row in csv.reader(fp):
            for cell in row:
                c = cell.strip().upper()
                if 30 <= len(c) <= 500 and set(c) <= set("ACDEFGHIKLMNPQRSTVWYXBZUO"):
                    out.append(c)
    # 去重保序
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="残基特征缓存 .h5")
    ap.add_argument("--valid", required=True, help="CLAlign 的 valid300.csv")
    ap.add_argument("--n", type=int, default=6, help="抽多少条来比")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    import h5py
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from splm_tm import Backbone                      # noqa: E402

    cand = collect_from_valid(Path(args.valid))
    if not cand:
        print(f"!! 从 {args.valid} 里没解析出序列")
        return 2

    with h5py.File(args.h5, "r") as h5:
        # 只挑缓存里真有、且长度 <512 的（避免截断成为混淆变量）
        pick = []
        for s in cand:
            k = seq_key(s)
            if k in h5 and len(s) < 512 and h5[k].shape[0] == len(s):
                pick.append((k, s))
            if len(pick) >= args.n:
                break
        if not pick:
            print("!! 缓存里找不到可用的对照序列")
            return 2
        print(f">>> 取到 {len(pick)} 条对照序列，长度 {[len(s) for _, s in pick]}")
        ref = {k: np.asarray(h5[k][:]) for k, _ in pick}

    bb = Backbone.load()
    # ⚠️ max_length=None：缓存是不截断建的，截断会让长序列的行数对不上，
    #    那样比出来的差异是口径差异而不是实现 bug。
    got = bb.encode_residues([s for _, s in pick], ids=[k for k, _ in pick],
                             batch_size=args.batch_size, max_length=None, verbose=True)

    print()
    print("%-26s %8s %10s %12s" % ("key", "len", "shape ok", "max|diff| (fp16)"))
    worst = 0.0
    bad_shape = 0
    for (k, s), a in zip(pick, got):
        r = ref[k]
        shape_ok = a.shape == r.shape
        bad_shape += (not shape_ok)
        if shape_ok:
            # 缓存是 fp16 存的 => 把本次结果也降到 fp16 再比，否则比的是存储精度
            d = float(np.abs(a.astype(np.float16).astype(np.float32)
                             - r.astype(np.float32)).max())
            worst = max(worst, d)
        else:
            d = float("nan")
        print("%-26s %8d %10s %12s" % (k[:24], len(s), shape_ok, f"{d:.6g}"))

    print()
    ok = bad_shape == 0 and worst == 0.0
    if ok:
        print("[PASS] 与缓存逐位一致 (max|diff| = 0) —— release 的残基级出口"
              "产出的就是已发布数字所用的特征")
    else:
        print(f"[FAIL] 形状不符 {bad_shape} 条, max|diff| = {worst}")
        print("       别发 —— 残基顺序或分批切分有问题，比对 F1 会塌且不报错")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
