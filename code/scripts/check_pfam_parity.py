#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核验 release 的 `splm_tm.pfam` 与研究仓库原实现**逐位等价**。

为什么需要这个：release 复刻了
`microbiom_searchproblem/bacteria_benchmark/bacteria_pfam.py` 的 Pfam 预筛逻辑
（为了不依赖研究仓库的目录结构）。复刻抄错的典型后果不是报错，是**悄悄改变排名** ——
mask 多盖一格或少盖一格，Recall 差几个千分点，没人能一眼看出来。
所以这里用随机数据把两份实现都跑一遍，逐位比较。

跑法（需要能 import 研究仓库的 bacteria_pfam）::

    python code/scripts/check_pfam_parity.py \
        --research-repo /path/to/splm_gvpgit/microbiom_searchproblem

期望：每一项都 `max|diff| = 0` / `identical`。

覆盖的点：
  1. parse_pfam_scan_txt  —— 第 5 列取 accession、# 注释行跳过、无命中序列留空列表
  2. load_clan_map        —— clan 为空的 family 不入表
  3. load_pfam_sets       —— 版本号剥离、family->clan 映射、无 clan 的 family 保留原值
  4. build_pfam_overlap_map
  5. apply_pfam_mask      —— 含「无候选 -> 整行回退全库」这条
  6. apply_pfam_mask_row
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-repo", default=".",
                    help="含 bacteria_benchmark/bacteria_pfam.py 的目录"
                         "（即 microbiom_searchproblem）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-queries", type=int, default=25)
    ap.add_argument("--n-targets", type=int, default=120)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))                  # <release>/code
    from splm_tm import pfam as R                          # noqa: E402

    repo = Path(args.research_repo).resolve()
    bench = repo / "bacteria_benchmark"
    if not (bench / "bacteria_pfam.py").is_file():
        print("[!] 找不到 %s/bacteria_pfam.py；--research-repo 要指到 "
              "microbiom_searchproblem" % bench)
        return 2
    sys.path.insert(0, str(bench))
    import bacteria_pfam as O                              # noqa: E402
    import torch                                           # noqa: E402

    rng = np.random.default_rng(args.seed)
    tmp = Path(tempfile.mkdtemp())
    ok = True

    def check(name, cond):
        nonlocal ok
        print("  %-28s %s" % (name, "identical" if cond else "*** MISMATCH ***"))
        ok = ok and bool(cond)

    # ── 1. parse_pfam_scan_txt ─────────────────────────────────────────────
    seq_ids = ["q%d" % i for i in range(6)]
    txt = tmp / "scan.txt"
    txt.write_text(
        "# comment line that must be skipped\n"
        "\n"
        "q0 1 50 1 55 PF00069.27 Pkinase stuff\n"
        "q0 60 90 58 95 PF00028.1 Cadherin more\n"
        "q1 5 40 3 45 PF00069.27 Pkinase x\n"
        "q2 too few cols\n"                       # <6 列，应被跳过
        "qX 1 9 1 9 PF99999.1 NotInSet y\n",      # 不在 seq_ids 里，应被忽略
        encoding="utf-8")
    check("parse_pfam_scan_txt",
          R.parse_pfam_scan_txt(txt, seq_ids) == O.parse_pfam_scan_txt(txt, seq_ids))

    # ── 2. load_clan_map ───────────────────────────────────────────────────
    clan_tsv = tmp / "clans.tsv"
    clan_tsv.write_text(
        "PF00069\tCL0016\tPkinase_clan\tPkinase\tdesc\n"
        "PF00028\t\t\tCadherin\tno clan here\n"     # clan 为空 -> 不入表
        "PF00001\tCL0192\tGPCR_A\t7tm_1\tdesc\n"
        "\n",
        encoding="utf-8")
    check("load_clan_map",
          R.load_clan_map(clan_tsv) == O.load_clan_map(clan_tsv))

    # ── 3. load_pfam_sets（fam 与 clan 两种模式）───────────────────────────
    q_json, t_json = tmp / "q.json", tmp / "t.json"
    fams = ["PF00069.27", "PF00028.1", "PF00001.20", "PF77777.3"]
    q_raw = {"q%d" % i: list(rng.choice(fams, size=rng.integers(0, 3), replace=False))
             for i in range(args.n_queries)}
    t_raw = {"t%d" % i: list(rng.choice(fams, size=rng.integers(0, 3), replace=False))
             for i in range(args.n_targets)}
    q_json.write_text(json.dumps(q_raw), encoding="utf-8")
    t_json.write_text(json.dumps(t_raw), encoding="utf-8")

    fam2clan = R.load_clan_map(clan_tsv)
    check("load_pfam_sets (family)",
          R.load_pfam_sets(q_json, False) == O.load_pfam_sets(q_json, False))
    check("load_pfam_sets (clan)",
          R.load_pfam_sets(q_json, True, fam2clan)
          == O.load_pfam_sets(q_json, True, fam2clan))

    # ── 4. build_pfam_overlap_map ──────────────────────────────────────────
    target_ids = ["t%d" % i for i in range(args.n_targets)]
    # 故意少给几个 target_id，检验"不在 target_ids 里的注释要被忽略"这条
    target_ids_subset = target_ids[:-5]
    qs = R.load_pfam_sets(q_json, True, fam2clan)
    ts = R.load_pfam_sets(t_json, True, fam2clan)
    ov_r = R.build_pfam_overlap_map(qs, ts, target_ids_subset)
    ov_o = O.build_pfam_overlap_map(qs, ts, target_ids_subset)
    check("build_pfam_overlap_map", ov_r == ov_o)
    n_fb = sum(1 for v in ov_r.values() if not v)
    print("     (%d/%d queries have no candidates -> exercises the fallback path)"
          % (n_fb, len(ov_r)))

    # ── 5. apply_pfam_mask ─────────────────────────────────────────────────
    query_ids = ["q%d" % i for i in range(args.n_queries)]
    sim = rng.standard_normal((len(query_ids), len(target_ids_subset))).astype(np.float32)
    m_r = R.apply_pfam_mask(sim, query_ids, target_ids_subset, ov_r, verbose=False)
    m_o = O.apply_pfam_mask(torch.tensor(sim), query_ids, target_ids_subset, ov_o,
                            torch.device("cpu")).numpy()
    same = (np.array_equal(np.isneginf(m_r), np.isneginf(m_o))
            and np.array_equal(np.nan_to_num(m_r, neginf=0.0),
                               np.nan_to_num(m_o, neginf=0.0)))
    check("apply_pfam_mask", same)
    if not same:
        print("     max|diff| on finite cells = %.3e"
              % np.nanmax(np.abs(np.nan_to_num(m_r, neginf=0.0)
                                 - np.nan_to_num(m_o, neginf=0.0))))
    print("     (%d cells masked out of %d)" % (int(np.isneginf(m_r).sum()), m_r.size))

    # ── 6. apply_pfam_mask_row ─────────────────────────────────────────────
    rows_same = True
    for i, qid in enumerate(query_ids):
        a = R.apply_pfam_mask_row(sim[i], qid, target_ids_subset, ov_r)
        b = O.apply_pfam_mask_row(sim[i], qid, target_ids_subset, ov_o)
        rows_same = rows_same and np.array_equal(np.isneginf(a), np.isneginf(b)) \
            and np.array_equal(np.nan_to_num(a, neginf=0.0), np.nan_to_num(b, neginf=0.0))
    check("apply_pfam_mask_row", rows_same)

    print()
    print("RESULT:", "all identical" if ok else "MISMATCH — do not ship")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
