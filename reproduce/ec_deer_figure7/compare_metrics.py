#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""比对两份 Figure 7 metrics.json，判定是否复现。

    python compare_metrics.py <实得.json> <参考.json>

退出码 0 = 复现，1 = 对不上。

判定口径：`hit_ratio` / `hit_counts` / `recall.per_family` 必须**完全相等**
（它们是计数和逐 family 的值，没有求和顺序问题）。
汇总统计量（recall.mean/std、各档 precision 的 mean/std）允许 **1e-12** 以内的差 ——
它们是对 90 个 family 求和得到的，求和顺序不同会在 float64 末位（1e-16~1e-17）产生
差异，那不是真实差异。2026-10-14 的实测就落在这个量级。
"""
from __future__ import annotations

import json
import sys

TOL = 1e-12


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) != 2:
        print(__doc__)
        return 2
    got, want = (load(p) for p in argv)

    ok = True
    print("==== 必须完全相等 ====")
    for k in ("10", "30", "100"):
        g, w = got["hit_ratio"][k], want["hit_ratio"][k]
        gc, wc = got["hit_counts"][k], want["hit_counts"][k]
        same = (g == w) and (gc == wc)
        ok &= same
        print(f"  Top-{k:<4} hit_ratio {g:.6f} vs {w:.6f}   "
              f"命中 {gc} vs {wc}   {'OK' if same else 'MISMATCH'}")

    gf, wf = got["recall"].get("per_family", {}), want["recall"].get("per_family", {})
    if set(gf) != set(wf):
        ok = False
        print(f"  per_family 的 family 集合不同：实得 {len(gf)} 个，参考 {len(wf)} 个")
    else:
        bad = [k for k in wf if gf[k] != wf[k]]
        ok &= not bad
        print(f"  recall.per_family  {len(wf)} 个 family  "
              f"{'全部相等 OK' if not bad else f'有 {len(bad)} 个不同 MISMATCH: {bad[:5]}'}")

    print(f"==== 允许 {TOL:g} 以内（求和顺序造成的 float64 末位舍入）====")
    worst = 0.0
    for sect in ("recall", "absolute_precision", "relative_precision"):
        for key, val in sorted(got[sect].items()):
            if key == "per_family":
                continue
            if isinstance(val, dict):
                for stat in ("mean", "std"):
                    if stat not in val:
                        continue
                    d = abs(val[stat] - want[sect][key][stat])
                    worst = max(worst, d)
                    if d > TOL:
                        ok = False
                        print(f"  {sect}/{key}/{stat}: 差 {d:.3e} 超出容差 MISMATCH")
            elif isinstance(val, (int, float)):
                d = abs(val - want[sect][key])
                worst = max(worst, d)
                if d > TOL:
                    ok = False
                    print(f"  {sect}/{key}: 差 {d:.3e} 超出容差 MISMATCH")
    print(f"  最大偏差 {worst:.3e}   {'OK' if worst <= TOL else 'MISMATCH'}")

    print("\nPASS 复现" if ok else "\nFAIL 对不上 —— 先查哪一步不同，别拿它发布")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
