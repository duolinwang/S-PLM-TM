#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DEER Figure 7 打分 —— 自包含，不依赖研究仓库。

输入是两个 npz（`code/scripts/embed.py --modes tm` 的产物，含 `ids` 与 `emb_h`），
套上 EC head 之后检索、按 DEER 论文 STAR Methods 的定义打分。

    python score_figure7.py --templates tpl.npz --database db.npz --out out/
    python score_figure7.py ... --ec-head <别的 head.pt>      # 默认用 release 里的

写出 `metrics.json` / `metrics_len1024.json` / `metrics_len512.json` 并打印三行成绩。

指标定义（Liu et al., Cell Host & Microbe 2026, STAR Methods "Evaluation metrics"）
-----------------------------------------------------------------------------
设某个 EC family 有 n 个已知的细菌同工酶（真阳性），模板检索结果里它们的 1-based
排名为 ranks：

    Top-k Hit Ratio（全数据集） = （排名 ≤ k 的真阳性总数）/ 449
    Top-k Absolute Precision（逐 family） = （排名 ≤ k 的真阳性数）/ k
    Top-k Relative Precision（逐 family） = （排名 ≤ k 的真阳性数）/ min(k, n)
    Recall（逐 family）                   = （排名 ≤ n 的真阳性数）/ n

真值直接来自 `data/metadata.csv` 的两个布尔列：`is_human_template`（90 行）与
`is_bacteria_target`（449 行）；一条细菌蛋白是某模板的真阳性，当且仅当两者
`ec_number` 相同。

检索：EC head 输出 L2 归一化后按**欧氏距离升序**排（= 余弦降序），这就是它被
EC 检索头那个度量。

⚠️ mean / std 都是**逐 family 等权平均**（90 个 family 各算一个值再平均），
不是逐 query 平均。std 是总体标准差（除以 n，不是 n-1）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent.parent
KS = (10, 30, 100)


# --------------------------------------------------------------------- 真值
def _as_bool(v) -> bool:
    return str(v).strip().lower() == "true"


def load_metadata(path):
    """-> (templates, targets_by_ec)。templates 是 90 个人类模板的行。"""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    templates = [r for r in rows if _as_bool(r["is_human_template"])]
    targets_by_ec = defaultdict(set)
    for r in rows:
        if _as_bool(r["is_bacteria_target"]):
            targets_by_ec[r["ec_number"]].add(r["structure_id"])
    return templates, dict(targets_by_ec)


def read_fasta_lengths(paths):
    """-> {id: 序列长度}。id 取 header 第一个空格前的部分，与 metadata 的 structure_id 对应。"""
    out = {}
    for p in paths:
        name, buf = None, []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        out[name] = len("".join(buf))
                    name, buf = line[1:].split()[0], []
                else:
                    buf.append(line)
        if name is not None:
            out[name] = len("".join(buf))
    return out


# --------------------------------------------------------------------- 检索
def rank_database(tpl_ids, tpl_vec, db_ids, db_vec):
    """-> {template_id: [database_id, ...]}，按欧氏距离升序（rank 1 = 最近）。

    两侧都已 L2 归一化，所以 ||a-b||^2 = 2 - 2a·b，按内积降序 == 按距离升序。
    直接用内积既省内存又避免 cdist 的数值误差。
    """
    sim = np.asarray(tpl_vec, dtype=np.float32) @ np.asarray(db_vec, dtype=np.float32).T
    order = np.argsort(-sim, axis=1, kind="stable")
    db_ids = np.asarray(db_ids, dtype=object)
    return {tid: list(db_ids[order[i]]) for i, tid in enumerate(tpl_ids)}


# --------------------------------------------------------------------- 指标
def _mean_std(vals):
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": mean, "std": var ** 0.5, "n": len(vals)}


def compute_metrics(templates, targets_by_ec, ranking, ks=KS):
    total_tp = sum(len(targets_by_ec[t["ec_number"]]) for t in templates
                   if t["ec_number"] in targets_by_ec)
    hit_counts = {k: 0 for k in ks}
    recall_per_family, abs_prec, rel_prec = {}, {k: [] for k in ks}, {k: [] for k in ks}

    for t in templates:
        ec, tid = t["ec_number"], t["structure_id"]
        tp_set = targets_by_ec.get(ec, set())
        n = len(tp_set)
        if n == 0 or tid not in ranking:
            continue
        pos = {bid: i + 1 for i, bid in enumerate(ranking[tid])}
        ranks = sorted(pos[b] for b in tp_set if b in pos)
        for k in ks:
            n_le_k = sum(1 for r in ranks if r <= k)
            hit_counts[k] += n_le_k
            abs_prec[k].append(n_le_k / k)
            m = min(k, n)
            rel_prec[k].append(n_le_k / m if m > 0 else 0.0)
        recall_per_family[ec] = sum(1 for r in ranks if r <= n) / n

    return {
        "n_templates": len(templates),
        "n_tp_total": total_tp,
        "hit_ratio": {k: (hit_counts[k] / total_tp if total_tp else 0.0) for k in ks},
        "hit_counts": hit_counts,
        "recall": {**_mean_std(list(recall_per_family.values())),
                   "per_family": recall_per_family},
        "absolute_precision": {k: _mean_std(abs_prec[k]) for k in ks},
        "relative_precision": {k: _mean_std(rel_prec[k]) for k in ks},
    }


def score(templates, targets_by_ec, ranking, keep=None):
    """keep=None 是全集；给了就是长度子集（只留序列不超过阈值的 id）。"""
    if keep is None:
        return compute_metrics(templates, targets_by_ec, ranking)
    t2 = [t for t in templates if t["structure_id"] in keep]
    g2 = {ec: kept for ec, ids in targets_by_ec.items()
          if (kept := {i for i in ids if i in keep})}
    r2 = {k: [x for x in v if x in keep] for k, v in ranking.items() if k in keep}
    return compute_metrics(t2, g2, r2)


# --------------------------------------------------------------------- 入口
def load_split(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "emb_h" not in d and "emb_tm" not in d:
        raise SystemExit(f"{npz_path} 里没有 emb_h/emb_tm —— "
                         f"要用 `embed.py --modes tm` 的产物（EC head 吃的是 emb_h）")
    emb = d["emb_h"] if "emb_h" in d else d["emb_tm"]
    return [str(x) for x in d["ids"]], np.asarray(emb, dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", required=True, help="90 个模板的 npz（含 ids/emb_h）")
    ap.add_argument("--database", required=True, help="4,213 个库蛋白的 npz")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--ec-head", default=None,
                    help="EC head .pt（默认用 release 的 models/ec_head/）")
    ap.add_argument("--metadata", default=str(HERE / "data" / "metadata.csv"))
    ap.add_argument("--fasta", nargs="+",
                    default=[str(HERE / "data" / "templates.fasta"),
                             str(HERE / "data" / "database.fasta")])
    ap.add_argument("--no-size-check", action="store_true",
                    help="跳过 90/4213 的条数检查（只在调试子集时用）")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(RELEASE_ROOT / "code"))
    from splm_tm import heads, paths                                  # noqa: E402

    head_path = a.ec_head or paths.default_ec_head()
    head = heads.ECRetrievalHead.load(str(head_path), map_location="cpu")
    print(f">>> EC head: {head_path}")
    print(f"    hidden={head.cfg['hidden']}, seed={head.meta.get('seed')}, "
          f"参数 {sum(p.numel() for p in head.parameters()):,}")

    tpl_ids, tpl_h = load_split(a.templates)
    db_ids, db_h = load_split(a.database)
    print(f">>> templates {len(tpl_ids)}  database {len(db_ids)}")
    if not a.no_size_check and (len(tpl_ids) != 90 or len(db_ids) != 4213):
        raise SystemExit(f"期望 90 个模板 / 4,213 个库蛋白，拿到 "
                         f"{len(tpl_ids)}/{len(db_ids)} —— 编码那步不完整？")

    tpl_z, _ = heads.apply_ec_head(head, tpl_h)
    db_z, _ = heads.apply_ec_head(head, db_h)
    ranking = rank_database(tpl_ids, tpl_z, db_ids, db_z)

    templates, targets_by_ec = load_metadata(a.metadata)
    lengths = read_fasta_lengths(a.fasta)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [("metrics.json", None, "full"),
            ("metrics_len1024.json", 1024, "<=1024"),
            ("metrics_len512.json", 512, "<=512")]
    print(f"\n==== Figure 7 ====")
    for name, lim, label in todo:
        keep = None if lim is None else {i for i, L in lengths.items() if L <= lim}
        m = score(templates, targets_by_ec, ranking, keep)
        (out_dir / name).write_text(json.dumps(m, indent=2), encoding="utf-8")
        h = m["hit_ratio"]
        print(f"  {label:7s} Top-10 {h[10]:.4f}  Top-30 {h[30]:.4f}  Top-100 {h[100]:.4f}"
              f"   (真阳性 {m['n_tp_total']})")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
