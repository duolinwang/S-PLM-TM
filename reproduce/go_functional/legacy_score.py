#!/usr/bin/env python
"""GO-MF Wang BMA 打分：逻辑照抄官方 compute_function_BMA.py（对称 BMA、seed 42
random padding、topk 10），只加了三件事：term-pair 相似度缓存（数学不变、纯提速）、
per-pair CSV 输出、以及 --subset（高置信 GO 子集）过滤。

    python score_go_bma.py --search_result search_splm_tmft.tsv \
        --go_json  protein-search-benchmark/Functional_consistency/GO.json \
        --obo      go-basic.obo \
        --subset   protein-search-benchmark/Functional_consistency/go_high_conf_subset.lst \
        --out_csv  scores_splm_tmft.csv

汇总同时给全集和子集（query 与 target 都在子集里的 pair）两套 median/mean。
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
from tqdm import tqdm

random.seed(42)
TOPK = 10


def read_search_result_tsv(file_path):
    results = defaultdict(list)
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            q, t = parts[:2]
            q = q[:-4] if q.endswith(".pdb") else q
            t = t[:-4] if t.endswith(".pdb") else t
            results[q].append(t)
    return results


def random_padding(result_dict, query_list, target_list):
    padding_dict = {}
    for q in query_list:
        if q not in result_dict:
            padding_dict[q] = random.sample(target_list, k=TOPK)
        elif len(result_dict[q]) < TOPK:
            available = list(set(target_list) - set(result_dict[q]))
            padding_dict[q] = result_dict[q] + random.sample(available, k=TOPK - len(result_dict[q]))
        else:
            padding_dict[q] = result_dict[q][:TOPK]
    return padding_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search_result", required=True)
    ap.add_argument("--go_json", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--subset", default=None, help="高置信蛋白列表（可选）")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    go_anno_raw = json.load(open(args.go_json))
    godag = get_godag(args.obo, optional_attrs={"relationship"})
    relationships = {"part_of"}
    cache = {}

    # GO.json 里在当前 obo 已废弃的 term 会让 SsWang 返回 None —— 照论文预处理
    # （"excluded proteins annotated with GO terms removed in the current GO
    # release"）把不在 DAG 里的 term 过滤掉；term 全被滤掉的蛋白整体剔除。
    go_anno, n_drop_t, n_drop_p = {}, 0, 0
    for p, terms in go_anno_raw.items():
        keep = [t for t in terms if t in godag]
        n_drop_t += len(terms) - len(keep)
        if keep:
            go_anno[p] = keep
        else:
            n_drop_p += 1
    print(f"obo 过滤：丢弃废弃 term {n_drop_t} 个，剔除蛋白 {n_drop_p} 个")

    def pairsim(a, b):
        key = (a, b) if a <= b else (b, a)
        if key not in cache:
            s = SsWang([a, b], godag, relationships).get_sim(a, b)
            cache[key] = 0.0 if s is None else s
        return cache[key]

    search_result = read_search_result_tsv(args.search_result)
    query_list = list(search_result.keys())
    target_list = list(go_anno.keys())
    padded = random_padding(search_result, query_list, target_list)
    print(f"queries {len(query_list)}, GO.json targets {len(target_list)}")

    scores = []
    for q, t_list in tqdm(padded.items(), desc="BMA"):
        if q not in go_anno:
            continue
        go_q = go_anno[q]
        for t in t_list:
            if t not in go_anno:
                continue
            go_t = go_anno[t]
            bma_qt = np.mean([max(pairsim(g1, g2) for g2 in go_t) for g1 in go_q])
            bma_tq = np.mean([max(pairsim(g1, g2) for g1 in go_q) for g2 in go_t])
            scores.append((q, t, (bma_qt + bma_tq) / 2))

    with open(args.out_csv, "w") as f:
        f.write("query,target,go_similarity\n")
        for q, t, s in scores:
            f.write(f"{q},{t},{s:.6f}\n")

    vals = np.array([s for _, _, s in scores])
    print(f"\n== FULL  pairs={len(vals)}  median={np.median(vals):.4f}  "
          f"mean={vals.mean():.4f}")
    per_q = defaultdict(list)
    for q, _, s in scores:
        per_q[q].append(s)
    qm = np.array([np.mean(v) for v in per_q.values()])
    print(f"   per-query mean-of-top10: median={np.median(qm):.4f}  mean={qm.mean():.4f}")

    if args.subset:
        sub = {line.strip() for line in open(args.subset) if line.strip()}
        sv = np.array([s for q, t, s in scores if q in sub and t in sub])
        print(f"== HIGH-CONF subset  pairs={len(sv)}  "
              f"median={np.median(sv):.4f}  mean={sv.mean():.4f}" if len(sv)
              else "== HIGH-CONF subset: 0 pairs")
    print(f"\nper-pair CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
