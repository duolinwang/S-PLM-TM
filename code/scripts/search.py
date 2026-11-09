#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索：embedding -> 排名，可选 Pfam 预筛。

这是 `embed.py` 的下游。`embed.py` 把序列变成向量，这里把向量变成每个 query 的
排名表 —— 也就是 `models/MODEL_CARD.md` 那些 benchmark 数字的产生方式。

两档口径（MODEL_CARD 的数字分别对应哪一档，见那份文件）
------------------------------------------------------
    不带预筛      python code/scripts/search.py --query q.npz --target t.npz --out hits.tsv
    +Pfam clan    加 --prefilter pfamclan   （SCOPe 远同源 / 细菌 EC 用这一档）
    +Pfam family  加 --prefilter pfamfam

预筛只需先跑一次 `bash code/scripts/setup_pfam.sh`，之后不用带路径参数：
clans.tsv 随包，HMM 库与 PfamScan 装在默认位置 `<release>/pfam_data/`。

典型用法
--------
    # 1) 先编码（检索用 tm，EC/功能检索用 ec）
    python code/scripts/embed.py --fasta queries.fasta --modes raw tm --out q.npz
    python code/scripts/embed.py --fasta targets.fasta --modes raw tm --out t.npz

    # 2) 再检索
    python code/scripts/search.py --query q.npz --target t.npz --topk 10 --out hits.tsv

    # 3) 要 Pfam 预筛这一档。先一次性 `bash code/scripts/setup_pfam.sh`，
    #    之后不用带任何路径参数（clans.tsv 随包，HMM 库装在默认位置）
    python code/scripts/search.py --query q.npz --target t.npz \
        --query-fasta queries.fasta --target-fasta targets.fasta \
        --prefilter pfamclan --out hits_pfamclan.tsv

    # 已经有别处算好的 Pfam JSON，就不必装 PfamScan
    ... --prefilter pfamclan --query-pfam-json q.json --target-pfam-json t.json

⚠️ `--emb` 默认 `emb_tm`：TM 校准头的输出**已经 L2 归一化**，内积就是预测 TM-score，
   所以这里**不再归一化**。换成 `emb_raw` 时它没归一化，需要 `--normalize`
   （不加就是在比未归一化的内积，排名会不同）。EC 检索用 `--emb emb_ec`。

⚠️ Pfam 预筛是**算完相似度矩阵之后的逐行 mask**，不省编码时间，只改排名；
   某个 query 一个 Pfam 都没命中时它那一行退回全库。这两点与产出已发表数字的实现
   一致，细节见 `splm_tm/pfam.py` 的模块说明。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))


def load_npz(path: Path, key: str):
    d = np.load(path, allow_pickle=True)
    if key not in d:
        raise KeyError(
            "%s 里没有 key %r，现有的是 %s。\n"
            "用 embed.py 的 --modes 产生需要的那种："
            "emb_raw<-raw / emb_tm<-tm / emb_ec<-ec。"
            % (path, key, sorted(d.keys())))
    ids = [str(x) for x in d["ids"]]
    emb = np.asarray(d[key], dtype=np.float32)
    if len(ids) != emb.shape[0]:
        raise ValueError("%s: ids (%d) 与 %s (%d) 行数不一致"
                         % (path, len(ids), key, emb.shape[0]))
    return ids, emb


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Retrieve targets for each query from S-PLM-TM embeddings, "
                    "optionally with a Pfam prefilter.")
    ap.add_argument("--query", required=True, type=Path, help="query 的 npz（embed.py 产出）")
    ap.add_argument("--target", required=True, type=Path, help="target 库的 npz")
    ap.add_argument("--emb", default="emb_tm",
                    choices=["emb_tm", "emb_h", "emb_raw", "emb_ec", "emb_ec_raw"],
                    help="用哪种向量（默认 emb_tm：已归一化，内积=预测 TM-score）")
    ap.add_argument("--normalize", action="store_true",
                    help="打分前先 L2 归一化。emb_raw / emb_ec_raw 需要；emb_tm / emb_ec 不需要")
    ap.add_argument("--metric", default="inner", choices=["inner", "euclidean"],
                    help="inner=内积（默认）；euclidean=负欧氏距离（越大越近）")
    ap.add_argument("--topk", type=int, default=10, help="每个 query 输出前几名（默认 10）")
    ap.add_argument("--out", type=Path, default=None, help="排名表 TSV 输出路径")
    ap.add_argument("--save-matrix", type=Path, default=None,
                    help="另存完整（掩码后的）分数矩阵为 npz")

    ap.add_argument("--prefilter", default="full", choices=["full", "pfamfam", "pfamclan"],
                    help="full=不预筛（默认）；pfamfam=共享 Pfam family；pfamclan=共享 clan")
    ap.add_argument("--query-fasta", type=Path, default=None, help="现跑 PfamScan 时需要")
    ap.add_argument("--target-fasta", type=Path, default=None, help="现跑 PfamScan 时需要")
    ap.add_argument("--pfam-data-root", type=Path, default=None,
                    help="含 PfamScan/ 与 Pfam_db/ 的目录。默认 <release>/pfam_data，"
                         "即 setup_pfam.sh 装的位置；已有 Pfam JSON 时用不到")
    ap.add_argument("--pfam-clan-tsv", type=Path, default=None,
                    help="Pfam-A.clans.tsv。默认用随包的 Pfam 36.0 那份；"
                         "⚠️ 换版本会改变 clan 归属从而改变复现结果")
    ap.add_argument("--query-pfam-json", type=Path, default=None, help="已算好的 query Pfam JSON")
    ap.add_argument("--target-pfam-json", type=Path, default=None, help="已算好的 target Pfam JSON")
    ap.add_argument("--pfam-cache-dir", type=Path, default=None,
                    help="Pfam JSON 缓存目录（默认 <out 所在目录>/pfam_cache）")
    args = ap.parse_args()

    q_ids, q_emb = load_npz(args.query, args.emb)
    t_ids, t_emb = load_npz(args.target, args.emb)
    print("queries %d x %d | targets %d x %d | emb=%s"
          % (q_emb.shape[0], q_emb.shape[1], t_emb.shape[0], t_emb.shape[1], args.emb))
    if q_emb.shape[1] != t_emb.shape[1]:
        raise ValueError("query 与 target 的维度不一致，很可能来自不同 checkpoint")

    if args.normalize:
        q_emb = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12)
        t_emb = t_emb / np.maximum(np.linalg.norm(t_emb, axis=1, keepdims=True), 1e-12)
    elif args.emb in ("emb_raw", "emb_ec_raw"):
        print("  [!] %s 未归一化，而没有加 --normalize；内积不是余弦，排名会与"
              "已发表口径不同。" % args.emb)

    if args.metric == "inner":
        sim = q_emb @ t_emb.T
    else:
        d2 = ((q_emb ** 2).sum(1)[:, None] + (t_emb ** 2).sum(1)[None, :]
              - 2.0 * (q_emb @ t_emb.T))
        sim = -np.sqrt(np.maximum(d2, 0.0))

    if args.prefilter != "full":
        from splm_tm import pfam as P                     # noqa: E402
        cache = (args.pfam_cache_dir
                 or ((args.out.parent if args.out else Path(".")) / "pfam_cache"))
        need_fasta = not (args.query_pfam_json and args.target_pfam_json)
        if need_fasta and not (args.query_fasta and args.target_fasta):
            ap.error("--prefilter 需要 --query-fasta/--target-fasta（现跑 PfamScan），"
                     "或者 --query-pfam-json/--target-pfam-json（用已有注释）")
        # 默认就用随包/一键装好的位置，所以 --prefilter 可以单独使用
        root = args.pfam_data_root
        if root is None and need_fasta:
            root = P.DEFAULT_PFAM_DATA_ROOT
        overlap = P.prepare_pfam_overlap(
            query_fasta=args.query_fasta,
            target_fasta=args.target_fasta,
            target_ids=t_ids,
            pfam_data_root=root,
            cache_dir=cache,
            use_clan=(args.prefilter == "pfamclan"),
            clan_tsv=args.pfam_clan_tsv,
            query_json=args.query_pfam_json,
            target_json=args.target_pfam_json,
        )
        sim = P.apply_pfam_mask(sim, q_ids, t_ids, overlap)

    k = min(args.topk, len(t_ids))
    # -inf 的格子被预筛排除掉了，不该出现在排名里
    order = np.argsort(-sim, axis=1)[:, :k]

    lines = ["query\trank\ttarget\tscore"]
    for i, qid in enumerate(q_ids):
        r = 0
        for j in order[i]:
            s = sim[i, j]
            if not np.isfinite(s):
                continue
            r += 1
            lines.append("%s\t%d\t%s\t%.6f" % (qid, r, t_ids[j], s))
    body = "\n".join(lines) + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        print("wrote %s (%d rows)" % (args.out, len(lines) - 1))
    else:
        sys.stdout.write(body)

    if args.save_matrix:
        args.save_matrix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_matrix, scores=sim,
                            query_ids=np.array(q_ids), target_ids=np.array(t_ids))
        print("wrote %s" % args.save_matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
