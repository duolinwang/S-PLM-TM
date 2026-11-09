#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`search.py` 与 Pfam 预筛的自测 —— 不需要 GPU、模型，也不需要装 PfamScan。

和 `check_pfam_parity.py` 的分工：
  · check_pfam_parity.py  对比研究仓库的原实现，证明"与发表数字同语义"。
    只有我们自己能跑（需要研究仓库）。
  · 本脚本              用合成数据验证对外行为，**任何人装好 release 就能跑**，
    用来确认安装没问题、预筛语义没被改坏。

    python code/scripts/selftest_search.py

期望：RESULT: all checks passed
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from splm_tm import pfam as P                              # noqa: E402

SEARCH = _HERE / "search.py"
NQ, NT, D = 8, 40, 16


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(7)
    q_ids = ["q%d" % i for i in range(NQ)]
    t_ids = ["t%d" % i for i in range(NT)]

    q_raw = rng.standard_normal((NQ, D)).astype(np.float32)
    t_raw = rng.standard_normal((NT, D)).astype(np.float32)
    q_npz, t_npz = tmp / "q.npz", tmp / "t.npz"
    np.savez(q_npz, ids=np.array(q_ids), emb_tm=unit(q_raw), emb_raw=q_raw)
    np.savez(t_npz, ids=np.array(t_ids), emb_tm=unit(t_raw), emb_raw=t_raw)

    # 前 5 个 query 有 Pfam 命中，后 3 个故意一个都没有 —— 覆盖全库回退那条路径
    FAMS = ["PF00069", "PF00028", "PF00001"]
    qj, tj = tmp / "q_pfam.json", tmp / "t_pfam.json"
    qj.write_text(json.dumps(
        {q: ([FAMS[i % 3]] if i < 5 else []) for i, q in enumerate(q_ids)}),
        encoding="utf-8")
    tj.write_text(json.dumps(
        {t: ([FAMS[i % 3]] if i % 4 else []) for i, t in enumerate(t_ids)}),
        encoding="utf-8")
    clans = tmp / "clans.tsv"
    clans.write_text("PF00069\tCL0016\tPkinase_clan\tPkinase\tdesc\n", encoding="utf-8")

    def run(*extra, out):
        cmd = [sys.executable, str(SEARCH), "--query", str(q_npz),
               "--target", str(t_npz), "--topk", str(NT), "--out", str(out)]
        cmd += [str(x) for x in extra]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit("search.py failed")
        return r.stdout

    def read_hits(path):
        hits = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
            q, rank, t, s = line.split("\t")
            hits.setdefault(q, []).append((int(rank), t, float(s)))
        return hits

    state = {"ok": True}

    def check(name, cond, note=""):
        state["ok"] = state["ok"] and bool(cond)
        print("  %-50s %s %s" % (name, "PASS" if cond else "*** FAIL ***", note))

    sim = unit(q_raw) @ unit(t_raw).T
    want = {q_ids[i]: [t_ids[j] for j in np.argsort(-sim[i])] for i in range(NQ)}

    # 1) 不预筛 == 直接内积排序
    out_full = tmp / "full.tsv"
    run(out=out_full)
    hits_full = read_hits(out_full)
    check("no prefilter: ranking == inner-product argsort",
          all([t for _, t, _ in hits_full[q]] == want[q] for q in q_ids))
    check("no prefilter: all queries, full depth",
          all(len(hits_full[q]) == NT for q in q_ids))

    # 2) pfamclan（用现成 JSON，不需要 PfamScan）
    out_pf, mat = tmp / "pfamclan.tsv", tmp / "m.npz"
    log = run("--prefilter", "pfamclan", "--query-pfam-json", qj,
              "--target-pfam-json", tj, "--pfam-clan-tsv", clans,
              "--save-matrix", mat, out=out_pf)
    hits_pf = read_hits(out_pf)

    fam2clan = P.load_clan_map(clans)
    ov = P.build_pfam_overlap_map(P.load_pfam_sets(qj, True, fam2clan),
                                  P.load_pfam_sets(tj, True, fam2clan), t_ids)
    masked = P.apply_pfam_mask(sim, q_ids, t_ids, ov, verbose=False)
    with_c = [q for q in q_ids if ov.get(q)]
    without_c = [q for q in q_ids if not ov.get(q)]

    check("pfamclan: candidate queries return only candidates",
          all(set(t for _, t, _ in hits_pf[q]) <= ov[q] for q in with_c),
          "(%d with candidates)" % len(with_c))
    check("pfamclan: unannotated queries fall back to full library",
          all([t for _, t, _ in hits_pf[q]] == want[q] for q in without_c),
          "(%d fell back)" % len(without_c))
    check("pfamclan: no -inf cell leaks into the hit table",
          all(np.isfinite(s) for q in q_ids for _, _, s in hits_pf[q]))
    check("pfamclan: depth == number of candidates",
          all(len(hits_pf[q]) == len(ov[q]) for q in with_c))

    d = np.load(mat, allow_pickle=True)
    got = d["scores"]
    check("--save-matrix == pfam.apply_pfam_mask (bitwise)",
          np.array_equal(np.isneginf(got), np.isneginf(masked))
          and np.allclose(np.nan_to_num(got, neginf=0.0),
                          np.nan_to_num(masked, neginf=0.0), atol=0, rtol=0))
    check("--save-matrix carries ids",
          list(d["query_ids"]) == q_ids and list(d["target_ids"]) == t_ids)

    # 3) emb_raw 的归一化提醒与行为
    warn = run("--emb", "emb_raw", out=tmp / "rawnn.tsv")
    check("emb_raw without --normalize warns", "[!]" in warn)
    out_rawn = tmp / "rawn.tsv"
    run("--emb", "emb_raw", "--normalize", out=out_rawn)
    check("emb_raw + --normalize == cosine ranking",
          all([t for _, t, _ in read_hits(out_rawn)[q]] == want[q] for q in q_ids))

    print()
    print("RESULT:", "all checks passed" if state["ok"] else "FAILURES PRESENT")
    return 0 if state["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
