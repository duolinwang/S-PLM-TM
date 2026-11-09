#!/usr/bin/env python
"""Evaluate dense full-database scores using the frozen A/B protocol.

The numerical operations match plmsearch_k1000_eval.ranked_metrics. No baseline
code, structure files or TM-align execution are required with the prepared pack.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def metrics(scores, positives, ignored):
    order = np.argsort(-scores, kind="stable")
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    ends = np.append(np.flatnonzero(np.diff(scores[order]) != 0), len(order) - 1)
    pos = np.zeros(len(order), dtype=np.int32)
    pos[inverse[positives]] = 1
    keep = np.ones(len(order), dtype=np.int32)
    keep[inverse[ignored]] = 0
    pos *= keep
    count = int(pos.sum())
    if not count:
        raise ValueError("The frozen protocol requires positives for every query")
    tp, total = np.cumsum(pos)[ends], np.cumsum(keep)[ends]
    precision = np.divide(tp, total, out=np.zeros(len(ends), dtype=np.float64), where=total > 0)
    recall = tp / count
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    kept = pos[keep.astype(bool)]
    return dict(n_pos=count, AP=ap, **{"P@1": float(kept[:1].sum()),
                                     "P@10": float(kept[:10].sum() / 10)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.data / "provenance.json").read_text())
    score_meta = json.loads(args.scores.with_suffix(args.scores.suffix + ".json").read_text())
    with (args.data / "target.fasta").open() as handle:
        target_ids = [line[1:].split()[0] for line in handle if line.startswith(">")]
    if score_meta["query_ids"] != data["query_ids"] or score_meta["target_ids"] != target_ids:
        raise ValueError("Score rows/columns do not match the frozen benchmark")
    if score_meta["score_mode"] != "benchmark_tm":
        raise ValueError("This case requires the original benchmark_tm score definition")
    matrix = np.load(args.scores, mmap_mode="r")
    if matrix.shape != (data["queries"], data["targets"]):
        raise ValueError("Wrong score matrix dimensions")
    rows = []
    with gzip.open(args.data / "truth.jsonl.gz", "rt") as handle:
        for i, line in enumerate(handle):
            truth = json.loads(line)
            if truth["query"] != data["query_ids"][i] or not np.isfinite(matrix[i]).all():
                raise ValueError("Invalid query order or nonfinite scores")
            own = [] if truth["self_column"] is None else [truth["self_column"]]
            positives = np.asarray(truth["positive"], dtype=np.int64)
            low = np.asarray(truth["low_identity"], dtype=np.int64)
            removed = np.setdiff1d(positives, low)
            for setting, pos, ignored in (("A", positives, own), ("B", low, own + removed.tolist())):
                row = dict(query=truth["query"], setting=setting,
                           **metrics(matrix[i], pos, np.asarray(ignored, dtype=np.int64)))
                rows.append(row)
    if len(rows) != data["queries"] * 2:
        raise ValueError("Incomplete ground truth")
    with (args.data / "reference_per_query.tsv").open() as handle:
        reference = {(r["query"], r["setting"]): r for r in csv.DictReader(handle, delimiter="\t")}
    differences = []
    for row in rows:
        ref = reference[(row["query"], row["setting"])]
        for key in ("n_pos", "AP", "P@1", "P@10"):
            if row[key] != float(ref[key]):
                differences.append(dict(query=row["query"], setting=row["setting"], metric=key,
                                        reference=float(ref[key]), actual=row[key],
                                        difference=row[key] - float(ref[key])))
    summary = {setting: {("MAP" if key == "AP" else key):
                         float(np.mean([r[key] for r in rows if r["setting"] == setting]))
                         for key in ("AP", "P@1", "P@10")} for setting in ("A", "B")}
    report = dict(queries=data["queries"], summary=summary, exact_match=not differences,
                  differences=differences)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_query.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "differences"}, indent=2))
    if differences:
        raise SystemExit(f"NOT VERIFIED: {len(differences)} per-query values differ; see comparison.json")


if __name__ == "__main__":
    main()
