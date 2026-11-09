#!/usr/bin/env python
"""Export the frozen SCOPe-only large-scale benchmark to a standalone data pack.

Run this once on a compute node with access to the original frozen work directory.
No structure alignment or label inference is performed; existing truth is copied.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np


def fasta_ids(path):
    with Path(path).open() as handle:
        return [line[1:].split()[0] for line in handle if line.startswith(">")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-work", type=Path, required=True)
    ap.add_argument("--reference-tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    work, out = args.source_work.resolve(), args.out.resolve()
    if not args.reference_tag.endswith("__full"):
        ap.error("This reproduction targets the final full-length model")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error("Output must be new or empty; existing packs are never overwritten")
    if out == work or work in out.parents:
        ap.error("Export outside the read-only source tree")
    target = work / "db/protein.fasta"
    query = work / "queries/orig100/scope/protein.fasta"
    db_ids, q_ids = fasta_ids(target), fasta_ids(query)
    if len(db_ids) != 430140 or len(q_ids) != 957:
        raise ValueError("Unexpected frozen dataset sizes")
    if len(set(db_ids)) != len(db_ids) or len(set(q_ids)) != len(q_ids):
        raise ValueError("Duplicate IDs")
    columns = {identifier: i for i, identifier in enumerate(db_ids)}
    identity = {identifier: {} for identifier in q_ids}
    with gzip.open(work / "select/positive_identity.tsv.gz", "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["query"] in identity:
                identity[row["query"]][row["target"]] = float(row["identity"])
    truth = np.load(work / "queries/orig100/scope/ss_mat.npy", mmap_mode="r")
    if truth.shape != (957, 430140):
        raise ValueError("Unexpected truth matrix shape")
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "truth.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i, query_id in enumerate(q_ids):
            positives = np.flatnonzero(truth[i] > 0.5).tolist()
            low = [c for c in positives if identity[query_id][db_ids[c]] < 30]
            if len(low) < 10:
                raise ValueError(f"{query_id}: fewer than ten low-identity positives")
            record = dict(query=query_id, positive=positives, low_identity=low,
                          self_column=columns.get(query_id))
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    reference = work / "preview/results/splm_length/k1000_per_query.tsv"
    method = f"SPLM full [{args.reference_tag}]"
    with reference.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames)
        rows = [r for r in reader if r["method"] == method and r["group"] == "scope"
                and r["setting"] in ("A", "B")]
    if len(rows) != 957 * 2:
        raise ValueError(f"Reference method {method!r} did not yield 1,914 rows")
    with (out / "reference_per_query.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    for source, name in ((target, "target.fasta"), (query, "query.fasta")):
        if source.resolve() != (out / name).resolve():
            shutil.copyfile(source, out / name)
    manifest = dict(schema=1, queries=957, targets=430140, reference_method=method,
                    query_ids=q_ids, source_work=str(work.resolve()),
                    files=["target.fasta", "query.fasta", "truth.jsonl.gz", "reference_per_query.tsv"])
    meta = work / "preview/splm" / args.reference_tag / "meta.json"
    manifest["reference_model"] = {key: value for key, value in json.loads(meta.read_text()).items()
                                   if key in ("tag", "config", "checkpoint", "head", "max_length",
                                              "split_len", "repo_commit", "slurm_job_id")}
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Standalone data pack: {out}")


if __name__ == "__main__":
    main()
