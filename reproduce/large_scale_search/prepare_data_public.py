#!/usr/bin/env python
"""Build the large-scale data pack from the published benchmark on Hugging Face.

`prepare_data.py` exports the pack from the frozen work directory on the
cluster. Everything it reads is now published, in a different layout and with
the method column relabelled, so this script builds the same pack from the
public download instead:

    hf download deepduoduo/S-PLM-TM-benchmarks --repo-type dataset \\
      --include "large_scale/*" --local-dir /your/benchmarks

    python reproduce/large_scale_search/prepare_data_public.py \\
      --hf-dir /your/benchmarks/large_scale --out /your/data/large_scale

The pack it writes has the same contract as `prepare_data.py`'s: `target.fasta`,
`query.fasta`, `truth.jsonl.gz`, `reference_per_query.tsv` and
`provenance.json`. `run_full.py`, `score_from_embeddings.py` and `evaluate.py`
all consume it unchanged.

No structure alignment or label inference happens here; the ground truth is
copied from the published TM-align matrix. Only FASTA, TSV, gzipped TSV and
.npy are read, never with `allow_pickle`. Nothing is downloaded and no input is
modified.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import fasta_ids  # noqa: E402  same identifier parsing

N_QUERY, N_TARGET = 957, 430140
# The published method column; method_labels.tsv in the dataset maps it back to
# the frozen label that prepare_data.py matches.
PUBLISHED_METHOD = "S-PLM-TM"
# Released weight files, for provenance. They are not paths on this machine, so
# run_full.py will ask for --config/--checkpoint/--tm-head explicitly.
RELEASED_MODEL = dict(
    repository="https://huggingface.co/deepduoduo/S-PLM-TM",
    config="models/backbone/splm_tm_backbone.yaml",
    checkpoint="models/backbone/splm_tm_backbone.pth",
    head="models/tm_head/tm_calib_head.pt",
    max_length=0, split_len=1024, batch_size=8, max_tokens=8192,
    note="Full-length TM-head encoding; max_length=0 means no truncation and "
         "split_len partitions the library at 1024 residues for batching only.")

INPUTS = dict(target="library/protein.fasta",
              query="queries/scope/protein.fasta",
              truth="queries/scope/ss_mat.npy",
              identity="select/positive_identity.tsv.gz",
              reference="results/k1000_per_query.tsv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True, type=Path,
                    help="The downloaded large_scale/ directory")
    ap.add_argument("--out", required=True, type=Path,
                    help="New, empty output directory owned by you")
    ap.add_argument("--method", default=PUBLISHED_METHOD,
                    help="Method column to copy as the reference (default: %(default)s)")
    args = ap.parse_args(argv)
    hf, out = args.hf_dir.resolve(), args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error("Output must be new or empty; existing packs are never overwritten")
    if out == hf or hf in out.parents:
        ap.error("Write the pack outside the downloaded directory")
    paths = {}
    for key, name in INPUTS.items():
        path = hf / name
        if not path.is_file():
            ap.error(f"Missing {path}\nDownload large_scale/ from "
                     "https://huggingface.co/datasets/deepduoduo/S-PLM-TM-benchmarks")
        paths[key] = path

    db_ids, q_ids = fasta_ids(paths["target"]), fasta_ids(paths["query"])
    if len(db_ids) != N_TARGET or len(q_ids) != N_QUERY:
        raise ValueError(f"Unexpected dataset sizes: {len(db_ids)} targets, {len(q_ids)} queries")
    if len(set(db_ids)) != len(db_ids) or len(set(q_ids)) != len(q_ids):
        raise ValueError("Duplicate IDs")
    columns = {identifier: i for i, identifier in enumerate(db_ids)}

    identity = {identifier: {} for identifier in q_ids}
    with gzip.open(paths["identity"], "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["query"] in identity:
                identity[row["query"]][row["target"]] = float(row["identity"])

    truth = np.load(paths["truth"], mmap_mode="r", allow_pickle=False)
    if truth.shape != (N_QUERY, N_TARGET):
        raise ValueError(f"Unexpected truth matrix shape {truth.shape}")

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

    with paths["reference"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames)
        rows = [r for r in reader if r["method"] == args.method and r["group"] == "scope"
                and r["setting"] in ("A", "B")]
    if len(rows) != N_QUERY * 2:
        raise ValueError(f"Method {args.method!r} yielded {len(rows)} rows, expected "
                         f"{N_QUERY * 2}. The published method column is "
                         f"{PUBLISHED_METHOD!r}; see results/method_labels.tsv.")
    with (out / "reference_per_query.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    for key, name in (("target", "target.fasta"), ("query", "query.fasta")):
        shutil.copyfile(paths[key], out / name)

    manifest = dict(schema=1, queries=N_QUERY, targets=N_TARGET,
                    reference_method=args.method, query_ids=q_ids,
                    source="deepduoduo/S-PLM-TM-benchmarks, large_scale/ (CC BY 4.0)",
                    source_files={key: INPUTS[key] for key in INPUTS},
                    files=["target.fasta", "query.fasta", "truth.jsonl.gz",
                           "reference_per_query.tsv"],
                    reference_model=RELEASED_MODEL)
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"Standalone data pack from the published benchmark: {out}")
    print("Next, either score from the published embeddings (no GPU):")
    print("  python %s --data %s --embeddings %s --work /your/results/large_scale"
          % (Path(__file__).with_name("score_from_embeddings.py"), out, hf / "embeddings"))
    print("or re-encode everything with run_full.py and the released weights.")


if __name__ == "__main__":
    main()
