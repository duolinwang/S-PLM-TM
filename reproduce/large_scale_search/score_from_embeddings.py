#!/usr/bin/env python
"""Score the large-scale benchmark from the published embeddings, without a model.

`run_full.py` re-encodes all 430,140 targets and 957 queries on a GPU and then
scores them. The embeddings of that published run are released, so the scoring
and evaluation can be repeated on a CPU in minutes:

    hf download deepduoduo/S-PLM-TM-benchmarks --repo-type dataset \\
      --include "large_scale/embeddings/*" --local-dir /your/benchmarks

    python reproduce/large_scale_search/score_from_embeddings.py \\
      --data /your/data/large_scale \\
      --embeddings /your/benchmarks/large_scale/embeddings \\
      --work /your/results/large_scale

    python reproduce/large_scale_search/evaluate.py \\
      --data /your/data/large_scale --scores /your/results/large_scale/scores.npy \\
      --out /your/results/large_scale/evaluation

The scoring itself is `frozen_search.FrozenSearchIndex.search_embeddings` with
`score_mode="benchmark_tm"` — the same frozen code path `run_full.py` uses, so
the score definition (re-normalize, float32 inner product, `(1 + s) / 2` in
float64, back to float32) is not reimplemented here. Only the encoder is
replaced by the released vectors.

Nothing is downloaded and no input is modified. Only .npy, .txt and .json are
read; `allow_pickle` is never enabled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_search  # noqa: E402

DIMENSION = 1024
# L2 norms of the released vectors, as measured: within 1.1e-7 of one.
NORM_TOLERANCE = 1e-5


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def fasta_ids(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [line[1:].split()[0] for line in handle if line.startswith(">")]


def load_matrix(path, ids, what):
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.dtype != np.float32:
        raise ValueError(f"{what}: dtype is {arr.dtype}, expected float32")
    if arr.shape != (len(ids), DIMENSION):
        raise ValueError(f"{what}: shape {arr.shape}, expected {(len(ids), DIMENSION)}")
    rows = np.arange(len(ids)) if len(ids) <= 2000 else \
        np.sort(np.random.default_rng(0).choice(len(ids), 2000, replace=False))
    norms = np.linalg.norm(np.asarray(arr[rows], dtype=np.float64), axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > NORM_TOLERANCE:
        raise ValueError(f"{what}: vectors are not L2-normalized (worst |norm-1| = {worst:.3e})")
    print(f"{what}: {arr.shape} float32, |norm-1| <= {worst:.2e} over {len(rows)} rows")
    return arr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="Data pack from prepare_data.py or prepare_data_public.py")
    ap.add_argument("--embeddings", required=True, type=Path,
                    help="Directory holding db_emb.npy, query_scope.npy and their id lists")
    ap.add_argument("--work", required=True, type=Path, help="Output directory")
    ap.add_argument("--top-k", type=int, default=10, help="Hits per query in hits.tsv")
    ap.add_argument("--query-block-size", type=int, default=0,
                    help="0 is the published full query-matrix product and needs about 5 GB "
                         "of RAM. A positive value scores that many queries at a time; it "
                         "lowers the peak but lets BLAS pick a different kernel, which can "
                         "move the last bits.")
    args = ap.parse_args(argv)
    if args.top_k < 1 or args.query_block_size < 0:
        ap.error("top-k must be positive and query-block-size nonnegative")

    data = json.loads((args.data / "provenance.json").read_text(encoding="utf-8"))
    db_ids = fasta_ids(args.data / "target.fasta")
    q_ids = list(data["query_ids"])
    if len(db_ids) != data["targets"] or len(q_ids) != data["queries"]:
        raise ValueError("The data pack is inconsistent with its own provenance.json")
    if len(set(db_ids)) != len(db_ids):
        raise ValueError("Duplicate target identifiers in the data pack")

    # The released vectors ship their own id lists, so row order is checked
    # rather than assumed.
    for name, expected in (("db_ids.txt", db_ids), ("query_ids.txt", q_ids)):
        path = args.embeddings / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path}\nDownload large_scale/embeddings/ from "
                "https://huggingface.co/datasets/deepduoduo/S-PLM-TM-benchmarks")
        got = read_ids(path)
        if got != expected:
            where = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
            raise ValueError(f"{name} does not match the data pack "
                             f"(length {len(got)} vs {len(expected)}, first difference at {where})")
        print(f"{name}: {len(got)} identifiers, order identical to the data pack")

    db = load_matrix(args.embeddings / "db_emb.npy", db_ids, "db_emb.npy")
    queries = load_matrix(args.embeddings / "query_scope.npy", q_ids, "query_scope.npy")

    args.work.mkdir(parents=True, exist_ok=True)
    scores = args.work / "scores.npy"
    if scores.exists():
        raise FileExistsError(f"{scores} already exists; use a new work directory")
    # Built in memory from vectors that were just validated, so that the 1.6 GB
    # array is not copied into an index directory.
    index = frozen_search.FrozenSearchIndex(
        args.work / "index",
        dict(schema=frozen_search.FrozenSearchIndex.SCHEMA, status="complete",
             count=len(db_ids), dimension=DIMENSION,
             source="published embeddings, not encoded locally"),
        db_ids, db)
    print(f"scoring {len(q_ids)} queries against {len(db_ids)} targets "
          f"(benchmark_tm, block={args.query_block_size or 'full'})", flush=True)
    result = index.search_embeddings(queries, q_ids, top_k=args.top_k,
                                     score_mode="benchmark_tm", exclude_self=True,
                                     query_block_size=args.query_block_size or None,
                                     score_matrix=scores)
    result.write_tsv(args.work / "hits.tsv")
    (args.work / "score_timing.json").write_text(
        json.dumps(dict(timings=result.timings, top_k=args.top_k,
                        query_block_size=args.query_block_size or None,
                        embeddings=str(args.embeddings.resolve()),
                        note="Scored from the published embeddings; no encoder was run."),
                   indent=2) + "\n", encoding="utf-8")
    print(f"wrote {scores} and {args.work / 'hits.tsv'}")
    print("Next:\n  python %s --data %s --scores %s --out %s"
          % (Path(__file__).with_name("evaluate.py"), args.data, scores,
             args.work / "evaluation"))


if __name__ == "__main__":
    main()
