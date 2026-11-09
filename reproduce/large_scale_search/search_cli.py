#!/usr/bin/env python
"""Build the frozen large-scale database from FASTA, or search it with sequence queries.

Separate processes keep model loading and fingerprinting outside online timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_search import FrozenSearchIndex, FrozenT5Encoder  # noqa: E402


def load_encoder(args):
    return FrozenT5Encoder.load(config_path=args.config, ckpt_path=args.checkpoint,
                                tm_head_path=args.tm_head, device=args.device,
                                head_batch_size=args.head_batch_size)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    for name in ("build", "query"):
        sub = commands.add_parser(name)
        sub.add_argument("--fasta", type=Path, required=True)
        sub.add_argument("--index", type=Path, required=True)
        sub.add_argument("--mode", choices=("tm",), default="tm")
        sub.add_argument("--config", type=Path)
        sub.add_argument("--checkpoint", type=Path)
        sub.add_argument("--tm-head", type=Path)
        sub.add_argument("--device")
        sub.add_argument("--head-batch-size", type=int, default=4096)
        sub.add_argument("--max-length", type=int, default=0, help="0=full length; positive=truncate")
        sub.add_argument("--max-tokens", type=int, default=8192)
        sub.add_argument("--timing-json", type=Path, help="Write separate stage timings")
        if name == "build":
            sub.add_argument("--batch-size", type=int, default=8)
            sub.add_argument("--shard-size", type=int, default=20000)
            sub.add_argument("--split-length", type=int, default=1024, help="0=no short/long partition")
            sub.add_argument("--no-resume", action="store_true")
        else:
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--topk", type=int, default=10)
            sub.add_argument("--query-batch-size", type=int, default=1)
            sub.add_argument("--query-block-size", type=int, default=0, help="0=one full scoring matrix")
            sub.add_argument("--score-mode", choices=("benchmark_tm",), default="benchmark_tm")
            sub.add_argument("--exclude-self", action="store_true")
            sub.add_argument("--score-matrix", type=Path)
            sub.add_argument("--save-query-embeddings", type=Path)
            sub.add_argument("--warmup", action="store_true", help="Warm one query; report separately")
            sub.add_argument("--benchmark-repeats", type=int, default=0,
                             help="Also repeat original speed-only scoring; 0 disables")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.max_length < 0 or args.max_tokens < 1:
        ap.error("--max-length must be nonnegative; --max-tokens must be positive")
    if args.command == "build" and min(args.batch_size, args.shard_size) < 1:
        ap.error("--batch-size and --shard-size must be positive")
    if args.command == "build" and args.split_length < 0:
        ap.error("--split-length must be nonnegative")
    if args.command == "query" and (min(args.topk, args.query_batch_size) < 1
                                     or args.query_block_size < 0 or args.benchmark_repeats < 0):
        ap.error("Invalid query batch, block or topk limit")
    # Loading an index before a model gives early errors for invalid index files.
    started = time.perf_counter()
    index = FrozenSearchIndex.load(args.index) if args.command == "query" else None
    timings = dict(index_load_sec=time.perf_counter() - started if index is not None else 0.0)
    started = time.perf_counter()
    encoder = load_encoder(args)
    encoder.synchronize()
    timings["encoder_setup_sec"] = time.perf_counter() - started
    timings["encoder_setup_scope"] = "model loading and model/code fingerprinting; excluded from online time"
    if args.command == "build":
        started = time.perf_counter()
        index = FrozenSearchIndex.build_fasta(
            encoder, args.fasta, args.index, max_length=args.max_length,
            batch_size=args.batch_size, max_tokens=args.max_tokens, shard_size=args.shard_size,
            split_length=args.split_length or None, resume=not args.no_resume)
        timings["build_command_sec"] = time.perf_counter() - started
        timings.update(index.metadata.get("timings", {}))
        print(f"index ready: {len(index.ids)} sequences, dimension={index.embeddings.shape[1]}")
    else:
        result = index.search_fasta(
            encoder, args.fasta, top_k=args.topk, max_length=args.max_length,
            query_batch_size=args.query_batch_size, max_tokens=args.max_tokens,
            score_mode=args.score_mode, exclude_self=args.exclude_self,
            query_block_size=args.query_block_size or None, score_matrix=args.score_matrix,
            warmup=args.warmup)
        result.write_tsv(args.out)
        timings.update(result.timings)
        if args.save_query_embeddings:
            args.save_query_embeddings.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_query_embeddings, result.query_embeddings, allow_pickle=False)
        if args.benchmark_repeats:
            speed = index.benchmark_scoring(result.query_embeddings, args.out.parent,
                                            repeats=args.benchmark_repeats)
            timings["original_scoring_profile"] = speed
            timings["original_profile_online_sec"] = (result.timings["query_encoding_sec"]
                                                        + speed["median_sec"])
        print(f"hits written: {args.out}")
    if args.timing_json:
        args.timing_json.parent.mkdir(parents=True, exist_ok=True)
        args.timing_json.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
        print(f"timings written: {args.timing_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
