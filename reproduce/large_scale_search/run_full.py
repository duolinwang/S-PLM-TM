#!/usr/bin/env python
"""Re-encode all targets and queries, score, and compare full per-query metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tm-head", type=Path)
    args = ap.parse_args()
    data = json.loads((args.data / "provenance.json").read_text())
    args.work.mkdir(parents=True, exist_ok=True)
    cli = Path(__file__).with_name("search_cli.py")
    model_args = []
    # Prefer explicitly supplied weights; internal validation uses the exact
    # original files in the exported provenance. External users supply HF paths.
    reference = data["reference_model"]
    for flag, given, key in (("--config", args.config, "config"),
                              ("--checkpoint", args.checkpoint, "checkpoint"),
                              ("--tm-head", args.tm_head, "head")):
        value = given or Path(reference[key])
        if not value.is_file():
            ap.error(f"Supply {flag}: original file {value} is not available on this machine")
        model_args += [flag, str(value)]

    def run(*arguments):
        subprocess.run([sys.executable, *map(str, arguments)], check=True)

    index = args.work / "index"
    run(cli, "build", "--fasta", args.data / "target.fasta", "--index", index,
        "--mode", "tm", "--max-length", 0, "--batch-size", 8, "--max-tokens", 8192,
        "--shard-size", 20000, "--split-length", 1024,
        "--timing-json", args.work / "build_timing.json", *model_args)
    scores = args.work / "scores.npy"
    run(cli, "query", "--fasta", args.data / "query.fasta", "--index", index,
        "--out", args.work / "hits.tsv", "--mode", "tm", "--max-length", 0,
        "--query-batch-size", 1, "--query-block-size", 0,
        "--score-mode", "benchmark_tm", "--exclude-self", "--score-matrix", scores,
        "--warmup", "--benchmark-repeats", 3,
        "--save-query-embeddings", args.work / "query_embeddings.npy",
        "--timing-json", args.work / "query_timing.json",
        *model_args)
    run(Path(__file__).with_name("evaluate.py"), "--data", args.data,
        "--scores", scores, "--out", args.work / "evaluation")


if __name__ == "__main__":
    main()
