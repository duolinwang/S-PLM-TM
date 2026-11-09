#!/usr/bin/env python
"""Compare original and release inference in separate processes on real sequences.

Use the accompanying SLURM script. This is a preflight check, not a substitute
for reproducing the full 957-query benchmark and its per-query metrics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def read_fasta(path):
    ids, seqs = [], []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
                seqs.append("")
            elif line.strip():
                seqs[-1] += line.strip()
    return ids, seqs


def sample(ids, seqs, per_bin):
    rng = np.random.default_rng(0)
    layers = np.searchsorted([128, 256, 512, 1024, 2048], [len(s) for s in seqs], side="left")
    chosen = []
    for layer in range(6):
        candidates = np.flatnonzero(layers == layer)
        chosen.extend(rng.choice(candidates, min(len(candidates), per_bin), replace=False).tolist())
    chosen.sort()
    return dict(ids=[ids[i] for i in chosen], seqs=[seqs[i] for i in chosen], indices=chosen,
                lengths=[len(seqs[i]) for i in chosen])


def model_info(args):
    path = args.source_work / "preview/splm" / args.reference_tag / "meta.json"
    return json.loads(path.read_text())


def original(args, selected):
    import torch

    meta = model_info(args)
    old_repo = Path(meta["config"]).parents[2]
    sys.path.insert(0, str(old_repo))
    os.environ["SCOPE_EXTRA_PATH"] = str(args.research_repo / "scope_benchmark")
    from tools.embed_tmvec_index import _load_encoder, encode
    from utils.tm_calib_head import TMCalibHead

    model, converter, device, impl = _load_encoder(meta["config"], meta["checkpoint"])
    head = TMCalibHead.load(meta["head"], map_location=device).to(device).eval()
    max_length = args.max_length or None

    def encode_part(ids, seqs, batch):
        raw = encode(seqs, ids, model, converter, device, batch,
                     max_length=max_length, max_tokens=8192)
        with torch.inference_mode():
            return np.concatenate([head(torch.from_numpy(raw[s:s + 4096]).float().to(device)).float().cpu().numpy()
                                   for s in range(0, len(raw), 4096)])

    ids, seqs = selected["db"]["ids"], selected["db"]["seqs"]
    db = np.empty((len(ids), 1024), dtype=np.float32)
    for short in (True, False):
        indices = [i for i, seq in enumerate(seqs) if (len(seq) <= 1024) == short]
        for start in range(0, len(indices), 20000):
            take = indices[start:start + 20000]
            db[take] = encode_part([ids[i] for i in take], [seqs[i] for i in take], 8)
    query = np.concatenate([encode_part([identifier], [seq], 1)
                            for identifier, seq in zip(selected["query"]["ids"], selected["query"]["seqs"])])
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    db_normalized = db / np.linalg.norm(db, axis=1, keepdims=True)
    scores = (((q @ db_normalized.T).astype(np.float64) + 1.0) / 2.0).astype(np.float32)
    np.savez(args.work / "original.npz", db=db, query=query, scores=scores)
    return dict(implementation=impl, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def release(args, selected):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frozen_search import FrozenSearchIndex, FrozenT5Encoder

    meta = model_info(args)
    encoder = FrozenT5Encoder.load(config_path=meta["config"], ckpt_path=meta["checkpoint"],
                                   tm_head_path=meta["head"])
    index = FrozenSearchIndex.build(encoder, selected["db"]["seqs"], selected["db"]["ids"],
                              args.work / "index", max_length=args.max_length,
                              shard_size=20000, split_length=1024)
    captured = []

    class CapturingEncoder:
        identity = encoder.identity

        def encode(self, *values, **kwargs):
            result = encoder.encode(*values, **kwargs)
            captured.append(result)
            return result

    index.search(CapturingEncoder(), selected["query"]["seqs"], selected["query"]["ids"],
                 max_length=args.max_length, top_k=len(index.ids), query_batch_size=1,
                 query_block_size=None, score_mode="benchmark_tm", score_matrix=args.work / "scores.npy")
    np.savez(args.work / "release.npz", db=np.asarray(index.embeddings),
             query=np.concatenate(captured), scores=np.load(args.work / "scores.npy"))
    return dict(encoder=encoder.identity)


def compare(args, selected):
    old, new = np.load(args.work / "original.npz"), np.load(args.work / "release.npz")
    report = {}
    for key in ("db", "query", "scores"):
        report[key] = dict(exact=np.array_equal(old[key], new[key]),
                           max_abs_error=float(np.max(np.abs(old[key] - new[key]))))
    report["rankings_equal"] = bool(np.array_equal(np.argsort(-old["scores"], kind="stable", axis=1),
                                                    np.argsort(-new["scores"], kind="stable", axis=1)))
    # Historical full-database batch shapes differ from this small preflight.
    # Record the difference rather than treating it as proof of full parity.
    cache = args.source_work / "preview/splm" / args.reference_tag
    for key, filename in (("db", "db_emb.npy"), ("query", "query_scope.npy")):
        values = np.load(cache / filename, mmap_mode="r")[selected[key]["indices"]]
        report["historical_" + key] = dict(
            exact=np.array_equal(values, new[key]),
            max_abs_error=float(np.max(np.abs(values - new[key]))),
            min_cosine=float(np.min(np.sum(values * new[key], axis=1)
                                       / (np.linalg.norm(values, axis=1) * np.linalg.norm(new[key], axis=1)))))
    report["pass"] = all(report[k]["exact"] for k in ("db", "query", "scores")) and report["rankings_equal"]
    report["scope"] = "sampled old-vs-new inference only; full benchmark not yet verified"
    (args.work / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Old/release inference differs; inspect parity.json before full reproduction")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("prepare", "original", "release", "compare"), required=True)
    ap.add_argument("--source-work", type=Path, required=True)
    ap.add_argument("--reference-tag", required=True)
    ap.add_argument("--research-repo", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=0)
    ap.add_argument("--per-bin", type=int, default=4)
    args = ap.parse_args()
    if args.max_length < 0 or args.per_bin < 1:
        ap.error("Invalid length or sample size")
    args.work.mkdir(parents=True, exist_ok=True)
    path = args.work / "selection.json"
    started = time.perf_counter()
    if args.phase == "prepare":
        db_ids, db_seqs = read_fasta(args.source_work / "db/protein.fasta")
        q_ids, q_seqs = read_fasta(args.source_work / "queries/orig100/scope/protein.fasta")
        selected = dict(db=sample(db_ids, db_seqs, args.per_bin),
                        query=sample(q_ids, q_seqs, args.per_bin), max_length=args.max_length)
        path.write_text(json.dumps(selected) + "\n")
        result = {key: value["lengths"] for key, value in selected.items() if isinstance(value, dict)}
    else:
        selected = json.loads(path.read_text())
        if selected["max_length"] != args.max_length:
            raise ValueError("Work directory belongs to another truncation setting")
        result = {"original": original, "release": release, "compare": compare}[args.phase](args, selected)
    (args.work / (args.phase + ".json")).write_text(
        json.dumps(dict(seconds=time.perf_counter() - started, details=result), indent=2) + "\n")


if __name__ == "__main__":
    main()
