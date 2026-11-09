#!/usr/bin/env python
"""Export the frozen GO functional-consistency inputs and references; run in a compute allocation.

Reads the original FASTA, query/target lists, annotation views, GO snapshot,
the frozen published vectors and original search/score outputs from the read-only
research source. Nothing is downloaded and no source file is modified.
"""
import argparse
import csv
from pathlib import Path
import shutil
import os
import sys

import numpy as np

SOURCE_TAG=os.environ.get("SPLM_SOURCE_TAG","")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_go import evaluate, fasta, write_json  # noqa: E402

PROFILE = dict(max_length=512, batch_size=32, max_tokens=16384)


def read_hits(path):
    """Original search TSV as ordered hits per query (".pdb" suffixes dropped)."""
    hits = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            query, target = (key[:-4] if key.endswith(".pdb") else key for key in fields[:2])
            hits.setdefault(query, []).append(target)
    return hits


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=","))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root",type=Path,required=True,help="Read-only splm_gvpgit source directory")
    ap.add_argument("--out",type=Path,required=True,help="New, empty output directory owned by you")
    args=ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("Output must be new or empty; existing packs are never overwritten")
    source=args.source_root.resolve(); out=args.out.resolve()
    if out==source or source in out.parents:
        ap.error("Export outside the read-only source tree")
    out.mkdir(parents=True,exist_ok=True)
    origins={}
    def copy(src,name):
        src=Path(src)
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copyfile(src,out/name)
        origins[name]=dict(path=str(src))
    manifest=dict(case="go_functional",profile=dict(PROFILE),source_files=origins,
                  reference_vectors={},redistribution="not yet published; review source data licenses before hosting")
    sub=source/"remote_homology_benchmark/remote_go"
    if not SOURCE_TAG:
        sys.exit("请先 export SPLM_SOURCE_TAG=<源目录里冻结产物的 tag，如 emb_<tag>.npz 中的 <tag>>")
    frozen=sub/f"emb_{SOURCE_TAG}.npz"
    with np.load(frozen,allow_pickle=True) as values:
        ids=values["ids"].astype(str).tolist()
        tm=values["emb_h"].astype(np.float32)
        for key,value in (("raw_0",values["emb_raw"]),("tm_0",tm)):
            name=f"reference_{key}.npy"
            np.save(out/name,np.asarray(value,dtype=np.float32),allow_pickle=False)
            manifest["reference_vectors"][key]=name
    origins["frozen_embeddings"]=dict(path=str(frozen))
    copy(sub/"functional_all.fasta","all.fasta")
    for name in ("GO_experimental.json","GO_paper_exact_official1587.json","go-basic.obo"):
        copy(sub/name,name)
    for name in ("query.lst","target.lst"):
        copy(sub/"protein-search-benchmark/Functional_consistency"/name,name)
    if len((out/"query.lst").read_text().split())!=1000 or len((out/"target.lst").read_text().split())!=7735:
        raise ValueError("Unexpected GO benchmark size")
    if ids!=fasta(out/"all.fasta")[0]:
        raise ValueError("GO frozen vector and FASTA ordering differ")
    reference=evaluate(out,ids,tm)
    copy(sub/f"search_splm_{SOURCE_TAG}.tsv","original_search.tsv")
    if reference["hits"]!=read_hits(out/"original_search.tsv"):
        raise ValueError("GO frozen-vector rankings differ from original search output")
    for view in ("experimental","official"):
        if read_csv(sub/f"scores_{view}_{SOURCE_TAG}.csv")!=reference["views"][view]["pairs"]:
            raise ValueError(f"Frozen {view} GO CSV does not match original-vector scoring")
    write_json(out/"reference.json",reference)
    manifest["files"]=[str(p.relative_to(out)) for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out/"manifest.json",manifest)
    print(f"Exported go_functional -> {out}")


if __name__=="__main__":
    main()
