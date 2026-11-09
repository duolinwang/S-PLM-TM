#!/usr/bin/env python
"""Export the frozen SCOPe retrieval inputs and references; run in a compute allocation.

Pickle is read only during this internal export from explicitly trusted sources;
the runner reads JSON / FASTA / TSV / numeric NumPy files only. Nothing is
downloaded and no source file is modified.
"""
import argparse
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scope import read_json, table, write_json  # noqa: E402
from splm_tm import write_fasta  # noqa: E402

MODEL=os.environ.get("SPLM_MODEL_DIR","")
CKPT=os.environ.get("SPLM_CKPT_TAG","")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-repo",type=Path,required=True,help="Checkout of the original benchmark code")
    ap.add_argument("--source-root",type=Path,required=True,help="Read-only splm_gvpgit source directory")
    ap.add_argument("--out",type=Path,required=True,help="New, empty output directory owned by you")
    ap.add_argument("--max-length",type=int,required=True,help="Profile of the frozen raw cache; 0=full, positive=truncate")
    ap.add_argument("--batch-size",type=int,required=True)
    ap.add_argument("--max-tokens",type=int,required=True)
    ap.add_argument("--query-pfam",type=Path,required=True)
    ap.add_argument("--target-pfam",type=Path,required=True)
    ap.add_argument("--identity-tsv",type=Path,required=True)
    ap.add_argument("--frozen-raw-pickle",type=Path,required=True,help="SCOPe original db_embeddings.pkl")
    args=ap.parse_args(argv)
    if args.max_length<0 or min(args.batch_size,args.max_tokens)<1:
        ap.error("Invalid inference profile")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("Output must be new or empty; existing packs are never overwritten")
    source=args.source_root.resolve(); research=args.research_repo.resolve()
    out=args.out.resolve()
    if out==source or source in out.parents or out==research or research in out.parents:
        ap.error("Export outside the read-only source/research trees")
    out.mkdir(parents=True,exist_ok=True)
    origins={}
    def copy(src,name):
        src=Path(src)
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copyfile(src,out/name)
        origins[name]=dict(path=str(src))
    profile=dict(max_length=args.max_length or None,batch_size=args.batch_size,
                 max_tokens=args.max_tokens)
    manifest=dict(case="scope_retrieval",profile=profile,source_files=origins,
                  reference_vectors={},redistribution="not yet published; review source data licenses before hosting")
    if not MODEL or not CKPT:
        sys.exit("请先 export SPLM_MODEL_DIR=<results/ 下的 model 目录名> 与 "
                 "SPLM_CKPT_TAG=<checkpoints/ 下的 checkpoint 文件名，不含后缀>")
    tag=f"{MODEL}_{CKPT}"
    references=source/"tm_calib/tmcalib_results"
    copy(args.query_pfam,"query_pfam.json");copy(args.target_pfam,"target_pfam.json")
    copy(research/"plmsearch_data/Pfam_db/Pfam-A.clans.tsv","Pfam-A.clans.tsv")
    manifest["prefilter"]="pfamclan"
    for name in ("scope_target.tsv","scope_fam_query.tsv","scope_supfam_query.tsv"):
        copy(research/"scope_benchmark"/name,name)
    copy(args.identity_tsv,"identity.tsv")
    records=table(out/"scope_target.tsv")
    ids=[r["target"] for r in records]
    if len(ids)!=29421 or len(table(out/"scope_fam_query.tsv"))!=5082 or len(table(out/"scope_supfam_query.tsv"))!=2065:
        raise ValueError("Unexpected SCOPe benchmark size")
    write_fasta(out/"target.fasta",ids,[r["sequence"] for r in records])
    with args.frozen_raw_pickle.open("rb") as handle:
        values=pickle.load(handle)
    # Original SCOPe stacks in target TSV order.
    if set(values)!=set(ids):
        raise ValueError("Frozen pickle ID set differs from exported inputs")
    np.save(out/"reference_raw_0.npy",np.asarray(np.vstack([np.asarray(values[q]).reshape(1,-1) for q in ids]),
                                                 dtype=np.float32),allow_pickle=False)
    manifest["reference_vectors"]["raw_0"]="reference_raw_0.npy"
    origins["trusted_original_raw_0"]=dict(path=str(args.frozen_raw_pickle))
    overall=read_json(references/f"tmcalib_{tag}_topk.json")["head_pfam"]
    binned=read_json(references/f"tmcalib_{tag}.json")["head_pfam"]
    # Convert trusted object-ID arrays to ordinary JSON before public use.
    p=references/f"tmcalib_{tag}_perquery.npz"
    with np.load(p,allow_pickle=True) as perq:
        bins={key:dict(summary=value,query_ids=perq[f"ids|{key}"].astype(str).tolist(),
                       aucs=perq[f"aucs|head_pfam|{key}"].tolist()) for key,value in binned.items()}
    reference=dict(tasks={task:dict(summary=value) for task,value in overall.items()},binned=bins)
    write_json(out/"reference.json",reference)
    manifest["files"]=[str(p.relative_to(out)) for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out/"manifest.json",manifest)
    print(f"Exported scope_retrieval -> {out}")


if __name__=="__main__":
    main()
