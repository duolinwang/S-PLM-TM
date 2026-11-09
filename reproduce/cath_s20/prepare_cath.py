#!/usr/bin/env python
"""Export the frozen CATH-S20 inputs and references; run in a compute allocation.

Reads the original FASTA, official label list, the frozen published vectors and
summary files from the read-only research source. Nothing is downloaded and no
source file is modified.
"""
import argparse
from pathlib import Path
import re
import shutil
import os
import sys

import numpy as np

SOURCE_TAG=os.environ.get("SPLM_SOURCE_TAG","")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cath import evaluate, fasta, read_labels, write_json  # noqa: E402

PROFILE = dict(max_length=512, batch_size=64, max_tokens=16384)


def check_cath_summaries(result, f1_path, auc_path):
    """Check the precision actually available in the original summary files.

    These four-decimal checks supplement full-precision frozen-vector references;
    they do not certify fresh inference or exact historical rankings.
    """
    f1_text = Path(f1_path).read_text(encoding="utf-8")
    auc_text = Path(auc_path).read_text(encoding="utf-8")
    for level, metrics in result["summary"].items():
        f1 = re.search(re.escape(level) + r"\s+mean F1-max = ([0-9.]+)\s+\(n=(\d+)\)", f1_text)
        auc = re.search(re.escape(level) + r"\s+AUC-sens1FP = ([0-9.]+)", auc_text)
        if not f1 or not auc:
            raise ValueError(f"Missing original CATH summary for {level}")
        if (f"{metrics['f1max']:.4f}" != f1[1]
                or metrics["queries_with_positives"] != int(f1[2])
                or f"{metrics['auc_sensitivity_1fp']:.4f}" != auc[1]):
            raise ValueError(f"CATH frozen-vector metrics differ from original summaries: {level}")


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
    manifest=dict(case="cath_s20",profile=dict(PROFILE),source_files=origins,
                  reference_vectors={},redistribution="not yet published; review source data licenses before hosting")
    sub=source/"remote_homology_benchmark/CATH"
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
    copy(sub/"cath_S20.fasta","target.fasta")
    copy(source/"remote_homology_benchmark/remote_go/protein-search-benchmark/Fold_classification/cath-domain-list-S20-v4_4_0.txt","labels.txt")
    if ids!=fasta(out/"target.fasta")[0] or len(ids)!=15043:
        raise ValueError("CATH frozen vector and FASTA ordering differ")
    reference=evaluate(ids,tm,read_labels(out/"labels.txt"))
    copy(sub/f"f1max_dense_{SOURCE_TAG}_emb_h.txt","original_f1max.txt")
    copy(sub/f"summary_{SOURCE_TAG}_head.txt","original_sensitivity.txt")
    check_cath_summaries(reference,out/"original_f1max.txt",out/"original_sensitivity.txt")
    write_json(out/"reference.json",reference)
    manifest["files"]=[str(p.relative_to(out)) for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out/"manifest.json",manifest)
    print(f"Exported cath_s20 -> {out}")


if __name__=="__main__":
    main()
