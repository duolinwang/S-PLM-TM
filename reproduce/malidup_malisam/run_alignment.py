"""Final T5 alignment benchmark: paired GPU encoding, CLAlign global gap=0."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))

DATASETS = ("malidup", "malisam")
REFERENCE = {dataset: HERE / "reference" / f"{dataset}_align_head.json" for dataset in DATASETS}
PROFILE = dict(max_length=None, batch_size=2, max_tokens=8192,
               scoring="paired GPU residue matmul; global gap=0")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def runtime(device):
    import torch
    info = dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                cuda=torch.version.cuda, device=str(device), cpu_threads=torch.get_num_threads(),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if torch.device(device).type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


TOL_MEAN_DECIMALS = 3   # the means are what the paper reports, to three decimals
TOL_PAIR_ABS = 5e-4     # per-pair values: at most half a unit in the third decimal


def differences(actual, reference, path=""):
    """Exact reference-field comparison, including ordered per-pair arrays."""
    out = []
    if isinstance(reference, dict):
        if not isinstance(actual, dict):
            return [path + ": wrong type"]
        for key, expected in reference.items():
            if key not in actual:
                out.append(path + "/" + key + ": missing")
            else:
                out.extend(differences(actual[key], expected, path + "/" + key))
    elif isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            return [path + ": length/type differs"]
        for i, (value, expected) in enumerate(zip(actual, reference)):
            out.extend(differences(value, expected, path + f"/{i}"))
    elif actual != reference:
        out.append(f"{path}: {actual!r} != {reference!r}")
    return out


def tolerant_differences(actual, reference, path="", deviations=None):
    """Same structure check, but floats may differ by last-digit noise.

    Structure, ordering and pair identifiers must match exactly. Each mean must
    round to the same three decimals as the reference, because those are the
    reported values; per-pair values may differ by at most half a unit in the
    third decimal. A different GPU, BLAS build or TF32 setting perturbs the
    cosine matrix in its last digits, which can flip a tie inside the global
    alignment and move one pair's F1, so an exact comparison would report a
    failure where none exists. Returns (messages, max_abs_deviation).
    """
    collected = [] if deviations is None else deviations
    out = []
    if isinstance(reference, dict):
        if not isinstance(actual, dict):
            out.append(path + ": wrong type")
        else:
            for key, expected in reference.items():
                if key not in actual:
                    out.append(path + "/" + key + ": missing")
                else:
                    out.extend(tolerant_differences(actual[key], expected,
                                                    path + "/" + key, collected)[0])
    elif isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            out.append(path + ": length/type differs")
        else:
            for i, (value, expected) in enumerate(zip(actual, reference)):
                out.extend(tolerant_differences(value, expected, path + f"/{i}", collected)[0])
    elif isinstance(reference, float) and isinstance(actual, (int, float))             and not isinstance(actual, bool):
        delta = abs(float(actual) - reference)
        collected.append(delta)
        if "/mean/" in path:
            if round(float(actual), TOL_MEAN_DECIMALS) != round(reference, TOL_MEAN_DECIMALS):
                out.append(f"{path}: {actual!r} != {reference!r} "
                           f"(differs at {TOL_MEAN_DECIMALS} decimals)")
        elif delta > TOL_PAIR_ABS:
            out.append(f"{path}: {actual!r} != {reference!r} "
                       f"(delta {delta:.3e} > {TOL_PAIR_ABS:.0e})")
    elif actual != reference:
        out.append(f"{path}: {actual!r} != {reference!r}")
    return out, (max(collected) if collected else 0.0)


def load_pairs(path):
    # Preserve the upstream HEC CSV layout; the header line is skipped.
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            fields=line.rstrip("\n").split(",")
            if len(fields) != 9:
                raise ValueError("Expected nine HEC CSV columns")
            pair,a,b,alna,alnb=fields[0],fields[3].upper(),fields[4].upper(),fields[7].upper(),fields[8].upper()
            if alna.replace("-","") != a or alnb.replace("-","") != b or len(alna)!=len(alnb):
                raise ValueError(f"Invalid reference alignment: {pair}")
            rows.append((pair,a,b,alna,alnb))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Duplicate alignment pair IDs")
    return rows


def reference():
    out = {}
    for dataset in DATASETS:
        value = read_json(REFERENCE[dataset])
        out[dataset] = {key: value[key] for key in ("mean", "pairs")}
    return out


def finish(out, metrics, provenance, skip_reference=False):
    write_json(out / "metrics.json", metrics)
    write_json(out / "run.json", provenance)
    if skip_reference:
        report = dict(status="unvalidated", reason="reference comparison explicitly skipped")
    else:
        expected = reference()
        delta, max_deviation = tolerant_differences(metrics, expected)
        bit_exact = not differences(metrics, expected)
        report = dict(status="different" if delta else
                      ("exact_metric_match" if bit_exact else "match_within_tolerance"),
                      differences=delta, bit_exact=bit_exact,
                      max_abs_deviation=max_deviation,
                      tolerance=dict(mean_decimals=TOL_MEAN_DECIMALS,
                                     per_pair_abs=TOL_PAIR_ABS),
                      inference="newly encoded sequences")
    write_json(out / "comparison.json", report)
    for dataset in sorted(metrics):
        mean = metrics[dataset]["mean"]
        print(f"  {dataset:8s} F1 {mean['f1']:.3f}   "
              f"(precision {mean['precision']:.3f}  recall {mean['recall']:.3f}  "
              f"{len(metrics[dataset]['pairs'])} pairs)")
    if not skip_reference:
        print(f"  max absolute deviation from reference: {report['max_abs_deviation']:.3e}"
              f"   bit-exact: {report['bit_exact']}")
    if report["status"] == "different":
        raise SystemExit("Reference check did not pass; see comparison.json")
    print(json.dumps(report, indent=2))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,default=HERE/"data",help="Directory with malidup_hec.csv / malisam_hec.csv")
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--config",type=Path)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--align-head",type=Path)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--skip-reference",action="store_true",help="Development only; record unvalidated status")
    args=ap.parse_args(argv)
    # Import the alignment kernel before allocating model memory. An installed
    # CLAlign wins, so a machine that has it keeps the original code path; the
    # vendored subset in splm_tm._clalign_core runs the same kernel elsewhere.
    try:
        from clalign.alignment import ProteinSeq,AlignmentResult,align_core
        from clalign.metrics import f1score
        kernel="clalign"
    except ImportError:
        from splm_tm._clalign_core import ProteinSeq,AlignmentResult,align_core,f1score
        kernel="splm_tm._clalign_core"
    print(f">>> alignment kernel: {kernel}")
    import torch
    import torch.nn.functional as F
    from splm_tm import Backbone,load_align_head
    from splm_tm.paths import default_align_head
    backbone=Backbone.load(config_path=args.config,ckpt_path=args.checkpoint,device=args.device)
    path=args.align_head or default_align_head()
    head,meta=load_align_head(path,device=args.device)
    outputs={}
    started=time.perf_counter()
    with torch.no_grad():
        for dataset in DATASETS:
            rows=[]
            for key,a,b,alna,alnb in load_pairs(args.data/f"{dataset}_hec.csv"):
                # Match the historical two-sequence padded batch and GPU matmul.
                _,_,tokens=backbone.batch_converter([("a",a),("b",b)])
                _,_,_,res=backbone.model(batch_tokens=tokens.to(backbone.device),mode="sequence",return_embedding=True)
                res=res.float()
                if res.shape[0]!=len(a)+len(b):
                    raise ValueError("Residue feature count changed")
                za=F.normalize(head(res[:len(a)]),dim=-1)
                zb=F.normalize(head(res[len(a):]),dim=-1)
                matrix=(za@zb.T).cpu().numpy().astype(np.float32)
                oa,ob=ProteinSeq(a),ProteinSeq(b)
                predicted=align_core(oa,ob,matrix,0.0)
                truth=AlignmentResult(oa,ob,alna,alnb)
                precision,recall,f1=f1score(truth,predicted)
                rows.append(dict(pair=key,precision=float(precision),recall=float(recall),f1=float(f1)))
            outputs[dataset]=dict(mean={key:float(np.mean([row[key] for row in rows]))
                                       for key in ("precision","recall","f1")},pairs=rows)
    provenance=dict(profile=PROFILE,alignment_kernel=kernel,runtime=runtime(backbone.device),
                    model_files=dict(config=str(backbone.config_path),checkpoint=str(backbone.ckpt_path),
                                     align_head=str(path)),
                    alignment_wall_sec=time.perf_counter()-started)
    finish(args.out,outputs,provenance,args.skip_reference)


if __name__=="__main__":
    main()
