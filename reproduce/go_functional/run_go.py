"""GO-MF Wang BMA: final-model original top-10 and annotation views."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from splm_tm import read_fasta  # noqa: E402

CASE = "go_functional"


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


def fasta(path):
    ids, seqs = read_fasta(path)
    if not ids or len(ids) != len(seqs):
        raise ValueError("ids and sequences must be nonempty and have equal length")
    if any(not isinstance(x, str) or not x or any(c.isspace() for c in x) for x in ids):
        raise ValueError("IDs must be nonempty strings without whitespace")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sequence IDs are not supported")
    if any(not isinstance(x, str) or not x or any(c.isspace() for c in x) for x in seqs):
        raise ValueError("Sequences must be nonempty strings without whitespace")
    return list(ids), list(seqs)


def verify_pack(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["case"] != CASE:
        raise ValueError("Data pack belongs to another benchmark")
    for name in manifest["files"]:
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError("Invalid data pack path")
        if not path.is_file():
            raise ValueError(f"Data pack file missing: {name}")
    return manifest


def load_tm_model(args):
    """Backbone and paired TM head; also returns the model file paths used."""
    from splm_tm import SPLMTM, paths

    files = dict(config=Path(args.config) if args.config else paths.default_config(),
                 checkpoint=Path(args.checkpoint) if args.checkpoint else paths.default_ckpt(),
                 tm_head=Path(args.tm_head) if args.tm_head else paths.default_tm_head())
    model = SPLMTM.load(modes=("tm",), config_path=files["config"], ckpt_path=files["checkpoint"],
                        tm_head_path=files["tm_head"], device=args.device)
    return model, {name: str(path) for name, path in files.items()}


def synchronize(device):
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def encode(args, manifest, fasta_names):
    """Encode each original FASTA together, then apply TM head to all its rows.

    This intentionally preserves historical batching/head calls rather than
    the large-scale index's shards or 4096-row head chunks.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    profile = manifest["profile"]
    input_ids = {name: fasta(args.data / name)[0] for name in fasta_names}
    if args.embeddings:
        with np.load(args.embeddings, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        provenance = read_json(Path(str(args.embeddings) + ".json"))
        if provenance["profile"] != profile or provenance.get("case") != manifest["case"]:
            raise ValueError("Cached embeddings belong to another benchmark or encoding profile")
        for i, name in enumerate(fasta_names):
            if arrays[f"ids_{i}"].tolist() != input_ids[name]:
                raise ValueError("Cached embeddings have different ID ordering")
        return arrays, provenance
    import torch
    model, model_files = load_tm_model(args)
    synchronize(model.device)
    started = time.perf_counter()
    arrays = {}
    for i, name in enumerate(fasta_names):
        ids, seqs = fasta(args.data / name)
        raw = model.backbone.encode(
            seqs, ids, batch_size=profile["batch_size"], max_tokens=profile["max_tokens"],
            max_length=profile["max_length"], verbose=True)
        with torch.no_grad():
            projected = model.tm_head(torch.from_numpy(raw).float().to(model.head_device))
            tm = projected.float().cpu().numpy()
        arrays.update({f"ids_{i}": np.asarray(ids, dtype=str), f"raw_{i}": raw, f"tm_{i}": tm})
    synchronize(model.device)
    provenance = dict(case=manifest["case"], profile=profile, model_files=model_files,
                      runtime=runtime(model.device),
                      reference_vectors=manifest.get("reference_vectors", {}),
                      head_inference="one call for all rows of each FASTA",
                      encoding_wall_sec=time.perf_counter() - started)
    np.savez(args.out / "embeddings.npz", **arrays)
    write_json(args.out / "embeddings.npz.json", provenance)
    return arrays, provenance


def differences(actual, reference, path=""):
    """Exact reference-field comparison, including ordered per-query arrays."""
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


def finish(args, metrics, provenance):
    write_json(args.out / "metrics.json", metrics)
    write_json(args.out / "run.json", provenance)
    reference = args.data / "reference.json"
    if args.skip_reference:
        report = dict(status="unvalidated", reason="reference comparison explicitly skipped")
    elif not reference.is_file():
        report = dict(status="unvalidated", reason="reference.json missing")
    else:
        delta = differences(metrics, read_json(reference))
        # Frozen raw/head vectors provide a stricter gate than rounded metrics.
        if not args.embeddings and provenance.get("reference_vectors"):
            with np.load(args.out / "embeddings.npz", allow_pickle=False) as current:
                for key, name in provenance["reference_vectors"].items():
                    expected = np.load(args.data / name, mmap_mode="r", allow_pickle=False)
                    if not np.array_equal(current[key], expected):
                        delta.append(key + ": differs from frozen vectors")
        report = dict(status="different" if delta else "exact_metric_match", differences=delta,
                      inference="cached vectors" if args.embeddings else "newly encoded sequences")
    write_json(args.out / "comparison.json", report)
    if report["status"] != "exact_metric_match" and not args.skip_reference:
        raise SystemExit("Reference check did not pass; see comparison.json")
    print(json.dumps(report, indent=2))


def retrieve(ids, embeddings, query_ids, target_ids, topk=10):
    positions = {key: i for i,key in enumerate(ids)}
    q = embeddings[[positions[key] for key in query_ids]].astype(np.float32)
    t = embeddings[[positions[key] for key in target_ids]].astype(np.float32)
    sim = q @ t.T
    if topk > len(target_ids):
        raise ValueError("GO benchmark requires at least ten targets")
    top = np.argpartition(-sim, topk-1, axis=1)[:, :topk]
    return {q: [target_ids[j] for j in top[i][np.argsort(-sim[i,top[i]])]]
            for i,q in enumerate(query_ids)}


def score_view(hits, annotations, godag, pairsim):
    # Dense SPLM retrieval always returns ten hits. This avoids the legacy
    # short-hit random fallback, whose set ordering depends on PYTHONHASHSEED.
    usable = {q: [term for term in terms if term in godag] for q,terms in annotations.items()}
    usable = {q:terms for q,terms in usable.items() if terms}
    rows, perq = [], defaultdict(list)
    for q, targets in hits.items():
        if q not in usable:
            continue
        for t in targets:
            if t not in usable:
                continue
            a,b = usable[q], usable[t]
            qt = np.mean([max(pairsim(x,y) for y in b) for x in a])
            tq = np.mean([max(pairsim(y,x) for x in a) for y in b])
            value = float((qt+tq)/2)
            # Frozen per-pair CSVs contain six decimals, preserving this representation.
            rows.append(dict(query=q,target=t,go_similarity=f"{value:.6f}"))
            perq[q].append(value)
    qm = np.asarray([np.mean(values) for values in perq.values()])
    return dict(pairs=rows, query_count=len(qm),
                per_query_means={q:float(np.mean(v)) for q,v in perq.items()},
                mean=float(qm.mean()) if len(qm) else None,
                median=float(np.median(qm)) if len(qm) else None)


def evaluate(data, ids, embeddings):
    from goatools.base import get_godag
    from goatools.semsim.termwise.wang import SsWang
    queries = (data/"query.lst").read_text().split()
    targets = (data/"target.lst").read_text().split()
    if len(set(queries)) != len(queries) or len(set(targets)) != len(targets) or set(queries)&set(targets):
        raise ValueError("GO query and target IDs must be unique and disjoint")
    hits = retrieve(ids, embeddings, queries, targets)
    godag = get_godag(str(data/"go-basic.obo"), optional_attrs={"relationship"})
    cache = {}
    def pairsim(a,b):
        key = (a,b) if a <= b else (b,a)
        if key not in cache:
            value = SsWang([a,b],godag,{"part_of"}).get_sim(a,b)
            cache[key] = 0.0 if value is None else value
        return cache[key]
    return dict(hits=hits, views={view:score_view(hits,read_json(data/name),godag,pairsim)
                                 for view,name in (("experimental","GO_experimental.json"),
                                                   ("official","GO_paper_exact_official1587.json"))})


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Data pack exported by prepare_go.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tm-head", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embeddings", type=Path,
                    help="Evaluate existing release embeddings.npz; does not certify new inference")
    ap.add_argument("--skip-reference", action="store_true", help="Development only; record unvalidated status")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = verify_pack(args.data)
    arrays, provenance = encode(args, manifest, ["all.fasta"])
    finish(args, evaluate(args.data, arrays["ids_0"].tolist(), arrays["tm_0"]), provenance)


if __name__ == "__main__":
    main()
