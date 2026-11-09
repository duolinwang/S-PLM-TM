"""SCOPe family/superfamily top-k and sequence-identity-binned AUC-1FP."""
import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
sys.path.insert(0, str(HERE))
from splm_tm import read_fasta  # noqa: E402
import multibin as mb  # noqa: E402
import retrieval_core as rc  # noqa: E402

CASE = "scope_retrieval"


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


def table(path, delimiter="\t"):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def overlap(data):
    from splm_tm.pfam import build_pfam_overlap_map
    # Match the original SCOPe get_clan_result, including exact accession keys.
    # The general release parser strips version suffixes; the historical one
    # does not. Reuse only the equivalent overlap-map construction here.
    clans = {}
    with (data / "Pfam-A.clans.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split("\t")
            clans[fields[0]] = fields[1] if fields[1] != "" else fields[0]
    def load(name):
        return {key: {clans.get(acc, acc) for acc in accessions}
                for key, accessions in read_json(data / name).items()}
    q = load("query_pfam.json")
    t = load("target_pfam.json")
    return build_pfam_overlap_map(q, t, list(t))


def scope_pfam_row(row, query, ids, index, candidates):
    """Original SCOPe-specific self retention and singleton-self fallback."""
    full = np.asarray(row, dtype=np.float64)
    keep = {index[t] for t in candidates.get(query, set()) if t in index}
    keep.add(index[query])
    if len(keep) == 1 and full[index[query]] >= .999:
        return full.copy()
    masked = np.full(len(ids), -np.inf, dtype=np.float64)
    columns = list(keep)
    masked[columns] = full[columns]
    return masked


def evaluate(data, ids, tm, device="cpu", prefilter="pfamclan"):
    import torch
    db = table(data / "scope_target.tsv")
    if [r["target"] for r in db] != ids:
        raise ValueError("SCOPe target order differs")
    ix = {key: i for i, key in enumerate(ids)}
    fa = np.asarray([r["sccs"] for r in db])
    sf = np.asarray([".".join(s.split(".")[:3]) for s in fa])
    fold = np.asarray([".".join(s.split(".")[:2]) for s in fa])
    ov = overlap(data) if prefilter == "pfamclan" else None
    # Original evaluator normalizes on CPU, then scores 32 queries at a time on GPU.
    z = torch.nn.functional.normalize(torch.from_numpy(tm).float(), dim=1).to(device)
    tasks, query_lists, row_lists, labels = {}, [], [], []
    for task, filename, dlabs, soft in (("family", "scope_fam_query.tsv", fa, sf),
                                       ("superfamily", "scope_supfam_query.tsv", sf, fold)):
        qrows = table(data / filename)
        queries = [r["query"] for r in qrows]
        labs = np.asarray([r["sccs"] if task == "family" else ".".join(r["sccs"].split(".")[:3])
                           for r in qrows])
        if any(q not in ix or dlabs[ix[q]] != lab for q, lab in zip(queries, labs)):
            raise ValueError("Query IDs/labels do not match SCOPe targets")
        rows = []
        for start in range(0, len(queries), 32):
            scores = (z[[ix[q] for q in queries[start:start + 32]]] @ z.t()).cpu().numpy()
            rows.extend(scope_pfam_row(row, q, ids, ix, ov) if ov is not None else row.copy()
                        for q, row in zip(queries[start:start + 32], scores))
        perq = []
        for q, lab, row in zip(queries, labs, rows):
            metrics = {f"Top{k}": rc.per_query_sensitivity(row, q, lab, ids, dlabs, ix, k)
                       for k in (1, 5, 10, 20, 50, 100, 200)}
            metrics["AUC-1FP"] = rc.per_query_auc_1fp(row, q, lab, ids, dlabs, ix, soft)
            perq.append(dict(query=q, **metrics))
        tasks[task] = dict(summary={key: float(np.nanmean([r[key] for r in perq]))
                                    for key in perq[0] if key != "query"}, per_query=perq)
        query_lists.append(queries); row_lists.append(rows); labels.append(labs)
    edges = mb.parse_edges(mb.DEFAULT_SEQ_ID_BINS)
    bins = mb.load_seq_identity_bins_by_query(data / "identity.tsv", sum(query_lists, []), ix, edges, "percent")
    # A missing identity pair changes denominators. Do not silently omit it.
    expected = mb.expected_labeled_pairs(queries=query_lists[0], q_labels=labels[0], db_domains=ids, db_labels=fa)
    expected += mb.expected_labeled_pairs(queries=query_lists[1], q_labels=labels[1], db_domains=ids, db_labels=sf)
    missing = [(q, t) for q, t in expected if t not in bins.get(q, {})]
    if missing:
        raise ValueError(f"Missing {len(missing)} positive sequence-identity pairs")
    rows = mb.collect_multibin_rows(family_rows=row_lists[0], family_queries=query_lists[0],
        q_family=labels[0], superfamily_rows=row_lists[1], superfamily_queries=query_lists[1],
        q_superfamily=labels[1], db_domains=ids, db_family=fa, db_superfamily=sf, db_fold=fold,
        dom_index=ix, seq_bins=bins, seq_labels=[mb.bin_label(edges, i) for i in range(len(edges)-1)])
    binned = {}
    for row in rows:
        vals = np.asarray(row["per_query_aucs"], dtype=float)
        if len(vals):
            binned[f"{row['task']}|{row['group_name']}"] = dict(
                query_ids=row["per_query_ids"], aucs=vals.tolist(),
                summary=[len(vals), float(vals.mean()), float((vals < .05).mean())])
    return dict(tasks=tasks, binned=binned)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Data pack exported by prepare_scope.py")
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
    arrays, provenance = encode(args, manifest, ["target.fasta"])
    metrics = evaluate(args.data, arrays["ids_0"].tolist(), arrays["tm_0"], args.device,
                       manifest["prefilter"])
    finish(args, metrics, provenance)


if __name__ == "__main__":
    main()
