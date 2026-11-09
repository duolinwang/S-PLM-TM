#!/usr/bin/env python
"""Build a SCOPe data pack from the published inputs on Hugging Face.

`prepare_scope.py` exports a pack from the read-only research tree and includes
the frozen reference vectors and the published metrics, so `run_scope.py` can
check a rerun against them exactly. Those references are internal and are not
part of the public release.

This script builds a pack from what *is* public — the query and target tables,
the Pfam clan annotations and the sequence-identity table in
`deepduoduo/S-PLM-TM-benchmarks` under `scope/` — plus the Pfam clan map shipped
in this repository. The resulting pack runs the full retrieval and scoring
pipeline and writes the same metrics; it carries no reference to compare them
against, so `run_scope.py` is run with `--skip-reference`.

Only JSON, TSV and FASTA are read. Nothing is downloaded and no input is
modified.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scope import read_json, table, write_json  # noqa: E402
from splm_tm import write_fasta  # noqa: E402

CASE = "scope_retrieval"
# Sizes of the published benchmark; a pack that differs is not this benchmark.
N_TARGET, N_FAMILY, N_SUPERFAMILY = 29421, 5082, 2065
# Downloaded from deepduoduo/S-PLM-TM-benchmarks, directory scope/.
FROM_HF = ("scope_target.tsv", "scope_fam_query.tsv", "scope_supfam_query.tsv",
           "query_pfam.json", "target_pfam.json", "identity.tsv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True, type=Path,
                    help="Directory holding the downloaded scope/ files")
    ap.add_argument("--out", required=True, type=Path,
                    help="New, empty output directory owned by you")
    ap.add_argument("--clans-tsv", type=Path,
                    help="Pfam-A.clans.tsv (default: pfam_data/ in this repository)")
    ap.add_argument("--max-length", type=int, required=True,
                    help="0 encodes full length; a positive value truncates. The profile of "
                         "the frozen cache behind the published numbers is not part of the "
                         "public data, so state the one you intend to use.")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    args = ap.parse_args(argv)
    if args.max_length < 0 or min(args.batch_size, args.max_tokens) < 1:
        ap.error("Invalid inference profile")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("Output must be new or empty; existing packs are never overwritten")

    hf = args.hf_dir.resolve()
    out = args.out.resolve()
    if out == hf or hf in out.parents:
        ap.error("Write the pack outside the downloaded directory")
    clans = (args.clans_tsv or
             Path(__file__).resolve().parents[2] / "pfam_data" / "Pfam-A.clans.tsv").resolve()
    out.mkdir(parents=True, exist_ok=True)

    origins = {}

    def copy(src, name):
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(f"{src}\nDownload scope/ from "
                                    "https://huggingface.co/datasets/deepduoduo/S-PLM-TM-benchmarks")
        shutil.copyfile(src, out / name)
        origins[name] = dict(path=str(src))

    for name in FROM_HF:
        copy(hf / name, name)
    copy(clans, "Pfam-A.clans.tsv")

    records = table(out / "scope_target.tsv")
    ids = [r["target"] for r in records]
    if (len(ids) != N_TARGET
            or len(table(out / "scope_fam_query.tsv")) != N_FAMILY
            or len(table(out / "scope_supfam_query.tsv")) != N_SUPERFAMILY):
        raise ValueError("Unexpected SCOPe benchmark size")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate target identifiers")
    # The published Pfam-clan prefilter needs an annotation for every identifier.
    missing = set(ids) - set(read_json(out / "target_pfam.json"))
    if missing:
        raise ValueError(f"{len(missing)} targets have no Pfam annotation")
    queries = {r["query"] for name in ("scope_fam_query.tsv", "scope_supfam_query.tsv")
               for r in table(out / name)}
    missing = queries - set(read_json(out / "query_pfam.json"))
    if missing:
        raise ValueError(f"{len(missing)} queries have no Pfam annotation")

    write_fasta(out / "target.fasta", ids, [r["sequence"] for r in records])
    manifest = dict(
        case=CASE,
        profile=dict(max_length=args.max_length or None, batch_size=args.batch_size,
                     max_tokens=args.max_tokens),
        source_files=origins,
        prefilter="pfamclan",
        # No reference_vectors and no reference.json: see the module docstring.
        reference_vectors={},
        source="deepduoduo/S-PLM-TM-benchmarks, scope/ (CC BY 4.0)",
        references="none; this pack carries published inputs only, so run with --skip-reference",
        redistribution="inputs are published under CC BY 4.0")
    manifest["files"] = [str(p.relative_to(out)) for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out / "manifest.json", manifest)
    print(f"Exported {CASE} (public inputs) -> {out}")
    print(json.dumps(manifest["profile"]))
    print("Next:\n  python reproduce/scope_retrieval/run_scope.py "
          f"--data {out} --out /your/results/scope --skip-reference")


if __name__ == "__main__":
    main()
