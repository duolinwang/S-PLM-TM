"""The installed `splmtm` command (`splm` is kept as an alias) and `python -m splm_tm` entry point."""
import argparse
import json
from pathlib import Path
import tempfile
import time


def _sequence_length(value):
    """Parse a residue limit, with readable full-length and legacy zero forms."""
    if value.lower() == "full":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a positive residue limit or 'full'") from exc
    if limit < 0:
        raise argparse.ArgumentTypeError("Use a positive residue limit or 'full'")
    return limit or None


def _model_arguments(parser, *, build=False):
    parser.add_argument("--backend", choices=("prot_t5","esm2_legacy"))
    parser.add_argument("--mode", choices=("raw","tm","ec"))
    parser.add_argument("--emb", choices=("emb_raw","emb_tm","emb_ec"), help="Representation alias for --mode")
    for name in ("config","checkpoint","tm-head","ec-head","legacy-repo"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--device")
    parser.add_argument("--head-batch-size",type=int,default=4096 if build else None)


def parser():
    ap=argparse.ArgumentParser(prog="splmtm",description="S-PLM-TM: embed sequences, build reusable indexes, search and align candidates.")
    commands=ap.add_subparsers(dest="command",required=True)
    from .embed_cli import __doc__ as embed_doc, add_arguments as add_embed_arguments
    add_embed_arguments(commands.add_parser("embed",help="Encode sequences into raw/tm/ec embeddings (.npz)",
                                            description=embed_doc,formatter_class=argparse.RawDescriptionHelpFormatter))
    index=commands.add_parser("index",help="Build or inspect a reusable target index").add_subparsers(dest="index_command",required=True)
    build=index.add_parser("build",help="Encode a target FASTA into a reusable index")
    build.add_argument("--fasta",type=Path,required=True)
    build.add_argument("--out","--index",dest="index",type=Path,required=True)
    _model_arguments(build,build=True)
    build.add_argument("--max-length",type=_sequence_length,default=None,
                       help="Residue limit, e.g. 512 or 1024; default: full sequences")
    build.add_argument("--batch-size",type=int,default=8)
    build.add_argument("--max-tokens",type=int,default=8192)
    build.add_argument("--shard-size",type=int,default=20000)
    build.add_argument("--split-length",type=int,default=1024)
    build.add_argument("--no-resume",action="store_true")
    build.add_argument("--metadata",type=Path,help="TSV beginning with an id column")
    build.add_argument("--pfam-json",type=Path,help="Target ID-to-accessions JSON")
    build.add_argument("--scan-pfam",action="store_true",help="Run optional PfamScan before building")
    build.add_argument("--pfam-data-root",type=Path)
    build.add_argument("--pfam-clan-tsv",type=Path)
    build.add_argument("--timing-json",type=Path)
    info=index.add_parser("info",help="Inspect an index without loading a model")
    info.add_argument("--index",type=Path,required=True)
    info.add_argument("--verify",action="store_true")
    search=commands.add_parser("search",help="Search a FASTA or provenance-tagged query vectors")
    query=search.add_mutually_exclusive_group(required=True)
    query.add_argument("--query","--fasta",dest="query",type=Path)
    query.add_argument("--query-vectors",type=Path)
    search.add_argument("--index",type=Path,required=True)
    search.add_argument("--out",type=Path,required=True)
    _model_arguments(search)
    search.add_argument("--max-length",type=_sequence_length,default=argparse.SUPPRESS,
                        help="Omit to inherit the index; explicit limits or 'full' must match it")
    search.add_argument("--topk",type=int,default=10)
    search.add_argument("--query-batch-size",type=int,default=1)
    search.add_argument("--query-block-size",type=int,default=32,help="0=all query rows")
    search.add_argument("--chunk-size",type=int,default=100000,help="Maximum target rows per score block")
    search.add_argument("--max-tokens",type=int,default=8192)
    search.add_argument("--metric",choices=("inner","cosine","euclidean"))
    search.add_argument("--score-mode",choices=("inner","cosine","negative_l2"))
    search.add_argument("--normalize",action="store_true",default=None)
    search.add_argument("--prefilter",choices=("full","pfamfam","pfamclan"),default="full")
    search.add_argument("--query-pfam-json",type=Path)
    search.add_argument("--pfam-data-root",type=Path)
    search.add_argument("--pfam-cache-dir",type=Path)
    # Retrieval is always exhaustive; --exact is accepted for older command lines.
    search.add_argument("--exact",action="store_true",default=None,help=argparse.SUPPRESS)
    search.add_argument("--exclude-self",action="store_true")
    search.add_argument("--save-matrix","--score-matrix",dest="score_matrix",type=Path)
    search.add_argument("--save-query-embeddings",type=Path,help="Write provenance-tagged .npz vectors")
    search.add_argument("--timing-json",type=Path)
    search.add_argument("--warmup",action="store_true")
    from .alignment_cli import add_arguments
    add_arguments(commands.add_parser("align", help="Align candidates from a saved search hit table"))
    return ap


def _encoder(args, saved=None):
    from .search_encoders import LegacyESMSearchEncoder,T5SearchEncoder
    backend=args.backend or (saved or {}).get("backend","prot_t5")
    mode=_mode(args, saved)
    if backend in ("esm2_legacy","legacy_esm2"):
        if mode != "raw" or not all((args.legacy_repo,args.config,args.checkpoint)):
            raise ValueError("ESM requires mode=raw, --legacy-repo, --config and --checkpoint")
        return LegacyESMSearchEncoder.load(args.legacy_repo,args.config,args.checkpoint,args.device)
    return T5SearchEncoder.load(mode=mode,config_path=args.config,ckpt_path=args.checkpoint,
                                tm_head_path=args.tm_head,ec_head_path=args.ec_head,
                                device=args.device,head_batch_size=args.head_batch_size if args.head_batch_size is not None
                                else (saved or {}).get("head_batch_size",4096))


def _mode(args, saved=None):
    alias = args.emb.removeprefix("emb_") if args.emb else None
    if alias is not None and args.mode is not None and alias != args.mode:
        raise ValueError("--emb and --mode disagree")
    mode = alias or args.mode or (saved or {}).get("mode", "tm")
    if saved is not None and mode != saved.get("mode"):
        raise ValueError("Requested representation does not match the index")
    return mode


def _scan(fasta, output, data_root):
    from .pfam import DEFAULT_PFAM_DATA_ROOT,run_pfam_scan
    run_pfam_scan(fasta,output,data_root or DEFAULT_PFAM_DATA_ROOT)
    return output


def main(argv=None):
    from .index import SearchIndex,QueryEmbeddings,_INHERIT,_save_json,file_sha256
    ap=parser()
    args=ap.parse_args(argv)
    if args.command == "embed":
        from .embed_cli import run
        return run(args)
    if args.command == "align":
        from .alignment_cli import run
        return run(args, ap)
    if args.command == "index" and args.index_command == "info":
        print(json.dumps(SearchIndex.load(args.index,verify=args.verify,check_model=False).info(),indent=2))
        return 0
    if args.command == "index":
        if args.split_length < 0:
            ap.error("--split-length must be nonnegative")
        encoder=_encoder(args)
        with tempfile.TemporaryDirectory() as temporary:
            pfam=args.pfam_json
            if args.scan_pfam and pfam is None:
                pfam=_scan(args.fasta,Path(temporary)/"pfam.json",args.pfam_data_root)
            index=SearchIndex.build(args.fasta,args.index,encoder=encoder,max_length=args.max_length,
                                    batch_size=args.batch_size,max_tokens=args.max_tokens,shard_size=args.shard_size,
                                    split_length=args.split_length or None,resume=not args.no_resume,
                                    metadata=args.metadata,pfam=pfam,clan_tsv=args.pfam_clan_tsv)
        timings=index.metadata["timings"]
        print(json.dumps(index.info(),indent=2))
    else:
        index=SearchIndex.load(args.index,check_model=False)
        _mode(args, index.meta["setup"]["encoder"])
        options=dict(topk=args.topk,normalize=args.normalize,metric=args.metric,
                     score_mode=args.score_mode,query_block_size=args.query_block_size or None,
                     chunk_size=args.chunk_size,prefilter=args.prefilter,exact=args.exact,
                     exclude_self=args.exclude_self,score_matrix=args.score_matrix,query_pfam=args.query_pfam_json)
        query_length = getattr(args, "max_length", _INHERIT)
        if query_length is not _INHERIT:
            index._query_length(query_length)
        if args.prefilter != "full" and args.query_pfam_json is None:
            if args.query is None:
                ap.error("Vector-query Pfam filtering requires --query-pfam-json")
            cache=args.pfam_cache_dir or args.out.parent/"pfam_cache"
            cache.mkdir(parents=True,exist_ok=True)
            target=cache/(file_sha256(args.query)+".json")
            options["query_pfam"]=target if target.is_file() else _scan(args.query,target,args.pfam_data_root)
        if args.query_vectors:
            result=index.search(QueryEmbeddings.load(args.query_vectors),**options)
        else:
            started=time.perf_counter()
            encoder=_encoder(args,index.metadata["setup"]["encoder"])
            index._check_encoder(encoder)
            setup_sec=time.perf_counter()-started
            result=index.search(args.query,encoder=encoder,max_length=query_length,
                                query_batch_size=args.query_batch_size,max_tokens=args.max_tokens,warmup=args.warmup,**options)
            result.timings["encoder_setup_sec"]=setup_sec
        result.to_tsv(args.out)
        if args.save_query_embeddings:
            if result.query_embeddings is None:
                ap.error("--save-query-embeddings requires sequence queries")
            QueryEmbeddings(result.query_ids,result.query_embeddings,index.metadata["setup"]["encoder"],
                            index.metadata["setup"]["max_length"]).save(args.save_query_embeddings)
        timings=result.timings
        print(f"Hits written to {args.out}")
    if args.timing_json:
        args.timing_json.parent.mkdir(parents=True,exist_ok=True)
        _save_json(args.timing_json,timings)
    return 0
