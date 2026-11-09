"""Candidate alignment CLI shared by `splmtm align` and the legacy script."""
from pathlib import Path


def add_arguments(parser):
    from .cli import _sequence_length
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--index", type=Path, help="Library containing original target sequences")
    targets.add_argument("--targets", type=Path, help="Target FASTA, streamed for selected IDs only")
    parser.add_argument("--out", type=Path, required=True, help="Alignment JSONL and metadata sidecar")
    parser.add_argument("--score-mode", choices=("inner", "cosine", "negative_l2"),
                        help="Required only for legacy tables without a score-mode sidecar")
    for name in ("config", "checkpoint", "align-head"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-length", type=_sequence_length, default=None,
                        help="Residue limit; omit or use 'full' to align complete sequences")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--cache-residues", type=int, default=32768)


def read_candidates(path, score_mode=None):
    from .index import SearchResult
    return SearchResult.from_tsv(path, score_mode=score_mode)


def fasta_mapping(path):
    """Compatibility helper; candidate alignment itself streams FASTA."""
    from .fasta import read_fasta
    from .index import _records
    ids, sequences = read_fasta(path)
    _records(ids, sequences)
    return dict(zip(ids, sequences))


def run(args, parser):
    from .alignment import CandidateAligner, CandidateAlignmentResult, _clalign_engine
    from .index import Index
    if args.topk < 1 or (args.max_length is not None and args.max_length < 0) or args.max_tokens < 1 or args.cache_residues < 0:
        parser.error("Invalid length, topk, token budget or cache limit")
    inputs = [args.hits, args.queries, args.targets, args.config, args.checkpoint, args.align_head,
              args.hits.with_suffix(args.hits.suffix + ".json")]
    protected = {path.resolve() for path in inputs if path is not None}
    outputs = [args.out, args.out.with_name(args.out.name + ".meta.json")]
    if any(path.resolve() in protected for path in outputs):
        parser.error("Output must not overwrite input sequences, hits, provenance or model files")
    if args.index and any(path.resolve().is_relative_to(args.index.resolve()) for path in outputs):
        parser.error("Alignment output must be outside the target index directory")
    result = read_candidates(args.hits, args.score_mode)
    targets = Index.load(args.index, check_model=False) if args.index else args.targets
    if any(result.hits):
        # Resolve the optional dependency before allocating the large model.
        _clalign_engine()
        aligner = CandidateAligner.load(config_path=args.config, ckpt_path=args.checkpoint,
                                         align_head_path=args.align_head, device=args.device,
                                         max_cached_residues=args.cache_residues)
        aligned = aligner.align_candidates(result, args.queries, targets, top_k=args.topk,
                                           max_length=args.max_length, max_tokens=args.max_tokens)
    else:
        aligned = CandidateAlignmentResult([], dict(candidate_count=0, search_score_mode=result.score_mode,
                                                    search_metadata=result.metadata, backbone_checked=False,
                                                    model_loaded=False), dict(alignment_wall_sec=0.0))
    aligned.write_jsonl(args.out)
    print(f"Aligned {len(aligned.records)} candidates -> {args.out}")
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)
