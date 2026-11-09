"""Align retrieved candidates using residue features and the separate align head.

The alignment kernel is imported only when alignment is requested. Upstream
CLAlign is used when it is installed; otherwise the vendored subset in
``_clalign_core`` is used, so this repository aligns on its own. Retrieval
scores and ranks are preserved; alignment does not silently rerank candidates.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from .heads import apply_align_head, load_align_head, residue_similarity
from .index import SearchResult, _length, _records, file_sha256


def _backbone_identity(identity):
    """Ignore retrieval heads/mode while binding alignments to the same backbone."""
    files = identity.get("files", {})
    return dict(backend=identity.get("backend"),
                files={key: files[key] for key in ("config", "checkpoint") if key in files},
                source=identity.get("source", {}))


def _candidate_sequences(source, needed):
    """Read selected IDs only, from mappings, stored indexes or streaming FASTA."""
    needed = dict.fromkeys(needed)
    if not needed:
        return {}
    if hasattr(source, "get_sequences"):
        return source.get_sequences(needed)
    if isinstance(source, Mapping):
        return {key: source[key] for key in needed}
    if not isinstance(source, (str, Path)):
        raise TypeError("Sequences must be an ID mapping, FASTA path or sequence-containing Index")
    selected = {}
    current, buffer = None, []

    def finish():
        if current in needed:
            if current in selected:
                raise ValueError(f"Duplicate candidate FASTA ID: {current}")
            selected[current] = "".join(buffer)

    with Path(source).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header = line[1:].split()
                if not header:
                    raise ValueError("Empty FASTA header")
                current, buffer = header[0], []
            elif current is None:
                raise ValueError("Sequence before the first FASTA header")
            elif current in needed:
                buffer.append(line)
        finish()
    for key in needed:
        if key not in selected:
            raise KeyError(f"Missing candidate sequence: {key}")
    return selected


def _clalign_engine():
    """Resolve the global alignment kernel, preferring an installed CLAlign.

    A machine with CLAlign keeps the exact code path that produced the published
    numbers; everywhere else the vendored subset runs the same kernel. Building
    or searching an index needs neither.
    """
    try:
        from clalign.alignment import ProteinSeq, align_core
        source = "clalign.align_core"
    except ImportError:
        try:
            from ._clalign_core import ProteinSeq, align_core
            source = "splm_tm._clalign_core.align_core"
        except ImportError as exc:
            raise ImportError(
                "Candidate alignment needs numba for the alignment kernel: "
                "pip install -e \".[align]\" in the repository root, or "
                "pip install numba. Installing CLAlign "
                "(https://github.com/yourh/CLAlign) and making its src directory "
                "importable also works and takes precedence. Building or searching "
                "an index requires neither."
            ) from exc

    def run(query, target, matrix):
        result = align_core(ProteinSeq(query), ProteinSeq(target),
                            np.ascontiguousarray(matrix, dtype=np.float32),
                            gap_penalty=0.0, local=False)
        return dict(aligned_query=str(result.aln1), aligned_target=str(result.aln2),
                    matched_pairs=[[int(i), int(j)] for i, j in result.get_matched_res()],
                    alignment_score=float(result.score), query_coverage=float(result.coverage))
    run.source = source
    return run


@dataclass
class CandidateAlignmentResult:
    records: list[dict]
    metadata: dict
    timings: dict = field(default_factory=dict)

    def to_dataframe(self):
        """Return records as a DataFrame; pandas remains optional."""
        import pandas as pd
        return pd.DataFrame(self.records, columns=list(self.records[0]) if self.records else
                            ["query", "target", "search_rank", "search_score", "alignment_score",
                             "aligned_query", "aligned_target", "matched_pairs"])

    def write_jsonl(self, path):
        """Write alignments and a .jsonl.meta.json provenance/timing sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        path.with_name(path.name + ".meta.json").write_text(
            json.dumps(dict(metadata=self.metadata, timings=self.timings),
                       indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path

    to_jsonl = write_jsonl


class CandidateAligner:
    """ProtT5 candidate alignment; full sequences by default.

    Pass the loaded backbone and its compatible align head, or use
    ``from_search_encoder`` to reuse a T5 search model. Residues are encoded
    individually; an LRU cache bounds retained projected features by residue
    count. The substitution matrix is materialized for one pair at a time.
    ``engine`` is an optional callable(query, target, float32_matrix) returning
    the same fields as the CLAlign adapter, intended for explicit integrations.
    """

    def __init__(self, backbone, align_head, device="cpu", *,
                 max_cached_residues=32768, engine=None, identity=None):
        if not isinstance(max_cached_residues, int) or isinstance(max_cached_residues, bool) or max_cached_residues < 0:
            raise ValueError("max_cached_residues must be a nonnegative integer")
        self.backbone, self.head, self.device = backbone, align_head.eval(), device
        self.max_cached_residues = max_cached_residues
        self.engine, self.identity = engine, dict(identity or {})
        self._cache = OrderedDict()
        self._cached_residues = 0

    @classmethod
    def load(cls, *, config_path=None, ckpt_path=None, align_head_path=None,
             device=None, verbose=True, **kwargs):
        """Load a standalone T5 residue aligner without TM or EC retrieval heads."""
        from .search_encoders import T5SearchEncoder
        encoder = T5SearchEncoder.load(mode="raw", config_path=config_path, ckpt_path=ckpt_path,
                                       device=device, verbose=verbose)
        return cls.from_search_encoder(encoder, align_head_path=align_head_path, **kwargs)

    @classmethod
    def from_search_encoder(cls, encoder, align_head_path=None, **kwargs):
        from .paths import default_align_head
        from .search_encoders import T5SearchEncoder

        if not isinstance(encoder, T5SearchEncoder):
            raise TypeError("This alignment adapter requires a T5SearchEncoder and a paired T5 align head")
        path = Path(align_head_path) if align_head_path else default_align_head()
        head, meta = load_align_head(path, device=encoder.model.head_device)
        identity = dict(search_encoder=encoder.identity, align_head_sha256=file_sha256(path),
                        align_head_architecture={key: meta[key] for key in
                                                 ("dim", "hidden", "head_type", "mlp_blocks")
                                                 if key in meta})
        return cls(encoder.model.backbone, head, encoder.model.head_device,
                   identity=identity, **kwargs)

    def clear_cache(self):
        self._cache.clear()
        self._cached_residues = 0

    def _encode(self, sequence, max_tokens):
        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return self._cache[sequence]
        raw = self.backbone.encode_residues([sequence], ids=["alignment"],
                                           batch_size=1, max_length=None,
                                           max_tokens=max_tokens, verbose=False)
        if len(raw) != 1 or np.asarray(raw[0]).ndim != 2 or len(raw[0]) != len(sequence):
            raise ValueError("Residue encoder returned the wrong sequence length or shape")
        features = np.asarray(apply_align_head(self.head, raw, device=self.device)[0],
                              dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite residue embeddings")
        if len(sequence) <= self.max_cached_residues:
            while self._cached_residues + len(sequence) > self.max_cached_residues:
                old, _ = self._cache.popitem(last=False)
                self._cached_residues -= len(old)
            self._cache[sequence] = features
            self._cached_residues += len(sequence)
        return features

    def align_candidates(self, result: SearchResult, query_sequences, target_sequences,
                         *, top_k=None, max_length=None, max_tokens=8192):
        """Align selected hits from mappings, FASTA paths or a target Index.

        ``max_length`` is independent of search truncation: None/0 means full,
        otherwise align prefixes. Matched residue pairs use zero-based indices
        in those prefixes (also valid coordinates in the original sequences).
        All selected IDs/sequences are validated before model inference starts.
        """
        max_length = _length(max_length)
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        expected = self.identity.get("search_encoder")
        actual = result.metadata.get("encoder")
        checked = expected is not None and actual is not None
        if checked and _backbone_identity(expected) != _backbone_identity(actual):
            raise ValueError("Alignment backbone does not match the search backbone")
        if hasattr(target_sequences, "meta") and result.metadata.get("index_setup_digest"):
            from .index import _digest
            if _digest(target_sequences.meta["setup"]) != result.metadata["index_setup_digest"]:
                raise ValueError("Target index does not match the search result")
        if len(result.query_ids) != len(result.hits) or len(set(result.query_ids)) != len(result.query_ids):
            raise ValueError("Invalid SearchResult query IDs or hit lists")
        selected = []
        for query, hits in zip(result.query_ids, result.hits):
            previous_rank, seen = 0, set()
            for hit in hits if top_k is None else hits[:top_k]:
                target, rank, score = hit["target"], hit["rank"], hit["score"]
                if (not isinstance(rank, int) or isinstance(rank, bool) or rank <= previous_rank
                        or target in seen or not np.isfinite(score)):
                    raise ValueError("Candidate ranks must increase, IDs must be unique, and scores finite")
                previous_rank = rank
                seen.add(target)
                selected.append((query, target, rank, float(score)))
        query_sequences = _candidate_sequences(query_sequences, [item[0] for item in selected])
        target_sequences = _candidate_sequences(target_sequences, [item[1] for item in selected])
        queries, targets = {}, {}
        for query, target, _, _ in selected:
            for identifier, source, dest in ((query, query_sequences, queries),
                                              (target, target_sequences, targets)):
                if identifier not in source:
                    raise KeyError(f"Missing candidate sequence: {identifier}")
                sequence = source[identifier]
                _records([identifier], [sequence])
                if "-" in sequence or "*" in sequence:
                    raise ValueError("Alignment inputs must be ungapped protein sequences without stop tokens")
                dest[identifier] = sequence.upper()
        engine = (self.engine or _clalign_engine()) if selected else None
        started = time.perf_counter()
        records = []
        target_metadata = {(q,h["target"]):dict(h.get("meta",{}))
                           for q,hits in zip(result.query_ids,result.hits) for h in hits}
        grouped = {query: [] for query in result.query_ids}
        for item in selected:
            grouped[item[0]].append(item)
        for query, chosen in grouped.items():
            if not chosen:
                continue
            qseq = queries[query][:max_length]
            qfeatures = self._encode(qseq, max_tokens)
            for _, target, rank, score in chosen:
                tseq = targets[target][:max_length]
                matrix = residue_similarity(qfeatures, self._encode(tseq, max_tokens))
                alignment = engine(qseq, tseq, matrix)
                del matrix
                records.append(dict(**alignment, query=query, target=target,
                                    search_rank=rank, search_score=score,
                                    target_metadata=target_metadata.get((query,target),{}),
                                    query_length=len(queries[query]), target_length=len(targets[target]),
                                    aligned_query_length=len(qseq), aligned_target_length=len(tseq)))
        metadata = dict(identity=self.identity, search_score_mode=result.score_mode,
                        search_metadata=result.metadata, backbone_checked=checked,
                        adapter_source_sha256=file_sha256(__file__),
                        engine="custom" if self.engine else getattr(engine, "source", None),
                        global_alignment=True, gap_penalty=0.0,
                        max_length=max_length, residue_batch_size=1, max_tokens=max_tokens,
                        coordinate_system="zero-based original-sequence residue indices",
                        candidate_count=len(records))
        if self.engine:
            metadata["global_alignment"] = None
            metadata["gap_penalty"] = None
        return CandidateAlignmentResult(records, metadata,
                                        dict(alignment_wall_sec=time.perf_counter() - started))


__all__ = ["CandidateAligner", "CandidateAlignmentResult"]
