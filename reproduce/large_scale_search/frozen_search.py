"""Frozen database construction, scoring and timing for the large-scale case.

A frozen copy of the release search code from snapshot 99385a4052bd,
restricted to what this case uses: full-length T5
TM-head encoding, original short/long partition and shards, benchmark_tm
scores and the original speed-only scoring profile. It deliberately does not
import the public search API, which may change independently. Real-model parity
with the historical Hellbender results has not yet been validated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

_CODE = Path(__file__).resolve().parents[2] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from splm_tm.fasta import read_fasta  # noqa: E402


def _records(ids, seqs):
    ids, seqs = list(ids), list(seqs)
    if not ids or len(ids) != len(seqs):
        raise ValueError("ids and sequences must be nonempty and have equal length")
    if any(not isinstance(x, str) or not x or any(c.isspace() for c in x) for x in ids):
        raise ValueError("IDs must be nonempty strings without whitespace")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sequence IDs are not supported")
    if any(not isinstance(x, str) or not x or any(c.isspace() for c in x) for x in seqs):
        raise ValueError("Sequences must be nonempty strings without whitespace")
    return ids, seqs


def _save_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _save_npy(path: Path, value: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(tmp, path)


def _length(value) -> Optional[int]:
    if value is None or value == 0:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_length must be a positive integer, 0 or None (full length)")
    return value


def _vectors(value, rows, dimension=None):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != rows or arr.shape[1] == 0:
        raise ValueError(f"Invalid embedding shape: {arr.shape}")
    if dimension is not None and arr.shape[1] != dimension:
        raise ValueError(f"Embedding dimension {arr.shape[1]} != index dimension {dimension}")
    if not np.isfinite(arr).all():
        raise ValueError("Embeddings contain nonfinite values")
    return arr


def _unit(arr):
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), np.float32(1e-12))


def _synchronize(encoder):
    sync = getattr(encoder, "synchronize", None)
    if sync is not None:
        sync()


def _top_indices(scores, k):
    """Same result as a stable descending full sort, including boundary ties."""
    valid = np.flatnonzero(np.isfinite(scores))
    k = min(k, len(valid))
    if k == 0:
        return valid[:0]
    if k < len(valid):
        values = scores[valid]
        cut = np.partition(values, len(values) - k)[len(values) - k]
        above = valid[values > cut]
        tied = valid[values == cut][:k - len(above)]
        valid = np.sort(np.concatenate((above, tied)))
    return valid[np.argsort(-scores[valid], kind="stable")][:k]


class FrozenT5Encoder:
    """Full T5 backbone pooling followed by the paired TM head in 4,096-row calls."""

    def __init__(self, model, identity, head_batch_size=4096):
        if head_batch_size < 1:
            raise ValueError("Invalid head batch size")
        self.model, self.identity = model, identity
        self.head_batch_size = head_batch_size

    @classmethod
    def load(cls, config_path=None, ckpt_path=None, tm_head_path=None, device=None,
             head_batch_size=4096, verbose=True):
        from splm_tm import paths
        from splm_tm.api import SPLMTM

        if head_batch_size < 1:
            raise ValueError("Invalid head batch size")
        config_path = Path(config_path) if config_path else paths.default_config()
        ckpt_path = Path(ckpt_path) if ckpt_path else paths.default_ckpt()
        tm_head_path = Path(tm_head_path) if tm_head_path else paths.default_tm_head()
        files = dict(config=config_path, checkpoint=ckpt_path, tm_head=tm_head_path)
        identity = dict(backend="prot_t5", mode="tm",
                        files={name: str(path) for name, path in files.items()},
                        head_batch_size=head_batch_size)
        model = SPLMTM.load(modes=("tm",), config_path=config_path, ckpt_path=ckpt_path,
                            tm_head_path=tm_head_path, device=device, verbose=verbose)
        return cls(model, identity, head_batch_size)

    def encode(self, seqs, ids, max_length=None, batch_size=8, max_tokens=8192):
        import torch

        raw = self.model.backbone.encode(seqs, ids, max_length=max_length,
                                         batch_size=batch_size, max_tokens=max_tokens,
                                         verbose=False)
        if raw.ndim != 2 or raw.shape[1] != 1024:
            raise RuntimeError("ProtT5 search requires 1,024-dimensional raw embeddings")
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(raw), self.head_batch_size):
                value = torch.from_numpy(raw[start:start + self.head_batch_size]).float()
                chunks.append(self.model.tm_head(value.to(self.model.head_device)).float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def synchronize(self):
        import torch

        device = torch.device(self.model.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@dataclass
class SearchResult:
    query_ids: list[str]
    hits: list[list[dict]]
    score_mode: str
    timings: dict = field(default_factory=dict)
    query_embeddings: Optional[np.ndarray] = field(default=None, repr=False)

    def write_tsv(self, path):
        started = time.perf_counter()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("query\trank\ttarget\tscore\n")
            for query, hits in zip(self.query_ids, self.hits):
                for hit in hits:
                    # Nine significant digits round-trip a float32 score.
                    handle.write(f"{query}\t{hit['rank']}\t{hit['target']}\t{hit['score']:.9g}\n")
        self.timings["hit_table_write_sec"] = time.perf_counter() - started
        return path


class FrozenSearchIndex:
    """Resumable target database with input/model provenance.

    Do not build into the same directory from multiple processes. Run encoding
    on a compute node; already completed and verified shards survive interruption.
    """

    SCHEMA = 1

    def __init__(self, directory, metadata, ids, embeddings):
        self.directory = Path(directory)
        self.metadata = metadata
        self.ids = ids
        self.embeddings = embeddings
        self._columns = {identifier: i for i, identifier in enumerate(ids)}

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if metadata.get("schema") != cls.SCHEMA or metadata.get("status") != "complete":
            raise ValueError("Index is incomplete or has an unsupported schema")
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        if len(ids) != len(set(ids)) or len(ids) != metadata["count"]:
            raise ValueError("Index ID count or uniqueness check failed")
        arr = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        if arr.shape != (len(ids), metadata["dimension"]) or arr.dtype != np.float32:
            raise ValueError("Index embeddings have the wrong shape or dtype")
        return cls(directory, metadata, ids, arr)

    @classmethod
    def build_fasta(cls, encoder, fasta, directory, **kwargs):
        ids, seqs = read_fasta(fasta)
        return cls.build(encoder, seqs, ids, directory, **kwargs)

    @classmethod
    def build(cls, encoder, seqs: Sequence[str], ids: Sequence[str], directory,
              max_length=None, batch_size: int = 8, max_tokens: int = 8192,
              shard_size: int = 20000, split_length: Optional[int] = 1024,
              resume: bool = True, verbose: bool = True):
        """Original-order shards within a short/long partition at split_length.

        Within each shard, the encoder sorts by length and restores the supplied
        order. The partition controls batching only; it does not truncate.
        """
        build_started = time.perf_counter()
        ids, seqs = _records(ids, seqs)
        max_length = _length(max_length)
        if min(batch_size, max_tokens, shard_size) < 1:
            raise ValueError("Batch, token and shard limits must be positive")
        if split_length is not None and split_length < 1:
            raise ValueError("split_length must be positive or None")
        identity = json.loads(json.dumps(encoder.identity, sort_keys=True))
        setup = dict(encoder=identity, count=len(ids),
                     max_length=max_length, batch_size=batch_size, max_tokens=max_tokens,
                     shard_size=shard_size, split_length=split_length)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "index.json"
        if manifest.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            saved_ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
            if meta.get("schema") != cls.SCHEMA or meta.get("setup") != setup or saved_ids != ids:
                raise ValueError("Existing index uses different inputs, weights or encoding settings")
            if not resume:
                raise FileExistsError("Index already exists; use resume or a new output directory")
            if meta.get("status") == "complete":
                return cls.load(directory)
        else:
            if any(directory.iterdir()):
                raise FileExistsError("New index directory must be empty")
            meta = dict(schema=cls.SCHEMA, status="building", setup=setup, shards={},
                        count=len(ids))
            _save_json(directory / "ids.json", ids)
            _save_json(manifest, meta)
        cache = directory / "shards"
        cache.mkdir(exist_ok=True)
        all_indices = list(range(len(ids)))
        if split_length is None:
            groups = [("all", all_indices)]
        else:
            groups = [("short", [i for i in all_indices if len(seqs[i]) <= split_length]),
                      ("long", [i for i in all_indices if len(seqs[i]) > split_length])]
        planned = []
        for group, indices in groups:
            for number, start in enumerate(range(0, len(indices), shard_size)):
                name = f"{group}-{number:05d}"
                selected = indices[start:start + shard_size]
                path = cache / f"{name}.npy"
                planned.append((name, selected, path))
                # Shards are written atomically and recorded afterwards, so a
                # recorded shard file is complete.
                if meta["shards"].get(name) and path.is_file():
                    _vectors(np.load(path, allow_pickle=False), len(selected), meta.get("dimension"))
                    continue
                started = time.perf_counter()
                _synchronize(encoder)
                encode_started = time.perf_counter()
                chunk = encoder.encode([seqs[i] for i in selected], [ids[i] for i in selected],
                                       max_length=max_length, batch_size=batch_size,
                                       max_tokens=max_tokens)
                _synchronize(encoder)
                encode_sec = time.perf_counter() - encode_started
                chunk = _vectors(chunk, len(selected), meta.get("dimension"))
                meta["dimension"] = int(chunk.shape[1])
                _save_npy(path, chunk)
                meta["shards"][name] = dict(count=len(selected),
                                             seconds=time.perf_counter() - started,
                                             encoding_sec=encode_sec)
                _save_json(manifest, meta)
                if verbose:
                    print(f"index {name}: {len(selected)} sequences", flush=True)
        tmp = directory / "embeddings.tmp.npy"
        merge_started = time.perf_counter()
        combined = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                             shape=(len(ids), meta["dimension"]))
        for _, selected, path in planned:
            combined[selected] = np.load(path, mmap_mode="r", allow_pickle=False)
        combined.flush()
        del combined
        os.replace(tmp, directory / "embeddings.npy")
        merge_sec = time.perf_counter() - merge_started
        meta["status"] = "complete"
        meta["timings"] = dict(
            target_encoding_sec=sum(part["encoding_sec"] for part in meta["shards"].values())
            if all("encoding_sec" in part for part in meta["shards"].values()) else None,
            shard_processing_sec=sum(part["seconds"] for part in meta["shards"].values()),
            merge_write_sec=merge_sec,
            build_wall_sec_this_run=time.perf_counter() - build_started,
            note="Encoding/shard totals include resumed shards; wall time describes this invocation.")
        _save_json(manifest, meta)
        return cls.load(directory)

    def search(self, encoder, seqs, ids, top_k: int = 10, max_length=None,
               query_batch_size: int = 1, max_tokens: int = 8192, warmup: bool = False, **kwargs):
        """Encode queries, by default one at a time, then search this index.

        The encoder/weights/head identity must match the database.
        """
        ids, seqs = _records(ids, seqs)
        if encoder.identity != self.metadata["setup"]["encoder"]:
            raise ValueError("Query encoder does not match the index model/head/code")
        max_length = _length(max_length)
        if query_batch_size < 1 or max_tokens < 1:
            raise ValueError("Query batch and token limits must be positive")
        _synchronize(encoder)
        warmup_sec = 0.0
        if warmup:
            started = time.perf_counter()
            encoder.encode(seqs[:1], ids[:1], max_length=max_length, batch_size=1, max_tokens=max_tokens)
            _synchronize(encoder)
            warmup_sec = time.perf_counter() - started
        started = time.perf_counter()
        blocks = []
        for start in range(0, len(ids), query_batch_size):
            blocks.append(encoder.encode(seqs[start:start + query_batch_size],
                                         ids[start:start + query_batch_size],
                                         max_length=max_length, batch_size=query_batch_size,
                                         max_tokens=max_tokens))
        _synchronize(encoder)
        encoding_sec = time.perf_counter() - started
        embeddings = np.concatenate(blocks, axis=0)
        result = self.search_embeddings(embeddings, ids, top_k=top_k, **kwargs)
        result.query_embeddings = embeddings
        result.timings.update(query_encoding_sec=encoding_sec, warmup_sec=warmup_sec,
                              online_wall_sec=encoding_sec + result.timings["search_wall_sec"])
        return result

    def search_fasta(self, encoder, fasta, **kwargs):
        ids, seqs = read_fasta(fasta)
        return self.search(encoder, seqs, ids, **kwargs)

    def search_embeddings(self, embeddings, query_ids, top_k: int = 10,
                          score_mode: str = "benchmark_tm", exclude_self: bool = False,
                          query_block_size: Optional[int] = None, score_matrix=None):
        """Frozen accuracy scores: normalize queries and targets, compute float32
        inner products, rescale (1 + s) / 2 in float64, then cast to float32.

        query_block_size=None is the original full query-matrix GEMM. A saved
        score matrix is finite and unmasked; self-hits are excluded only from
        the hit table, so the evaluator applies its own rules.
        """
        search_started = time.perf_counter()
        timings = dict(normalization_sec=0.0, scoring_sec=0.0, score_transform_sec=0.0,
                       score_matrix_write_sec=0.0, ranking_sec=0.0)
        query_ids = list(query_ids)
        q = _vectors(embeddings, len(query_ids), self.metadata["dimension"])
        if top_k < 1 or (query_block_size is not None and query_block_size < 1):
            raise ValueError("top_k and query_block_size must be positive")
        if score_mode != "benchmark_tm":
            raise ValueError("The large-scale case uses only benchmark_tm scores")
        if not query_ids:
            raise ValueError("At least one query is required")
        started = time.perf_counter()
        q = _unit(q)
        db = _unit(self.embeddings)
        timings["normalization_sec"] = time.perf_counter() - started
        destination = Path(score_matrix) if score_matrix is not None else None
        matrix = None
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_name(destination.name + ".tmp.npy")
            if any(destination.resolve() == (self.directory / name).resolve()
                   for name in ("embeddings.npy", "index.json", "ids.json")):
                raise ValueError("Score matrix must not overwrite index files")
            matrix = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                               shape=(len(q), len(self.ids)))
        hits = []
        block = query_block_size or len(q)
        try:
            for start in range(0, len(q), block):
                chunk = q[start:start + block]
                started = time.perf_counter()
                scores = chunk @ db.T
                timings["scoring_sec"] += time.perf_counter() - started
                started = time.perf_counter()
                scores = ((scores.astype(np.float64) + 1.0) / 2.0).astype(np.float32)
                timings["score_transform_sec"] += time.perf_counter() - started
                if not np.isfinite(scores).all():
                    raise ValueError("Scoring produced nonfinite values")
                if matrix is not None:
                    started = time.perf_counter()
                    matrix[start:start + len(chunk)] = scores
                    timings["score_matrix_write_sec"] += time.perf_counter() - started
                started = time.perf_counter()
                for offset, row in enumerate(scores):
                    if exclude_self:
                        column = self._columns.get(query_ids[start + offset])
                        if column is not None:
                            row = row.copy()
                            row[column] = -np.inf
                    columns = _top_indices(row, top_k)
                    hits.append([dict(rank=rank + 1, target=self.ids[int(column)],
                                      target_index=int(column), score=float(row[column]))
                                 for rank, column in enumerate(columns)])
                timings["ranking_sec"] += time.perf_counter() - started
            if matrix is not None:
                started = time.perf_counter()
                matrix.flush()
                del matrix
                matrix = None
                os.replace(tmp, destination)
                _save_json(destination.with_suffix(destination.suffix + ".json"),
                           dict(query_ids=query_ids, target_ids=self.ids,
                                score_mode=score_mode, shape=[len(q), len(self.ids)]))
                timings["score_matrix_write_sec"] += time.perf_counter() - started
        finally:
            if matrix is not None:
                del matrix
        timings["search_wall_sec"] = time.perf_counter() - search_started
        return SearchResult(query_ids, hits, score_mode, timings)

    def benchmark_scoring(self, embeddings, directory, repeats: int = 3):
        """Original speed-only scope, separate from accuracy and top-k search.

        Normalize query vectors, warm database pages and BLAS before timing.
        Measure float32 q @ db.T, float32 (score+1)/2 and np.save; exclude sorting,
        normalization, database loading, model loading, and warm-up. Targets must
        already be normalized TM vectors. This does not replace accuracy scores.
        """
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if self.metadata["setup"]["encoder"].get("mode") != "tm":
            raise ValueError("Original scoring timing requires a TM embedding index")
        q = _vectors(embeddings, len(embeddings), self.metadata["dimension"])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        db = self.embeddings
        _ = float(db[:, 0].sum())
        _ = q[:2] @ np.asarray(db[:1024]).T
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "benchmark_score.tmp.npy"
        if path.exists():
            raise FileExistsError(path)
        seconds = []
        try:
            for _ in range(repeats):
                started = time.perf_counter()
                scores = q @ db.T
                scores = (scores + np.float32(1.0)) * np.float32(0.5)
                np.save(path, scores)
                seconds.append(time.perf_counter() - started)
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite timing scores")
                del scores
        finally:
            path.unlink(missing_ok=True)
        return dict(seconds=seconds, median_sec=float(np.median(seconds)), repeats=repeats,
                    queries=len(q), targets=len(self.ids),
                    scope="float32 matrix product, score rescaling, matrix output; excludes sorting")


__all__ = ["FrozenSearchIndex", "FrozenT5Encoder", "SearchResult"]
