"""Exhaustive target-block retrieval with bounded score/normalization buffers."""
from pathlib import Path
import os
import time

import numpy as np

from .index import _INHERIT, _digest, _records, _save_json, _top_indices, _unit, _vectors
from .index_data import PfamFilter
from .scoring import fixed_scores


def _hit(index, column, score, rank):
    if index.store is not None:
        target, row = index.store.row(column)
    else:
        target = index.ids[int(column)]
        row = index.target_metadata.get(target, {})
    meta = {field: row.get(field, "") for field in index.metadata_columns}
    return dict(rank=rank, target=target, target_index=int(column), score=float(score), meta=meta)


def search_vectors(index, embeddings, query_ids, *, identity=None, max_length=_INHERIT,
                   top_k=10, topk=None, score_mode=None, metric=None, normalize=None,
                   exclude_self=False, query_block_size=32, chunk_size=100000,
                   target_block_size=None, prefilter=None, query_pfam=None,
                   exact=None, score_matrix=None):
    from .index import SearchResult
    started = time.perf_counter()
    if identity is None:
        raise ValueError("Vector queries require model identity and max_length provenance")
    if identity != index.metadata["setup"]["encoder"]:
        raise ValueError("Vector model identity does not match the index")
    if max_length is _INHERIT:
        raise ValueError("Vector queries must declare max_length provenance")
    max_length = index._query_length(max_length)
    query_ids, _ = _records(query_ids, ["X"] * len(query_ids))
    q = _vectors(embeddings, len(query_ids), index.metadata["dimension"])
    if topk is not None:
        if top_k != 10 and top_k != topk:
            raise ValueError("topk and top_k disagree")
        top_k = topk
    if target_block_size is not None:
        chunk_size = target_block_size
    for name, value in (("top_k", top_k), ("chunk_size", chunk_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if query_block_size is not None and (not isinstance(query_block_size, int) or query_block_size < 1):
        raise ValueError("query_block_size must be positive or None")
    if exact is not None and not isinstance(exact, bool):
        raise ValueError("exact must be True, False or None")
    if exact is False:
        raise ValueError("Approximate search is not supported; retrieval is always exhaustive")
    aliases = {"inner": "inner", "cosine": "cosine", "euclidean": "negative_l2", "negative_l2": "negative_l2"}
    if metric is not None:
        if metric not in aliases or (score_mode is not None and score_mode != aliases[metric]):
            raise ValueError("Invalid or conflicting metric/score_mode")
        score_mode = aliases[metric]
    score_mode = score_mode or "inner"
    if score_mode not in ("inner", "cosine", "negative_l2"):
        raise ValueError("Unknown score_mode; historical benchmark scoring belongs in reproduce/")
    raw = index.metadata["setup"]["encoder"].get("mode") == "raw"
    normalize = bool(normalize)
    if raw and not normalize and score_mode != "cosine":
        raise ValueError("Raw retrieval requires explicit normalize=True / --normalize")
    normalize = normalize or score_mode == "cosine"
    prefilter = None if prefilter in (None, "full") else prefilter
    filtering = PfamFilter(index, query_ids, query_pfam, prefilter) if prefilter else None
    normalized = normalize or index.metadata["setup"]["encoder"].get("mode") in ("tm","ec")
    result_meta = dict(exact=True, prefilter=prefilter, normalized=normalized,
                       normalize_before_scoring=normalize,
                       max_length=max_length, encoder=identity, metadata_columns=index.metadata_columns,
                       numerical_policy="float32 feature-order multiply/add; block-invariant v1")
    result_meta["index_setup_digest"] = _digest(index.metadata["setup"])
    timing = dict(normalization_sec=0., scoring_sec=0., score_transform_sec=0., ranking_sec=0., score_matrix_write_sec=0.)
    destination = Path(score_matrix) if score_matrix is not None else None
    matrix = None
    tmp = None
    completed = False
    hits = []
    maximum = [0, 0]
    try:
        if destination is not None:
            protected = ("vectors.npy", "meta.json", "ids.txt", "records.sqlite", "embeddings.npy", "index.json", "ids.json", "metadata.tsv", "pfam.json", "pfam_clans.json")
            if any(destination.resolve() == (index.directory / name).resolve() for name in protected):
                raise ValueError("Score output must not overwrite index files")
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_name(destination.name + ".tmp.npy")
            header = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(q), len(index.ids)))
            matrix_offset = header.offset
            del header
            matrix = tmp.open("r+b")
        block = query_block_size or len(q)
        k = min(top_k, len(index.ids))
        self_columns = {key: index._columns.get(key, -1) for key in query_ids} if exclude_self else {}
        for start in range(0, len(q), block):
            keys = query_ids[start:start+block]
            tic = time.perf_counter()
            queries = _unit(q[start:start+block]) if normalize else q[start:start+block]
            timing["normalization_sec"] += time.perf_counter() - tic
            best_scores = [np.empty(0, dtype=np.float32) for _ in keys]
            best_cols = [np.empty(0, dtype=np.int64) for _ in keys]
            for offset in range(0, len(index.ids), chunk_size):
                tic = time.perf_counter()
                targets = _vectors(index.target_block(offset,min(offset+chunk_size,len(index.ids))),
                                   min(chunk_size,len(index.ids)-offset), index.metadata["dimension"])
                if normalize:
                    targets = _unit(targets)
                timing["normalization_sec"] += time.perf_counter() - tic
                tic = time.perf_counter()
                scores = fixed_scores(queries, targets, negative_l2=score_mode == "negative_l2")
                timing["scoring_sec"] += time.perf_counter() - tic
                tic = time.perf_counter()
                if not np.isfinite(scores).all():
                    raise ValueError("Scoring produced nonfinite values")
                maximum = [max(maximum[0],len(keys)), max(maximum[1],len(targets))]
                if filtering:
                    filtering.apply(scores, keys, index.ids[offset:offset+len(targets)], offset=offset)
                if exclude_self:
                    for i,key in enumerate(keys):
                        column = self_columns[key]-offset
                        if 0 <= column < len(targets):
                            scores[i,column] = -np.inf
                timing["score_transform_sec"] += time.perf_counter() - tic
                if matrix is not None:
                    tic = time.perf_counter()
                    for i, row in enumerate(scores):
                        matrix.seek(matrix_offset + ((start+i)*len(index.ids)+offset)*np.dtype(np.float32).itemsize)
                        row.tofile(matrix)
                    timing["score_matrix_write_sec"] += time.perf_counter() - tic
                tic = time.perf_counter()
                for i,row in enumerate(scores):
                    take = _top_indices(row,k)
                    cols = np.concatenate((best_cols[i],take+offset))
                    values = np.concatenate((best_scores[i],row[take]))
                    order = np.lexsort((cols,-values))[:k]
                    best_cols[i],best_scores[i] = cols[order],values[order]
                timing["ranking_sec"] += time.perf_counter() - tic
            hits.extend([_hit(index,c,s,r+1) for r,(c,s) in enumerate(zip(cols,values))]
                        for cols,values in zip(best_cols,best_scores))
        if matrix is not None:
            tic = time.perf_counter()
            matrix.flush()
            matrix.close()
            matrix = None
            os.replace(tmp,destination)
            _save_json(destination.with_suffix(destination.suffix+".json"),
                       dict(query_ids=query_ids,target_ids_digest=index.metadata["ids_digest"],
                            score_mode=score_mode,normalized=normalized,normalize_before_scoring=normalize,prefilter=prefilter,
                            exclude_self=exclude_self,shape=[len(q),len(index.ids)],
                            index_setup_digest=_digest(index.metadata["setup"])))
            timing["score_matrix_write_sec"] += time.perf_counter() - tic
        completed = True
    finally:
        if matrix is not None:
            matrix.close()
        if not completed and tmp is not None:
            tmp.unlink(missing_ok=True)
    timing["search_wall_sec"] = time.perf_counter() - started
    result_meta.update(max_score_block=maximum,chunk_size=chunk_size,query_block_size=query_block_size,
                       pfam_fallback_queries=sum(filtering.fallback.values()) if filtering else 0)
    return SearchResult(query_ids,hits,score_mode,timing,metadata=result_meta)
