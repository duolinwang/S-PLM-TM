"""Unified query dispatch and strict model/length checks for persistent indexes."""
from pathlib import Path
from collections.abc import Mapping
import time

import numpy as np


class QueryAPI:
    def search(self, queries, *legacy, encoder=None, ids=None, topk=None, top_k=10,
               query_batch_size=1, max_tokens=8192, warmup=False, **kwargs):
        """Search FASTA, sequences, ID-to-sequence mappings or QueryEmbeddings.

        The previous search(encoder, sequences, ids) form remains supported.
        Unspecified query truncation inherits the database setting.
        """
        from .index import QueryEmbeddings, _INHERIT, _length, _records, _synchronize
        from .fasta import read_fasta, make_ids
        if hasattr(queries, "encode") and not isinstance(queries, str):
            if len(legacy) != 2:
                raise TypeError("Expected search(encoder, sequences, ids)")
            encoder, queries, ids = queries, legacy[0], legacy[1]
        elif legacy:
            raise TypeError("Supply query IDs with ids=")
        if topk is not None:
            if top_k != 10 and top_k != topk:
                raise ValueError("topk and top_k disagree")
            top_k = topk
        if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if isinstance(queries, QueryEmbeddings):
            if ids is not None or "identity" in kwargs or "max_length" in kwargs:
                raise ValueError("QueryEmbeddings already contains IDs and provenance")
            return self.search_embeddings(queries.vectors, queries.ids, top_k=top_k,
                                          identity=queries.identity, max_length=queries.max_length, **kwargs)
        if isinstance(queries, np.ndarray):
            if ids is None:
                raise ValueError("Vector queries require ids and model/length provenance")
            return self.search_embeddings(queries, ids, top_k=top_k, **kwargs)
        if isinstance(queries, (str, Path)):
            if ids is not None:
                raise ValueError("FASTA contains query IDs; do not supply ids=")
            ids, queries = read_fasta(queries)
        elif isinstance(queries, Mapping):
            if ids is not None:
                raise ValueError("A sequence mapping already contains IDs")
            ids, queries = list(queries), list(queries.values())
        else:
            queries = list(queries)
            ids = list(ids) if ids is not None else make_ids(len(queries), prefix="query")
        ids, queries = _records(ids, queries)
        max_length = self._query_length(kwargs.pop("max_length", _INHERIT))
        encoder = encoder or self.encoder
        if encoder is None:
            from .search_encoders import T5SearchEncoder
            saved = self.metadata["setup"]["encoder"]
            if saved.get("backend") != "prot_t5":
                raise ValueError("Supply an encoder compatible with this index")
            encoder = T5SearchEncoder.load(mode=saved["mode"], head_batch_size=saved.get("head_batch_size", 4096))
        self._check_encoder(encoder)
        if query_batch_size < 1 or max_tokens < 1:
            raise ValueError("Query batch and token limits must be positive")
        _synchronize(encoder)
        warmup_sec = 0.0
        if warmup:
            started = time.perf_counter()
            encoder.encode(queries[:1], ids[:1], max_length=max_length, batch_size=1, max_tokens=max_tokens)
            _synchronize(encoder)
            warmup_sec = time.perf_counter() - started
        started = time.perf_counter()
        blocks = [encoder.encode(queries[s:s+query_batch_size], ids[s:s+query_batch_size],
                                 max_length=max_length, batch_size=query_batch_size, max_tokens=max_tokens)
                  for s in range(0, len(ids), query_batch_size)]
        _synchronize(encoder)
        encoding_sec = time.perf_counter() - started
        vectors = np.concatenate(blocks)
        result = self.search_embeddings(vectors, ids, identity=encoder.identity, max_length=max_length,
                                        top_k=top_k, **kwargs)
        result.query_embeddings = vectors
        result.timings.update(query_encoding_sec=encoding_sec, warmup_sec=warmup_sec,
                              online_wall_sec=encoding_sec + result.timings["search_wall_sec"])
        self.encoder = encoder
        return result

    def search_fasta(self, encoder, fasta, **kwargs):
        return self.search(fasta, encoder=encoder, **kwargs)

    def search_embeddings(self, embeddings, query_ids, *, identity=None, **kwargs):
        from .retrieval import search_vectors
        return search_vectors(self, embeddings, query_ids, identity=identity, **kwargs)


    def info(self):
        return dict(count=len(self.ids), dimension=self.metadata["dimension"],
                    has_sequences=self.metadata.get("has_sequences", False),
                    format_version=self.metadata.get("format_version",self.metadata["schema"]),
                    normalized=self.metadata.get("normalized"),created=self.metadata.get("created"),
                    mode=self.metadata["setup"]["encoder"].get("mode"),
                    max_length=self.metadata["setup"]["max_length"],
                    encoder=self.metadata["setup"]["encoder"],
                    has_pfam=(self.directory / "pfam.json").is_file(),
                    metadata_columns=self.metadata_columns)
