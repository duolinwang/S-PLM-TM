"""Persistent sequence indexes and deterministic, bounded-memory retrieval.

Loading checks a supplied or default model; explicit metadata-only loading avoids
model initialization. Encoders implement ``identity`` (a JSON-serializable model
fingerprint) and ``encode``.
The saved index always retains the original FASTA order.
"""
from __future__ import annotations

import hashlib
import csv
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .fasta import read_fasta
from .index_queries import QueryAPI

_INHERIT = object()


@dataclass
class QueryEmbeddings:
    """Vectors with their model identity, ID ordering and truncation provenance."""
    ids: Sequence[str]
    vectors: np.ndarray
    identity: dict
    max_length: Optional[int]

    def save(self, path):
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Query embeddings must use a .npz path")
        length = _length(self.max_length)
        _digest(self.identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids, _ = _records(self.ids, ["X"] * len(self.ids))
        np.savez(path, ids=np.asarray(ids,dtype=str), vectors=_vectors(self.vectors,len(ids)))
        _save_json(path.with_suffix(".npz.json"),dict(identity=self.identity,max_length=length,
                                                   vectors_sha256=file_sha256(path)))

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(path.suffix+".json").read_text(encoding="utf-8"))
        if meta.get("vectors_sha256") != file_sha256(path):
            raise ValueError("Query vector file does not match its provenance sidecar")
        with np.load(path,allow_pickle=False) as data:
            return cls(data["ids"].tolist(),data["vectors"].copy(),meta["identity"],meta["max_length"])


class HitRows(list):
    """Preserve positional access while allowing result.hits[query_id]."""
    def __init__(self, query_ids, rows):
        super().__init__(rows)
        self._positions = {key: i for i, key in enumerate(query_ids)}

    def __getitem__(self, key):
        if isinstance(key, str):
            return [HitTuple(row) for row in super().__getitem__(self._positions[key])]
        return super().__getitem__(key)


class HitTuple(tuple):
    """Public (target, score, metadata) tuple with compatible named lookups."""
    def __new__(cls, row):
        result = super().__new__(cls, (row["target"], row["score"], row.get("meta", {})))
        result.row = row
        return result

    def __getitem__(self, key):
        return self.row[key] if isinstance(key, str) else super().__getitem__(key)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


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
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_length must be a positive integer, 0 or None (full length)")
    return value or None


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
    from .scoring import squared_norms
    norms = np.sqrt(squared_norms(arr))[:, None]
    return arr / np.maximum(norms, np.float32(1e-12))


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


@dataclass
class SearchResult:
    query_ids: list[str]
    hits: list[list[dict]]
    score_mode: str
    timings: dict = field(default_factory=dict)
    query_embeddings: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.hits = HitRows(self.query_ids, self.hits)

    @classmethod
    def from_tsv(cls, path, score_mode=None):
        """Read public hit tables and their optional provenance sidecars.

        Legacy tables without a sidecar require an explicit score mode. Empty
        query rows are recovered from the sidecar rather than silently dropped.
        """
        path = Path(path)
        sidecar = path.with_suffix(path.suffix + ".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        if metadata.get("hit_table_sha256") not in (None, file_sha256(path)):
            raise ValueError("Hit table does not match its provenance sidecar")
        saved = metadata.get("score_mode")
        if saved is not None and score_mode is not None and saved != score_mode:
            raise ValueError("Requested score mode differs from the hit table")
        score_mode = saved or score_mode
        if score_mode not in ("inner", "cosine", "negative_l2"):
            raise ValueError("Hit tables without a score-mode sidecar require score_mode")
        query_ids = metadata.get("query_ids", [])
        if query_ids:
            _records(query_ids, ["X"] * len(query_ids))
        grouped = {key: [] for key in query_ids}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if (not reader.fieldnames or reader.fieldnames[:4] != ["query", "rank", "target", "score"]
                    or len(set(reader.fieldnames)) != len(reader.fieldnames)):
                raise ValueError("Expected unique TSV columns: query, rank, target, score, then optional metadata")
            fields = reader.fieldnames[4:]
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed candidate TSV row")
                key = row["query"]
                if query_ids and key not in grouped:
                    raise ValueError("Hit table query is absent from its provenance sidecar")
                grouped.setdefault(key, []).append(dict(target=row["target"], rank=int(row["rank"]),
                                                         score=float(row["score"]),
                                                         meta={f: row[f] for f in fields}))
        metadata = dict(metadata, metadata_columns=fields)
        return cls(list(grouped), list(grouped.values()), score_mode, metadata=metadata)

    def to_dataframe(self):
        """Return tabular hits; pandas is an optional dependency."""
        import pandas as pd
        return pd.DataFrame(self._rows(), columns=self._fields())

    def _fields(self):
        return ["query", "rank", "target", "score", *self.metadata.get("metadata_columns", [])]

    def _rows(self):
        return [dict(query=q, rank=h["rank"], target=h["target"], score=h["score"], **h.get("meta", {}))
                for q, hits in zip(self.query_ids, self.hits) for h in hits]

    def write_tsv(self, path):
        started = time.perf_counter()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, self._fields(), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in self._rows():
                row["score"] = f"{row['score']:.9g}"
                writer.writerow(row)
        self.timings["hit_table_write_sec"] = time.perf_counter() - started
        _save_json(path.with_suffix(path.suffix + ".json"),
                   dict(self.metadata, score_mode=self.score_mode, query_ids=self.query_ids,
                        hit_table_sha256=file_sha256(path)))
        return path

    to_tsv = write_tsv


class SearchIndex(QueryAPI):
    """A vector database with input/model provenance and resumable construction.

    Do not build into the same directory from multiple processes. Run encoding
    on a compute node; already completed and verified shards survive interruption.
    """

    SCHEMA = 1

    def __init__(self, directory, metadata, ids, embeddings, store=None):
        self.directory = Path(directory)
        self.metadata = metadata
        self.ids = ids
        self.embeddings = embeddings
        self.store = store
        if store is not None:
            from .index_store import DiskLookup
            self._columns = DiskLookup(store)
            self.target_metadata = DiskLookup(store, metadata=True)
            self.metadata_columns = metadata.get("metadata_columns", [])
        else:
            self._columns = {identifier: i for i, identifier in enumerate(ids)}
            from .index_data import read_metadata
            self.target_metadata, self.metadata_columns = read_metadata(self.directory / "metadata.tsv", ids)
        self.encoder = None

    def __len__(self):
        return len(self.ids)

    @property
    def meta(self):
        return self.metadata

    def _check_encoder(self, encoder):
        if encoder.identity != self.metadata["setup"]["encoder"]:
            raise ValueError("Query encoder does not match the index model/head/code")

    def _query_length(self, value):
        saved = self.metadata["setup"]["max_length"]
        value = saved if value is _INHERIT else _length(value)
        if value != saved:
            raise ValueError("Query max_length must match the index max_length")
        return value

    def target_block(self, start, stop):
        """Read a bounded vector buffer without retaining touched mmap pages."""
        if isinstance(self.embeddings, np.memmap):
            dimension = self.metadata["dimension"]
            with open(self.embeddings.filename, "rb") as handle:
                handle.seek(self.embeddings.offset + start * dimension * np.dtype(np.float32).itemsize)
                block = np.fromfile(handle, dtype=np.float32, count=(stop-start)*dimension)
            if block.size != (stop-start)*dimension:
                raise ValueError("Index vectors were truncated during search")
            return block.reshape(stop-start, dimension)
        return self.embeddings[start:stop]

    def get_sequences(self, ids):
        """Fetch only requested original target sequences for candidate alignment."""
        if self.store is None or not self.metadata.get("has_sequences", False):
            raise ValueError("This library has no stored sequences; provide target FASTA or rebuild the library")
        return self.store.sequences(ids)

    @classmethod
    def load(cls, directory, verify: bool = False, *, encoder=None, mmap=True, check_model=True):
        directory = Path(directory)
        canonical = (directory / "meta.json").is_file()
        metadata = json.loads((directory / ("meta.json" if canonical else "index.json")).read_text(encoding="utf-8"))
        if metadata.get("schema") != cls.SCHEMA or metadata.get("status") != "complete":
            raise ValueError("Index is incomplete or has an unsupported schema")
        store = None
        if canonical:
            from .index_store import RecordStore, DiskIDs
            for name, digest in metadata["auxiliary_sha256"].items():
                path = directory / name
                if not path.is_file() or file_sha256(path) != digest:
                    if name == "ids.txt":
                        raise ValueError("Index IDs have changed")
                    raise ValueError(f"Index IDs, metadata or Pfam annotations have changed: {name}")
            store = RecordStore(directory / "records.sqlite", metadata["count"])
            ids = DiskIDs(store)
        else:
            ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
            if len(ids) != len(set(ids)) or len(ids) != metadata["count"]:
                raise ValueError("Index ID count or uniqueness check failed")
            if _digest(ids) != metadata["ids_digest"]:
                raise ValueError("Index IDs have changed")
        vector_path = directory / ("vectors.npy" if canonical else "embeddings.npy")
        arr = np.load(vector_path, mmap_mode="r" if mmap else None, allow_pickle=False)
        if arr.shape != (len(ids), metadata["dimension"]) or arr.dtype != np.float32:
            raise ValueError("Index embeddings have the wrong shape or dtype")
        if verify and file_sha256(vector_path) != metadata["embeddings_sha256"]:
            raise ValueError("Index embedding checksum does not match")
        index = cls(directory, metadata, ids, arr, store=store)
        if not canonical and "annotations_digest" in metadata["setup"]:
            annotations_path = directory / "pfam.json"
            clans_path = directory / "pfam_clans.json"
            annotations = json.loads(annotations_path.read_text(encoding="utf-8")) if annotations_path.is_file() else None
            clans = json.loads(clans_path.read_text(encoding="utf-8")) if clans_path.is_file() else None
            if _digest(dict(metadata=index.target_metadata,pfam=annotations,clans=clans)) != metadata["setup"]["annotations_digest"]:
                raise ValueError("Index metadata or Pfam annotations have changed")
        if encoder is None and check_model:
            saved = metadata["setup"]["encoder"]
            if saved.get("backend") != "prot_t5":
                raise ValueError("Index.load requires a compatible encoder for this backend; use check_model=False only for inspection or provenance-tagged vector queries")
            from .search_encoders import T5SearchEncoder
            encoder = T5SearchEncoder.load(mode=saved["mode"], head_batch_size=saved.get("head_batch_size", 4096))
        if encoder is not None:
            index._check_encoder(encoder)
            index.encoder = encoder
        return index

    @classmethod
    def build_fasta(cls, encoder, fasta, directory, **kwargs):
        ids, seqs = read_fasta(fasta)
        return cls._build(encoder, seqs, ids, directory, **kwargs)

    @classmethod
    def build(cls, fasta, out_dir, *legacy, encoder=None, mode=None, **kwargs):
        """Build from FASTA; also accept the original four-positional-argument API."""
        if hasattr(fasta, "encode") and not isinstance(fasta, str):
            if len(legacy) != 2:
                raise TypeError("Expected build(encoder, sequences, ids, directory)")
            return cls._build(fasta, out_dir, legacy[0], legacy[1], **kwargs)
        if legacy:
            raise TypeError("Expected build(fasta, out_dir, *, encoder=None, mode='tm')")
        if encoder is None:
            from .search_encoders import T5SearchEncoder
            options = {key: kwargs.pop(key) for key in
                       ("config_path", "ckpt_path", "tm_head_path", "ec_head_path", "device", "head_batch_size") if key in kwargs}
            options["verbose"] = kwargs.get("verbose",True)
            encoder = T5SearchEncoder.load(mode=mode or "tm", **options)
        elif mode is not None and mode != encoder.identity.get("mode"):
            raise ValueError("Requested mode does not match the supplied encoder")
        kwargs.setdefault("max_length", None)
        return cls.build_fasta(encoder, fasta, out_dir, **kwargs)

    @classmethod
    def _build(cls, encoder, seqs: Sequence[str], ids: Sequence[str], directory,
              max_length=None, batch_size: int = 8, max_tokens: int = 8192,
              shard_size: int = 20000, split_length: Optional[int] = 1024,
              resume: bool = True, verbose: bool = True, metadata=None, pfam=None,
              clan_tsv=None):
        """Build without changing preprocessing inside the selected encoder.

        ``split_length=1024`` reproduces the short/long partition used for the
        final large-scale benchmark. Within each original-order shard, the
        encoder sorts by length and restores the supplied order. None disables
        the partition; this option is independent of ``max_length`` truncation.
        """
        build_started = time.perf_counter()
        ids, seqs = _records(ids, seqs)
        max_length = _length(max_length)
        if min(batch_size, max_tokens, shard_size) < 1:
            raise ValueError("Batch, token and shard limits must be positive")
        if split_length is not None and split_length < 1:
            raise ValueError("split_length must be positive or None")
        identity = json.loads(json.dumps(encoder.identity, sort_keys=True))
        from .index_data import metadata_records, pfam_records, clan_records, write_metadata
        rows, fields = metadata_records(metadata, ids)
        annotations = pfam_records(pfam, ids) if pfam is not None else None
        clans = clan_records(clan_tsv) if annotations is not None else None
        inputs = hashlib.sha256()
        for identifier, seq in zip(ids, seqs):
            inputs.update(json.dumps([identifier, seq], separators=(",", ":")).encode())
            inputs.update(b"\n")
        setup = dict(encoder=identity, input_digest=inputs.hexdigest(), count=len(ids),
                     max_length=max_length, batch_size=batch_size, max_tokens=max_tokens,
                     shard_size=shard_size, split_length=split_length)
        if metadata is not None or annotations is not None:
            setup["annotations_digest"] = _digest(dict(metadata=rows, pfam=annotations, clans=clans))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "meta.json"
        if manifest.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if meta.get("schema") != cls.SCHEMA or meta.get("setup") != setup:
                raise ValueError("Existing index uses different inputs, weights, code or encoding settings")
            if not resume:
                raise FileExistsError("Index already exists; use resume or a new output directory")
            if meta.get("status") == "complete":
                return cls.load(directory, verify=True, encoder=encoder)
        else:
            if any(directory.iterdir()):
                raise FileExistsError("New index directory must be empty")
            meta = dict(schema=cls.SCHEMA, status="building", setup=setup, shards={},
                        count=len(ids), ids_digest=_digest(ids), created=datetime.now(timezone.utc).isoformat(),
                        mode=identity.get("mode"), normalized=identity.get("mode") in ("tm", "ec"),
                        splm_tm_version="3.3.0")
            (directory / "ids.txt").write_text("".join(key + "\n" for key in ids), encoding="utf-8")
            _save_json(manifest, meta)
        write_metadata(directory / "metadata.tsv", rows, fields)
        if annotations is not None:
            _save_json(directory / "pfam.json", annotations)
            _save_json(directory / "pfam_clans.json", clans)
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
                previous = meta["shards"].get(name)
                if previous and path.is_file() and file_sha256(path) == previous["sha256"]:
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
                meta["shards"][name] = dict(count=len(selected), sha256=file_sha256(path),
                                             seconds=time.perf_counter() - started,
                                             encoding_sec=encode_sec)
                _save_json(manifest, meta)
                if verbose:
                    print(f"index {name}: {len(selected)} sequences", flush=True)
        tmp = directory / "vectors.tmp.npy"
        merge_started = time.perf_counter()
        combined = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                             shape=(len(ids), meta["dimension"]))
        for _, selected, path in planned:
            combined[selected] = np.load(path, mmap_mode="r", allow_pickle=False)
        combined.flush()
        del combined
        os.replace(tmp, directory / "vectors.npy")
        merge_sec = time.perf_counter() - merge_started
        meta["status"] = "complete"
        files = identity.get("files",{})
        backbone = getattr(getattr(encoder,"model",None),"backbone",None)
        checkpoint = getattr(backbone,"ckpt_path",None)
        meta.update(format_version=cls.SCHEMA,n=len(ids),dim=meta["dimension"],max_length=max_length,
                    backbone_sha256=files.get("checkpoint"),
                    backbone_id=Path(checkpoint).stem if checkpoint is not None else identity.get("checkpoint"),
                    head_sha256=files.get("ec_head" if identity.get("mode") == "ec" else "tm_head"))
        meta["embeddings_sha256"] = file_sha256(directory / "vectors.npy")
        meta["timings"] = dict(
            target_encoding_sec=sum(part["encoding_sec"] for part in meta["shards"].values())
            if all("encoding_sec" in part for part in meta["shards"].values()) else None,
            shard_processing_sec=sum(part["seconds"] for part in meta["shards"].values()),
            merge_write_sec=merge_sec,
            build_wall_sec_this_run=time.perf_counter() - build_started,
            note="Encoding/shard totals include resumed shards; wall time describes this invocation.")
        from .index_store import RecordStore
        RecordStore.create(directory / "records.sqlite", ids, rows, annotations, clans, sequences=seqs)
        meta["has_sequences"] = True
        meta["metadata_columns"] = fields
        meta["has_pfam"] = annotations is not None
        meta["has_clans"] = clans is not None
        meta["auxiliary_sha256"] = {name: file_sha256(directory / name) for name in
                                    ("ids.txt", "records.sqlite", "metadata.tsv", "pfam.json", "pfam_clans.json")
                                    if (directory / name).is_file()}
        meta.pop("shards")
        _save_json(manifest, meta)
        return cls.load(directory, encoder=encoder)



Index = SearchIndex

__all__ = ["Index", "SearchIndex", "SearchResult", "QueryEmbeddings", "file_sha256"]
