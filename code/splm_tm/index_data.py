"""Portable metadata and Pfam annotations for persistent sequence indexes."""
import csv
import json
from pathlib import Path


RESERVED = {"id", "query", "rank", "target", "score", "target_index", "meta"}


def read_metadata(path, ids):
    if not Path(path).is_file():
        return {}, []
    return metadata_records(path, ids)


def metadata_records(value, ids):
    if value is None:
        return {}, []
    if isinstance(value, (str, Path)):
        with Path(value).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or reader.fieldnames[0] != "id" or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("Metadata TSV must have unique columns beginning with 'id'")
            pairs = [(row.pop("id"), row) for row in reader]
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate metadata IDs")
        value = dict(pairs)
    allowed = set(ids)
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Metadata must map index IDs to column dictionaries")
    columns = set()
    for row in value.values():
        if not isinstance(row, dict) or any(not isinstance(k, str) or not k or k in RESERVED for k in row):
            raise ValueError("Metadata has invalid or reserved column names")
        columns.update(row)
    fields = sorted(columns)
    if not fields:
        return {}, []
    rows = {key: {f: str(row.get(f, "")) for f in fields} for key, row in value.items()}
    return rows, fields


def write_metadata(path, rows, fields):
    if not fields:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ["id", *fields], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(dict(id=key, **value) for key, value in rows.items())


def pfam_records(value, ids):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) - set(ids):
        raise ValueError("Pfam annotations must map supplied IDs to accession lists")
    result = {}
    for key, accessions in value.items():
        if not isinstance(accessions, (list, tuple, set)) or any(not isinstance(a, str) or not a for a in accessions):
            raise ValueError("Pfam accessions must be nonempty strings")
        result[key] = sorted({a.split(".")[0] for a in accessions})
    return result


def clan_records(path=None):
    from .pfam import DEFAULT_CLAN_TSV, load_clan_map
    explicit = path is not None
    path = Path(path) if path is not None else DEFAULT_CLAN_TSV
    if not path.is_file() and not explicit:
        from importlib.metadata import PackageNotFoundError, distribution
        try:
            package = distribution("splm-tm")
            for entry in package.files or ():
                if entry.name == "Pfam-A.clans.tsv":
                    path = Path(package.locate_file(entry))
                    break
        except PackageNotFoundError:
            pass
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(path)
        return None
    return load_clan_map(path)


class PfamFilter:
    """Mask blocks after scoring, without materializing a Q-by-N candidate map."""
    def __init__(self, index, query_ids, query_pfam, mode):
        if mode not in ("pfamfam", "pfamclan"):
            raise ValueError("prefilter must be None, 'full', 'pfamfam' or 'pfamclan'")
        path = index.directory / "pfam.json"
        if not path.is_file():
            raise ValueError("This index has no Pfam annotations; rebuild with pfam= or --pfam-json")
        if query_pfam is None:
            raise ValueError("Pfam filtering requires query_pfam annotations")
        queries = pfam_records(query_pfam, query_ids)
        self.store = index.store
        self.mode = mode
        if self.store is not None:
            if mode == "pfamclan" and not index.meta.get("has_clans"):
                raise ValueError("Pfam clan filtering requires a clan table when building the index")
            self.queries = {key: self.store.convert_terms(values, mode) for key, values in queries.items()}
            self.fallback = {key: not self.store.has_terms(self.queries.get(key, ()), mode) for key in query_ids}
            return
        targets = pfam_records(path, index.ids)
        clans = {}
        if mode == "pfamclan":
            clans = json.loads((index.directory / "pfam_clans.json").read_text(encoding="utf-8"))
            if clans is None:
                raise ValueError("Pfam clan filtering requires a clan table when building the index")
        convert = lambda values: {clans.get(a, a) for a in values}
        self.targets = {key: convert(values) for key, values in targets.items()}
        self.queries = {key: convert(values) for key, values in queries.items()}
        vocabulary = {term for values in self.targets.values() for term in values}
        self.fallback = {key: not bool(self.queries.get(key, set()) & vocabulary) for key in query_ids}

    def apply(self, scores, query_ids, target_ids, offset=None):
        import numpy as np
        inverted = {}
        if self.store is not None:
            for term, column in self.store.block_terms(offset, offset+len(target_ids), self.mode):
                inverted.setdefault(term, []).append(column-offset)
        else:
            for j,target in enumerate(target_ids):
                for term in self.targets.get(target, ()):
                    inverted.setdefault(term, []).append(j)
        for i, key in enumerate(query_ids):
            if self.fallback[key]:
                continue
            keep = np.zeros(len(target_ids),dtype=bool)
            for term in self.queries[key]:
                keep[inverted.get(term, [])] = True
            scores[i, ~keep] = -np.inf
