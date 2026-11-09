"""Disk-backed IDs, metadata and Pfam lookups with a bounded SQLite cache."""
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3


class RecordStore:
    def __init__(self, path, count):
        self.path, self.count = Path(path), count

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA cache_size=-1024")
            connection.execute("PRAGMA mmap_size=0")
            yield connection
        finally:
            connection.close()

    @classmethod
    def create(cls, path, ids, metadata, pfam, clans, sequences=None):
        path = Path(path)
        temporary = path.with_suffix(".tmp.sqlite")
        temporary.unlink(missing_ok=True)
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                CREATE TABLE records (position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                                      metadata TEXT NOT NULL);
                CREATE TABLE terms (kind TEXT, term TEXT, position INTEGER,
                                    PRIMARY KEY(kind, term, position)) WITHOUT ROWID;
                CREATE INDEX terms_position ON terms(kind, position);
                CREATE TABLE clans (family TEXT PRIMARY KEY, clan TEXT) WITHOUT ROWID;
                CREATE TABLE sequences (id TEXT PRIMARY KEY, sequence TEXT NOT NULL) WITHOUT ROWID;
            """)
            db.executemany("INSERT INTO records VALUES (?,?,?)",
                           ((i, key, json.dumps(metadata.get(key, {}))) for i, key in enumerate(ids)))
            db.executemany("INSERT INTO clans VALUES (?,?)", (clans or {}).items())
            if sequences is not None:
                db.executemany("INSERT INTO sequences VALUES (?,?)", zip(ids, sequences, strict=True))
            for i, key in enumerate(ids):
                families = (pfam or {}).get(key, ())
                for kind, terms in (("pfamfam", families),
                                    ("pfamclan", {(clans or {}).get(f, f) for f in families})):
                    db.executemany("INSERT INTO terms VALUES (?,?,?)", ((kind, t, i) for t in terms))
        temporary.replace(path)
        return cls(path, len(ids))

    def sequences(self, ids):
        result = {}
        with self.connect() as db:
            for key in dict.fromkeys(ids):
                row = db.execute("SELECT sequence FROM sequences WHERE id=?", (key,)).fetchone()
                if row is None:
                    raise KeyError(f"Missing candidate sequence: {key}")
                result[key] = row[0]
        return result

    def row(self, position):
        with self.connect() as db:
            row = db.execute("SELECT id, metadata FROM records WHERE position=?", (int(position),)).fetchone()
        if row is None:
            raise IndexError(position)
        return row[0], json.loads(row[1])

    def convert_terms(self, families, mode):
        if mode == "pfamfam":
            return set(families)
        with self.connect() as db:
            return {row[0] if (row := db.execute("SELECT clan FROM clans WHERE family=?", (f,)).fetchone())
                    else f for f in families}

    def has_terms(self, terms, mode):
        with self.connect() as db:
            return any(db.execute("SELECT 1 FROM terms WHERE kind=? AND term=? LIMIT 1",
                                  (mode, term)).fetchone() for term in terms)

    def block_terms(self, start, stop, mode):
        with self.connect() as db:
            yield from db.execute("SELECT term, position FROM terms WHERE kind=? AND position>=? AND position<?",
                                  (mode, start, stop))


class DiskIDs(Sequence):
    def __init__(self, store):
        self.store = store

    def __len__(self):
        return self.store.count

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            with self.store.connect() as db:
                return [r[0] for r in db.execute("SELECT id FROM records WHERE position>=? AND position<? ORDER BY position",
                                               (start, stop))]
        key = int(key)
        if key < 0:
            key += len(self)
        return self.store.row(key)[0]

    def __iter__(self):
        with self.store.connect() as db:
            for row in db.execute("SELECT id FROM records ORDER BY position"):
                yield row[0]

    def __eq__(self, other):
        return isinstance(other, Sequence) and len(self) == len(other) and all(a == b for a, b in zip(self, other))


class DiskLookup(Mapping):
    def __init__(self, store, metadata=False):
        self.store, self.metadata = store, metadata

    def __len__(self):
        return self.store.count

    def __iter__(self):
        return iter(DiskIDs(self.store))

    def __getitem__(self, key):
        field = "metadata" if self.metadata else "position"
        with self.store.connect() as db:
            row = db.execute(f"SELECT {field} FROM records WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0]) if self.metadata else row[0]
