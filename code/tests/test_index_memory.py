"""Measure RSS growth for real disk-backed indexes, including IDs and annotations."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm.index import _save_json, file_sha256
from splm_tm.index_store import RecordStore


class IDs:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        return (f"target_{i:09d}_" + "x" * 100 for i in range(self.count))


class Annotations:
    def __init__(self, value):
        self.value = value

    def get(self, key, default=None):
        return self.value


def fixture(directory, count):
    directory.mkdir()
    ids = IDs(count)
    with (directory / "ids.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for key in ids:
            handle.write(key + "\n")
    vectors = np.lib.format.open_memmap(directory / "vectors.npy", mode="w+", dtype=np.float32, shape=(count, 64))
    for start in range(0, count, 1024):
        vectors[start:start+1024] = 0.125
    vectors.flush()
    del vectors
    RecordStore.create(directory / "records.sqlite", ids, Annotations({"description": "a" * 1024}),
                       Annotations(["PF00001"]), {})
    (directory / "pfam.json").write_text("{}", encoding="utf-8")
    auxiliary = {name: file_sha256(directory / name) for name in ("ids.txt", "records.sqlite", "pfam.json")}
    _save_json(directory / "meta.json",
               dict(schema=1, format_version=1, status="complete", count=count, n=count, dimension=64, dim=64,
                    setup=dict(encoder=dict(backend="test", mode="tm"), max_length=512), max_length=512,
                    metadata_columns=["description"], has_pfam=True, has_clans=False, ids_digest="fixture",
                    auxiliary_sha256=auxiliary, embeddings_sha256=file_sha256(directory / "vectors.npy")))


class IndexMemoryTests(unittest.TestCase):
    def test_load_and_search_memory_do_not_scale_with_ids_or_annotations(self):
        root = Path(__file__).resolve().parent / ("_tmp_" + uuid.uuid4().hex)
        root.mkdir()
        try:
            reports = []
            for count in (2_000, 120_000):
                directory = root / str(count)
                fixture(directory, count)
                completed = subprocess.run([sys.executable, str(Path(__file__).with_name("index_memory_probe.py")), str(directory)],
                                           capture_output=True, text=True, check=True, timeout=60)
                reports.append(json.loads(completed.stdout))
            small, large = reports
            # A Python dictionary containing the large fixture would exceed 120 MB.
            # Allow fixed interpreter/allocator noise, not one object per target.
            allowance = 16 * 1024 * 1024
            self.assertLess(large["load_peak_rss"] - small["load_peak_rss"], allowance, reports)
            self.assertLess(large["search_peak_rss"] - small["search_peak_rss"], allowance, reports)
            self.assertLess(large["python_peak"], 24 * 1024 * 1024, reports)
            self.assertEqual(large["max_score_block"], [1, 1024])
            self.assertEqual(large["returned"], 3)
            print("Disk-index memory measurements: " + json.dumps(reports))
        finally:
            assert root.resolve().parent == Path(__file__).resolve().parent and root.name.startswith("_tmp_")
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
