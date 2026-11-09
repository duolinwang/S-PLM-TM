"""CPU-only contract tests; no checkpoints, downloads or GPU are required."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import shutil
import unittest
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm.index import SearchIndex, _top_indices


class Encoder:
    def __init__(self, version="one", fail_after=None):
        self.identity = dict(backend="test", checkpoint=version, mode="tm")
        self.calls = []
        self.fail_after = fail_after

    def encode(self, seqs, ids, max_length=None, batch_size=8, max_tokens=8192):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls.append((list(ids), max_length, batch_size, max_tokens))
        return np.array([[len(seq if max_length is None else seq[:max_length]),
                          sum(ord(c) for c in (seq if max_length is None else seq[:max_length])) % 17,
                          1.0] for seq in seqs], dtype=np.float32)


class IndexTests(unittest.TestCase):
    def setUp(self):
        # Keep test artifacts inside the checkout, including on restricted Windows.
        self.root = Path(__file__).resolve().parent / ("_tmp_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.ids = ["long", "short", "middle", "last"]
        self.seqs = ["A" * 12, "C", "DEFG", "WW"]

    def tearDown(self):
        assert self.root.resolve().parent == Path(__file__).resolve().parent
        assert self.root.name.startswith("_tmp_")
        shutil.rmtree(self.root)

    def build(self, encoder=None, **kwargs):
        return SearchIndex.build(encoder or Encoder(), self.seqs, self.ids,
                                 self.root / "db", verbose=False, **kwargs)

    def search_vectors(self, index, vectors, ids, **kwargs):
        return index.search_embeddings(vectors,ids,identity=index.metadata["setup"]["encoder"],
                                       max_length=index.metadata["setup"]["max_length"],**kwargs)

    def test_restores_original_order_after_partition_and_sharding(self):
        encoder = Encoder()
        index = self.build(encoder, split_length=5, shard_size=2)
        self.assertEqual(index.ids, self.ids)
        self.assertEqual([call[0] for call in encoder.calls],
                         [["short", "middle"], ["last"], ["long"]])
        expected = Encoder().encode(self.seqs, self.ids)
        np.testing.assert_array_equal(index.embeddings, expected)
        del index
        loaded = SearchIndex.load(self.root / "db", verify=True, encoder=Encoder())
        np.testing.assert_array_equal(loaded.embeddings, expected)
        del loaded

    def test_interrupted_build_resumes_and_reuses_completed_shards(self):
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.build(Encoder(fail_after=1), split_length=5, shard_size=2)
        encoder = Encoder()
        index = self.build(encoder, split_length=5, shard_size=2)
        self.assertEqual([call[0] for call in encoder.calls], [["last"], ["long"]])
        np.testing.assert_array_equal(index.embeddings, Encoder().encode(self.seqs, self.ids))
        del index
        encoder = Encoder()
        loaded = self.build(encoder, split_length=5, shard_size=2)
        self.assertEqual(encoder.calls, [])
        del loaded

    def test_corrupt_incomplete_shard_is_recomputed(self):
        with self.assertRaises(RuntimeError):
            self.build(Encoder(fail_after=1), split_length=5, shard_size=2)
        (self.root / "db/shards/short-00000.npy").write_bytes(b"partial")
        encoder = Encoder()
        index = self.build(encoder, split_length=5, shard_size=2)
        self.assertEqual(len(encoder.calls), 3)
        del index

    def test_rejects_changed_model_input_and_settings(self):
        index = self.build(max_length=512)
        del index
        with self.assertRaisesRegex(ValueError, "different"):
            self.build(Encoder("different"), max_length=512)
        with self.assertRaisesRegex(ValueError, "different"):
            self.build(max_length=None)
        with self.assertRaisesRegex(ValueError, "different"):
            SearchIndex.build(Encoder(), ["Y"] + self.seqs[1:], self.ids,
                              self.root / "db", max_length=512)

    def test_length_option_changes_actual_encoding(self):
        for length, expected in ((None, 700), (0, 700), (512, 512), (1024, 700), (100, 100)):
            index = SearchIndex.build(Encoder(), ["A" * 700], ["x"], self.root / str(length),
                                      max_length=length, verbose=False)
            self.assertEqual(index.embeddings[0, 0], expected)
            del index

    def test_query_encoder_must_match_and_queries_are_serial_by_default(self):
        index = self.build()
        with self.assertRaisesRegex(ValueError, "does not match"):
            index.search(Encoder("wrong"), ["A"], ["q"])
        query_encoder = Encoder()
        index.search(query_encoder, ["AA", "CCC"], ["q1", "q2"])
        self.assertEqual([c[0] for c in query_encoder.calls], [["q1"], ["q2"]])
        self.assertTrue(all(c[2] == 1 for c in query_encoder.calls))
        del index

    def test_stable_topk_boundary_ties_and_exclusions(self):
        rng = np.random.default_rng(7)
        for n in (1, 3, 51):
            for k in (1, 2, 10, 100):
                scores = rng.integers(-2, 3, n).astype(np.float32)
                scores[0] = -np.inf
                valid = np.flatnonzero(np.isfinite(scores))
                expected = valid[np.argsort(-scores[valid], kind="stable")][:k]
                np.testing.assert_array_equal(_top_indices(scores, k), expected)

    def test_historical_scoring_is_not_part_of_public_api(self):
        index = self.build()
        with self.assertRaisesRegex(ValueError, "historical benchmark"):
            self.search_vectors(index, np.array([[1, 2, 3]], dtype=np.float32), ["q"], score_mode="benchmark_tm")
        self.assertFalse(hasattr(index, "benchmark_scoring"))
        del index

    def test_cosine_scores_and_negative_l2(self):
        index = self.build()
        q = np.array([[2, 3, 1]], dtype=np.float32)
        cosine = self.search_vectors(index, q, ["q"], top_k=4, score_mode="cosine")
        for hit in cosine.hits[0]:
            target = index.embeddings[hit["target_index"]]
            expected = np.dot(q[0], target) / (np.linalg.norm(q[0]) * np.linalg.norm(target))
            self.assertAlmostEqual(hit["score"], float(expected), places=6)
        euclidean = self.search_vectors(index, q, ["q"], top_k=4, score_mode="negative_l2", normalize=False)
        for hit in euclidean.hits[0]:
            expected = -np.linalg.norm(q[0] - index.embeddings[hit["target_index"]])
            self.assertAlmostEqual(hit["score"], float(expected), places=5)
        del index

    def test_validation_and_checksum_failure(self):
        with self.assertRaises(ValueError):
            SearchIndex.build(Encoder(), ["A", "B"], ["dup", "dup"], self.root / "bad")
        with self.assertRaises(ValueError):
            self.build(max_length=-1)
        index = self.build()
        del index
        ids = self.root / "db/ids.txt"
        ids.write_text("\n".join(reversed(self.ids)) + "\n")
        with self.assertRaisesRegex(ValueError, "IDs have changed"):
            SearchIndex.load(self.root / "db")

    def test_stage_timings_and_warmup_are_reported_separately(self):
        index = self.build()
        self.assertGreaterEqual(index.metadata["timings"]["target_encoding_sec"], 0)
        encoder = Encoder()
        result = index.search(encoder, ["AA", "CCC"], ["q1", "q2"], warmup=True,
                              score_matrix=self.root / "timed_scores.npy")
        self.assertEqual([c[0] for c in encoder.calls], [["q1"], ["q1"], ["q2"]])
        result.write_tsv(self.root / "timed_hits.tsv")
        for key in ("warmup_sec", "query_encoding_sec", "scoring_sec", "normalization_sec",
                    "score_transform_sec", "score_matrix_write_sec", "ranking_sec", "hit_table_write_sec"):
            self.assertGreaterEqual(result.timings[key], 0)
        self.assertEqual(result.timings["online_wall_sec"],
                         result.timings["query_encoding_sec"] + result.timings["search_wall_sec"])
        del index



if __name__ == "__main__":
    unittest.main()
