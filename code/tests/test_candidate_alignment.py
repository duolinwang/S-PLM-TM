"""Candidate-flow contract tests, independent of model weights and CLAlign."""
from pathlib import Path
import importlib.util
import json
import shutil
import sys
import types
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm import CandidateAligner, SearchResult
from splm_tm.alignment import _clalign_engine


class Backbone:
    def __init__(self):
        self.calls = []

    def encode_residues(self, seqs, **kwargs):
        self.calls.append((list(seqs), kwargs))
        return [np.array([[ord(c) % 7 + 1, 2] for c in seq], dtype=np.float32)
                for seq in seqs]


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.backbone = Backbone()
        self.matrices = []
        self.aligner = CandidateAligner(self.backbone, torch.nn.Identity(), engine=self.engine)
        self.hits = SearchResult(["q"], [[dict(target="t2", rank=1, score=.8),
                                         dict(target="t1", rank=2, score=.7)]], "cosine")
        self.queries = dict(q="ACDE")
        self.targets = dict(t1="GG", t2="WQ", unused="HHH")

    def engine(self, q, t, matrix):
        self.matrices.append(matrix.copy())
        return dict(aligned_query=q, aligned_target=t,
                    matched_pairs=[[i, i] for i in range(min(len(q), len(t)))],
                    alignment_score=float(matrix[0, 0]), query_coverage=min(len(q), len(t)) / len(q))

    def test_encodes_only_selected_candidates_and_preserves_ranks(self):
        result = self.aligner.align_candidates(self.hits, self.queries, self.targets, top_k=1)
        self.assertEqual([call[0] for call in self.backbone.calls], [["ACDE"], ["WQ"]])
        record = result.records[0]
        self.assertEqual((record["target"], record["search_rank"], record["search_score"]), ("t2", 1, .8))
        self.assertEqual(self.matrices[0].shape, (4, 2))
        self.assertTrue(np.all(np.abs(self.matrices[0]) <= 1.000001))
        self.assertIsNone(result.metadata["gap_penalty"])

    def _kernel(self):
        """The resolved kernel, or skip where neither it nor numba is installed."""
        try:
            return _clalign_engine()
        except ImportError as exc:  # pragma: no cover - environment dependent
            self.skipTest(str(exc))

    def test_resolved_kernel_produces_a_real_global_alignment(self):
        engine = self._kernel()
        self.assertIn(engine.source,
                      ("clalign.align_core", "splm_tm._clalign_core.align_core"))
        out = engine("ACDEFG", "ACDWFG", np.eye(6, dtype=np.float32))
        self.assertEqual((out["aligned_query"], out["aligned_target"]),
                         ("ACDEFG", "ACDWFG"))
        self.assertEqual(out["matched_pairs"], [[i, i] for i in range(6)])
        self.assertAlmostEqual(out["query_coverage"], 1.0)

    def test_default_aligner_runs_without_a_supplied_engine(self):
        source = self._kernel().source
        aligner = CandidateAligner(self.backbone, torch.nn.Identity())
        result = aligner.align_candidates(self.hits, self.queries, self.targets, top_k=1)
        self.assertEqual(result.metadata["engine"], source)
        self.assertEqual(result.metadata["gap_penalty"], 0.0)
        self.assertTrue(result.metadata["global_alignment"])
        record = result.records[0]
        self.assertTrue(record["aligned_query"] and record["aligned_target"])
        self.assertEqual(len(record["aligned_query"]), len(record["aligned_target"]))

    def test_real_head_projection_precedes_normalization(self):
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[0., 1.], [3., 0.]]))
        aligner = CandidateAligner(self.backbone, head, engine=self.engine)
        aligner.align_candidates(self.hits, self.queries, self.targets, top_k=1)
        def projected(seq):
            raw = torch.tensor([[ord(c) % 7 + 1, 2] for c in seq], dtype=torch.float32)
            return torch.nn.functional.normalize(head(raw), dim=-1).detach().numpy()
        np.testing.assert_array_equal(self.matrices[0], projected("ACDE") @ projected("WQ").T)

    def test_truncation_and_zero_based_coordinates_are_explicit(self):
        result = self.aligner.align_candidates(self.hits, self.queries, self.targets, max_length=1)
        self.assertEqual(self.backbone.calls[0][0], ["A"])
        self.assertIsNone(self.backbone.calls[0][1]["max_length"])
        self.assertEqual(result.records[0]["query_length"], 4)
        self.assertEqual(result.records[0]["aligned_query_length"], 1)
        self.assertEqual(result.records[0]["matched_pairs"], [[0, 0]])
        self.assertEqual(result.metadata["max_length"], 1)

    def test_missing_ids_and_bad_sequences_fail_before_encoding(self):
        for targets in (dict(t2="WQ"), dict(t1="GG", t2="W-Q"), dict(t1="GG", t2="")):
            with self.assertRaises((KeyError, ValueError)):
                self.aligner.align_candidates(self.hits, self.queries, targets)
        self.assertFalse(self.backbone.calls)

    def test_cache_is_reused_bounded_and_clearable(self):
        self.aligner.align_candidates(self.hits, self.queries, self.targets)
        self.aligner.align_candidates(self.hits, self.queries, self.targets)
        self.assertEqual(len(self.backbone.calls), 3)
        self.aligner.clear_cache()
        self.assertEqual(self.aligner._cached_residues, 0)
        self.aligner.max_cached_residues = 2
        self.aligner.align_candidates(self.hits, self.queries, self.targets)
        self.assertLessEqual(self.aligner._cached_residues, 2)
        self.assertNotIn("ACDE", self.aligner._cache)

    def test_empty_candidates_do_not_load_clalign_or_encode(self):
        with patch("splm_tm.alignment._clalign_engine", side_effect=AssertionError("unexpected import")):
            result = CandidateAligner(self.backbone, torch.nn.Identity()).align_candidates(
                SearchResult(["q"], [[]], "inner"), {}, {})
        self.assertEqual(result.records, [])
        self.assertFalse(self.backbone.calls)

    def test_invalid_candidate_order_and_arguments_fail(self):
        for kw in (dict(top_k=0), dict(top_k=True), dict(max_length=-1), dict(max_tokens=0)):
            with self.assertRaises(ValueError):
                self.aligner.align_candidates(self.hits, self.queries, self.targets, **kw)
        reversed_hits = SearchResult(["q"], [list(reversed(self.hits.hits[0]))], "inner")
        with self.assertRaises(ValueError):
            self.aligner.align_candidates(reversed_hits, self.queries, self.targets)
        self.assertFalse(self.backbone.calls)

    def test_optional_adapter_uses_global_gap_zero_and_converts_output(self):
        calls = []
        def align_core(q, t, matrix, **kwargs):
            calls.append((q, t, matrix, kwargs))
            return types.SimpleNamespace(aln1="AC", aln2="A-", score=np.float32(.5),
                                         coverage=np.float32(.5), get_matched_res=lambda: [(0, 0)])
        module = types.ModuleType("clalign.alignment")
        module.ProteinSeq, module.align_core = str, align_core
        with patch.dict(sys.modules, {"clalign": types.ModuleType("clalign"),
                                      "clalign.alignment": module}):
            engine = _clalign_engine()
            result = engine("AC", "A", np.ones((2, 1), dtype=np.float64))
        self.assertEqual(calls[0][3], dict(gap_penalty=0.0, local=False))
        self.assertEqual(calls[0][2].dtype, np.float32)
        self.assertEqual(result["matched_pairs"], [[0, 0]])
        self.assertEqual(result["query_coverage"], .5)

    def test_hit_table_roundtrip_and_alignment_output(self):
        root = Path(__file__).resolve().parent / ("_tmp_" + uuid.uuid4().hex)
        root.mkdir()
        try:
            path = Path(__file__).resolve().parents[1] / "scripts/align_candidates.py"
            spec = importlib.util.spec_from_file_location("align_candidates_cli_test", path)
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            self.hits.metadata["metadata_columns"] = ["ec"]
            self.hits.hits[0][0]["meta"] = dict(ec="1.2.3.4")
            self.hits.write_tsv(root / "hits.tsv")
            result = cli.read_candidates(root / "hits.tsv", "cosine")
            aligned = self.aligner.align_candidates(result, self.queries, self.targets)
            aligned.write_jsonl(root / "alignment.jsonl")
            rows = [json.loads(line) for line in (root / "alignment.jsonl").read_text().splitlines()]
            meta = json.loads((root / "alignment.jsonl.meta.json").read_text())
            self.assertEqual([row["target"] for row in rows], ["t2", "t1"])
            self.assertEqual(rows[0]["target_metadata"],dict(ec="1.2.3.4"))
            self.assertEqual(meta["metadata"]["search_score_mode"], "cosine")
            self.assertEqual(meta["metadata"]["candidate_count"], 2)
            self.assertIn("alignment_wall_sec", meta["timings"])
            with self.assertRaises(ValueError):
                cli.fasta_mapping(self._duplicate_fasta(root))
        finally:
            assert root.resolve().parent == Path(__file__).resolve().parent
            assert root.name.startswith("_tmp_")
            shutil.rmtree(root)

    @staticmethod
    def _duplicate_fasta(root):
        path = root / "duplicates.fa"
        path.write_text(">t1\nAA\n>t1\nGG\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
