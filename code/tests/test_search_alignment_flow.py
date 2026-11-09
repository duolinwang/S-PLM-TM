"""TM/EC search-to-alignment contracts using real adapters and tiny model doubles."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm import CandidateAligner, Index, SearchResult, T5SearchEncoder
from splm_tm import cli


class Backbone:
    def __init__(self):
        self.residue_calls = []

    def encode(self, sequences, ids, max_length=None, **kwargs):
        values = np.zeros((len(sequences), 1024), dtype=np.float32)
        for i, sequence in enumerate(sequences):
            sequence = sequence[:max_length]
            values[i, :3] = [sequence.count("A") + 1, sequence.count("C") + 1, len(sequence)]
        return values

    def encode_residues(self, sequences, **kwargs):
        self.residue_calls.extend(sequences)
        return [np.array([[ord(c) % 7 + 1, 2] for c in sequence], dtype=np.float32) for sequence in sequences]


class TMHead(nn.Module):
    def forward(self, values):
        result = values.clone()
        result[:, 0] += 2
        result[:, 1] *= 3
        return torch.nn.functional.normalize(result, dim=-1)


class ECHead(nn.Module):
    in_dim = 1024

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, values):
        self.inputs.append(values.detach().cpu().numpy().copy())
        result = values.clone()
        result[:, 0] *= 5
        result[:, 2] += 0.5
        return result


def encoder(mode):
    identity = dict(backend="prot_t5", mode=mode, files=dict(config="config-hash", checkpoint="backbone-hash"),
                    source=dict(backbone="code-hash"), head_batch_size=4096)
    identity["files"].update({key: "paired-" + key for key in
                               (("tm_head", "ec_head") if mode == "ec" else ("tm_head",) if mode == "tm" else ())})
    model = SimpleNamespace(backbone=Backbone(), tm_head=TMHead(), ec_head=ECHead(), device="cpu", head_device="cpu")
    return T5SearchEncoder(model, identity, mode=mode)


def engine(query, target, matrix):
    return dict(aligned_query=query, aligned_target=target, alignment_score=float(matrix[0, 0]),
                matched_pairs=[[i, i] for i in range(min(len(query), len(target)))], query_coverage=1.)


class SearchAlignmentFlowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("_tmp_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.targets = self.root / "targets.fasta"
        self.targets.write_text(">t0\nAAAAAA\n>t1\nCCCCCC\n>t2\nGGGGG\n", encoding="utf-8")
        self.queries = self.root / "queries.fasta"
        self.queries.write_text(">q\nACDE\n", encoding="utf-8")
        self.head = self.root / "align_head.pt"
        self.head.write_bytes(b"synthetic align-head fingerprint")

    def tearDown(self):
        assert self.root.resolve().parent == Path(__file__).resolve().parent and self.root.name.startswith("_tmp_")
        shutil.rmtree(self.root)

    def aligner(self, search_encoder):
        with patch("splm_tm.alignment.load_align_head", return_value=(nn.Identity(), dict(dim=2))):
            return CandidateAligner.from_search_encoder(search_encoder, self.head, engine=engine)

    def test_tm_and_ec_search_align_only_candidates_using_original_library_sequences(self):
        outputs = {}
        for mode in ("tm", "ec"):
            with self.subTest(mode=mode):
                search_encoder = encoder(mode)
                index = Index.build(self.targets, self.root / mode, encoder=search_encoder, max_length=3,
                                    metadata={"t0": {"ec": "1.2.3.4"}}, verbose=False)
                result = index.search(self.queries, topk=2)
                outputs[mode] = np.asarray(index.embeddings).copy()
                if mode == "ec":
                    raw = search_encoder.model.backbone.encode(["AAA", "CCC", "GGG"], ["t0", "t1", "t2"])
                    expected_tm = search_encoder.model.tm_head(torch.from_numpy(raw)).numpy()
                    np.testing.assert_array_equal(search_encoder.model.ec_head.inputs[0], expected_tm)
                self.assertEqual(index.get_sequences(["t0"]), {"t0": "AAAAAA"})
                with self.assertRaises(KeyError):
                    index.get_sequences(["missing"])
                # A raw-mode aligner shares the backbone even after EC retrieval.
                residue_encoder = encoder("raw")
                aligner = self.aligner(residue_encoder)
                aligned = aligner.align_candidates(result, self.queries, index, top_k=1)
                best = result.hits[0][0]
                sequence = index.get_sequences([best["target"]])[best["target"]]
                self.assertEqual(residue_encoder.model.backbone.residue_calls, ["ACDE", sequence])
                self.assertEqual(aligned.records[0]["search_score"], best["score"])
                self.assertEqual(aligned.records[0]["aligned_query_length"], 4)
                self.assertEqual(aligned.records[0]["aligned_target_length"], len(sequence))
                self.assertTrue(aligned.metadata["backbone_checked"])
                self.assertEqual(aligned.metadata["search_metadata"]["encoder"]["mode"], mode)
                self.assertEqual(aligned.to_dataframe().iloc[0]["target"], best["target"])
                del index
        self.assertFalse(np.array_equal(outputs["tm"], outputs["ec"]))

    def test_backbone_and_library_mismatch_fail_before_residue_encoding(self):
        search_encoder = encoder("tm")
        index = Index.build(self.targets, self.root / "index", encoder=search_encoder, verbose=False)
        result = index.search(self.queries, topk=1)
        residue_encoder = encoder("raw")
        residue_encoder.identity["files"]["checkpoint"] = "other-backbone"
        with self.assertRaisesRegex(ValueError, "backbone"):
            self.aligner(residue_encoder).align_candidates(result, self.queries, index)
        self.assertEqual(residue_encoder.model.backbone.residue_calls, [])
        other = Index.build(encoder("tm"), ["HH"], ["t0"], self.root / "other", verbose=False)
        with self.assertRaisesRegex(ValueError, "Target index"):
            self.aligner(encoder("raw")).align_candidates(result, self.queries, other)
        del index, other

    def test_standalone_load_uses_raw_backbone_and_separate_alignment_head(self):
        residue_encoder = encoder("raw")
        with patch("splm_tm.search_encoders.T5SearchEncoder.load", return_value=residue_encoder) as loader, \
             patch("splm_tm.alignment.load_align_head", return_value=(nn.Identity(), dict(dim=2))):
            aligner = CandidateAligner.load(align_head_path=self.head, device="cpu", engine=engine)
        self.assertEqual(loader.call_args.kwargs["mode"], "raw")
        self.assertIs(aligner.backbone, residue_encoder.model.backbone)

    def test_hit_file_roundtrip_keeps_empty_queries_provenance_and_detects_changes(self):
        result = SearchResult(["q", "empty"], [[dict(target="t0", rank=1, score=.5, meta={})], []],
                              "inner", metadata=dict(encoder=encoder("tm").identity))
        path = result.to_tsv(self.root / "hits.tsv")
        loaded = SearchResult.from_tsv(path)
        self.assertEqual(loaded.query_ids, ["q", "empty"])
        self.assertEqual(loaded.hits, result.hits)
        self.assertEqual(loaded.metadata["encoder"], result.metadata["encoder"])
        with self.assertRaisesRegex(ValueError, "score mode"):
            SearchResult.from_tsv(path, score_mode="cosine")
        path.write_text(path.read_text() + "q\t2\tt1\t0.1\n")
        with self.assertRaisesRegex(ValueError, "provenance"):
            SearchResult.from_tsv(path)

    def test_streamed_fasta_uses_selected_ids_and_rejects_duplicate_candidates(self):
        aligner = self.aligner(encoder("raw"))
        hits = SearchResult(["q"], [[dict(target="t0", rank=1, score=1.)]], "inner")
        self.targets.write_text(">unused\nBAD-SEQ\n>t0\nAA\n", encoding="utf-8")
        self.assertEqual(aligner.align_candidates(hits, self.queries, self.targets).records[0]["target_length"], 2)
        aligner.backbone.residue_calls.clear()
        self.targets.write_text(">t0\nAA\n>t0\nGG\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            aligner.align_candidates(hits, self.queries, self.targets)
        self.assertEqual(aligner.backbone.residue_calls, [])

    def test_splm_align_cli_consumes_search_output_and_empty_results_need_no_model(self):
        search_encoder = encoder("ec")
        index = Index.build(self.targets, self.root / "index", encoder=search_encoder, verbose=False)
        result = index.search(self.queries, topk=2)
        hits = result.to_tsv(self.root / "hits.tsv")
        with patch("splm_tm.alignment._clalign_engine", return_value=engine), \
             patch("splm_tm.alignment.CandidateAligner.load", return_value=self.aligner(encoder("raw"))), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["align", "--hits", str(hits), "--queries", str(self.queries),
                                       "--index", str(index.directory), "--out", str(self.root / "aligned.jsonl"), "--topk", "1"]), 0)
        rows = [json.loads(row) for row in (self.root / "aligned.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], result.hits[0][0]["target"])
        empty = SearchResult(["q"], [[]], "inner").to_tsv(self.root / "empty.tsv")
        with patch("splm_tm.alignment.CandidateAligner.load", side_effect=AssertionError("unexpected model")), \
             patch("splm_tm.alignment._clalign_engine", side_effect=AssertionError("unexpected dependency")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["align", "--hits", str(empty), "--queries", str(self.queries),
                                       "--targets", str(self.targets), "--out", str(self.root / "empty.jsonl")]), 0)
        self.assertEqual((self.root / "empty.jsonl").read_text(), "")
        del index

    def test_old_library_without_stored_sequences_can_use_target_fasta(self):
        search_encoder = encoder("tm")
        index = Index.build(self.targets, self.root / "index", encoder=search_encoder, verbose=False)
        result = index.search(self.queries, topk=1)
        index.metadata.pop("has_sequences")
        aligner = self.aligner(encoder("raw"))
        with self.assertRaisesRegex(ValueError, "provide target FASTA"):
            aligner.align_candidates(result, self.queries, index)
        self.assertEqual(aligner.backbone.residue_calls, [])
        self.assertEqual(len(aligner.align_candidates(result, self.queries, self.targets).records), 1)
        del index

    def test_align_cli_rejects_input_overwrite_before_loading_a_model(self):
        hits = SearchResult(["q"], [[dict(target="t0", rank=1, score=1., meta={})]], "inner")
        path = hits.to_tsv(self.root / "hits.tsv")
        with patch("splm_tm.alignment.CandidateAligner.load", side_effect=AssertionError("unexpected model")), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["align", "--hits", str(path), "--queries", str(self.queries),
                      "--targets", str(self.targets), "--out", str(path)])


if __name__ == "__main__":
    unittest.main()
