"""Public specification contracts: defaults, provenance and exact score bits."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm import Index, QueryEmbeddings
from splm_tm import cli
from splm_tm.index import _unit
from splm_tm.scoring import fixed_scores
from test_public_search_api import Encoder


class SearchSpecificationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("_tmp_" + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        assert self.root.resolve().parent == Path(__file__).resolve().parent
        assert self.root.name.startswith("_tmp_")
        shutil.rmtree(self.root)

    def build(self, vectors, mode="tm"):
        ids = ["t" + str(i) for i in range(len(vectors))]
        encoder = Encoder(dict(zip(ids, vectors)), mode=mode)
        index = Index.build(encoder, ["A"] * len(ids), ids, self.root / "index", verbose=False)
        return index, encoder

    def test_dense_and_every_partition_have_identical_score_bits_and_order(self):
        rng = np.random.default_rng(7)
        targets = rng.normal(size=(37, 129)).astype(np.float32)
        targets[9] = targets[8]
        queries = rng.normal(size=(7, 129)).astype(np.float32)
        index, encoder = self.build(targets)
        for metric in ("inner", "cosine", "euclidean"):
            left, right = (_unit(queries), _unit(targets)) if metric == "cosine" else (queries, targets)
            dense = fixed_scores(left, right, negative_l2=metric == "euclidean")
            expected_order = np.argsort(-dense, axis=1, kind="stable")[:, :12]
            for query_block in (1, 3, 7):
                for target_block in (1, 8, 37, 100):
                    with self.subTest(metric=metric, q=query_block, t=target_block):
                        path = self.root / "scores.npy"
                        result = index.search(QueryEmbeddings([f"q{i}" for i in range(7)], queries,
                                                              encoder.identity, None),
                                              metric=metric, topk=12, query_block_size=query_block,
                                              chunk_size=target_block, score_matrix=path, exact=True)
                        actual = np.load(path)
                        np.testing.assert_array_equal(actual.view(np.uint32), dense.view(np.uint32))
                        for i, row in enumerate(result.hits):
                            self.assertEqual([h["target_index"] for h in row], expected_order[i].tolist())
                            np.testing.assert_array_equal(np.array([h["score"] for h in row], dtype=np.float32).view(np.uint32),
                                                          dense[i, expected_order[i]].view(np.uint32))
        del index

    def test_load_checks_default_model_before_return_and_metadata_only_is_explicit(self):
        index, encoder = self.build([[1, 0, 0, 0]])
        meta = index.meta
        meta["setup"]["encoder"]["backend"] = "prot_t5"
        encoder.identity["backend"] = "prot_t5"
        (index.directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with patch("splm_tm.search_encoders.T5SearchEncoder.load", return_value=encoder) as loader:
            loaded = Index.load(index.directory)
            loader.assert_called_once()
            self.assertIs(loaded.encoder, encoder)
        with patch("splm_tm.search_encoders.T5SearchEncoder.load", return_value=Encoder(version="wrong")):
            with self.assertRaisesRegex(ValueError, "does not match"):
                Index.load(index.directory)
        with patch("splm_tm.search_encoders.T5SearchEncoder.load", side_effect=AssertionError("unexpected model")):
            inspected = Index.load(index.directory, check_model=False)
            self.assertIsNone(inspected.encoder)
        del index, loaded, inspected

    def test_documented_files_fields_defaults_and_tuple_results(self):
        fasta = self.root / "targets.fasta"
        fasta.write_text(">t0\n" + "A" * 700 + "\n", encoding="utf-8")
        encoder = Encoder()
        index = Index.build(fasta, self.root / "index", encoder=encoder, verbose=False)
        self.assertIsNone(encoder.calls[0][1]["max_length"])
        self.assertEqual((index.directory / "ids.txt").read_text(), "t0\n")
        self.assertEqual(np.load(index.directory / "vectors.npy").dtype, np.float32)
        for field in ("format_version", "n", "dim", "mode", "normalized", "backbone_sha256",
                      "backbone_id", "head_sha256", "max_length", "created", "splm_tm_version"):
            self.assertIn(field, index.meta)
        result = index.search({"q": "A"})
        target, score, metadata = result.hits["q"][0]
        self.assertEqual(target, "t0")
        self.assertEqual(score, result.hits[0][0]["score"])
        self.assertEqual(metadata, {})
        del index

    def test_raw_requires_explicit_normalization(self):
        index, encoder = self.build([[3, 4, 0, 0], [1, 0, 0, 0]], mode="raw")
        vectors = QueryEmbeddings(["q"], np.array([[3, 4, 0, 0]], dtype=np.float32), encoder.identity, None)
        for metric in ("inner", "euclidean"):
            with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, "explicit normalize"):
                index.search(vectors, metric=metric)
        result = index.search(vectors, normalize=True)
        self.assertEqual(result.hits["q"][0][0], "t0")
        np.testing.assert_array_equal(np.float32(result.hits["q"][0][1]),
                                      fixed_scores(_unit(vectors.vectors), _unit(np.array([[3, 4, 0, 0]], np.float32)))[0, 0])
        del index

    def test_cli_emb_alias_default_length_conflict_and_raw_flag(self):
        fasta = self.root / "targets.fasta"
        fasta.write_text(">t0\nAAA\n", encoding="utf-8")
        query = self.root / "queries.fasta"
        query.write_text(">q\nA\n", encoding="utf-8")
        encoder = Encoder(mode="ec")
        with patch("splm_tm.cli._encoder", return_value=encoder), contextlib.redirect_stdout(io.StringIO()):
            cli.main(["index", "build", "--fasta", str(fasta), "--out", str(self.root / "index"), "--emb", "emb_ec"])
            cli.main(["search", "--index", str(self.root / "index"), "--query", str(query),
                      "--out", str(self.root / "hits.tsv"), "--emb", "emb_ec"])
        self.assertIsNone(encoder.calls[-1][1]["max_length"])
        arguments = cli.parser().parse_args(["index", "build", "--fasta", str(fasta), "--out", "unused",
                                            "--mode", "tm", "--emb", "emb_ec"])
        with self.assertRaisesRegex(ValueError, "disagree"):
            cli._mode(arguments)
        self.assertEqual(cli._mode(cli.parser().parse_args(["index", "build", "--fasta", str(fasta), "--out", "unused"])), "tm")

    def test_full_length_cli_default_aliases_and_explicit_query_length_checks(self):
        fasta = self.root / "targets.fasta"
        fasta.write_text(">t0\n" + "A" * 700 + "\n", encoding="utf-8")
        query = self.root / "queries.fasta"
        query.write_text(">q\n" + "C" * 800 + "\n", encoding="utf-8")
        for label, options in (("default", []), ("full", ["--max-length", "full"]),
                               ("legacy", ["--max-length", "0"]), ("limited", ["--max-length", "512"])):
            with self.subTest(label=label):
                encoder = Encoder()
                library = self.root / label
                with patch("splm_tm.cli._encoder", return_value=encoder), contextlib.redirect_stdout(io.StringIO()):
                    cli.main(["index", "build", "--fasta", str(fasta), "--out", str(library), *options])
                    cli.main(["search", "--index", str(library), "--query", str(query),
                              "--out", str(self.root / (label + ".tsv"))])
                    expected = 512 if label == "limited" else None
                    self.assertEqual(encoder.calls[0][1]["max_length"], expected)
                    self.assertEqual(encoder.calls[-1][1]["max_length"], expected)
                    if label == "limited":
                        encoder.calls.clear()
                        for token in ("full", "0"):
                            with self.assertRaisesRegex(ValueError, "max_length"):
                                cli.main(["search", "--index", str(library), "--query", str(query),
                                          "--out", str(self.root / "mismatch.tsv"), "--max-length", token])
                        self.assertEqual(encoder.calls, [])


if __name__ == "__main__":
    unittest.main()
