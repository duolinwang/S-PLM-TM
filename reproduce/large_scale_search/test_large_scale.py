"""CPU checks for the frozen large-scale index: ordering, scores and resume."""
from pathlib import Path
from contextlib import redirect_stderr
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_search import FrozenSearchIndex  # noqa: E402
import prepare_data  # noqa: E402
import prepare_data_public  # noqa: E402
import score_from_embeddings as sfe  # noqa: E402


class LargeScaleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_rejects_existing_data_and_source_tree_before_reading_inputs(self):
        source = self.root / "source"
        source.mkdir()
        pack = self.root / "pack"
        pack.mkdir()
        original = {name: b"existing frozen data" for name in
                    ("truth.jsonl.gz", "reference_per_query.tsv", "target.fasta", "provenance.json")}
        for name, value in original.items():
            (pack / name).write_bytes(value)
        file_output = self.root / "file"
        file_output.write_bytes(b"keep this file")
        for output in (pack, file_output, source, source / "new_pack"):
            with self.subTest(output=output), patch.object(prepare_data, "fasta_ids") as read:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    prepare_data.main(["--source-work", str(source), "--reference-tag", "test__full",
                                       "--out", str(output)])
                self.assertEqual(error.exception.code, 2)
                read.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in pack.iterdir()}, original)
        self.assertEqual(file_output.read_bytes(), b"keep this file")
        self.assertEqual(list(source.iterdir()), [])

    def _public_source(self, n_db=12, n_q=2):
        """A miniature stand-in for the published large_scale/ directory.

        At least ten targets: the published protocol requires ten low-identity
        positives per query, and the builder enforces it.
        """
        import csv
        import gzip
        hf = self.root / "hf"
        (hf / "library").mkdir(parents=True)
        (hf / "queries/scope").mkdir(parents=True)
        (hf / "select").mkdir(parents=True)
        (hf / "results").mkdir(parents=True)
        db_ids = ["t%d" % i for i in range(n_db)]
        q_ids = ["q%d" % i for i in range(n_q)]
        (hf / "library/protein.fasta").write_text(
            "".join(">%s desc\nAC\n" % i for i in db_ids))
        (hf / "queries/scope/protein.fasta").write_text(
            "".join(">%s\nAD\n" % i for i in q_ids))
        # Every target is a positive of every query, all below 30% identity.
        truth = np.full((n_q, n_db), 0.9, dtype=np.float32)
        np.save(hf / "queries/scope/ss_mat.npy", truth)
        with gzip.open(hf / "select/positive_identity.tsv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["query", "target", "identity"])
            for q in q_ids:
                for t in db_ids:
                    writer.writerow([q, t, "12.5"])
        with (hf / "results/k1000_per_query.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["method", "group", "query", "setting", "n_pos", "AP", "P@1", "P@10"])
            for q in q_ids:
                for setting in ("A", "B"):
                    writer.writerow(["S-PLM-TM", "scope", q, setting, n_db, "0.5", "1.0", "0.5"])
                writer.writerow(["S-PLM-TM", "scope", q, "bin_0-10", n_db, "0.5", "1.0", "0.5"])
                writer.writerow(["TM-Vec", "scope", q, "A", n_db, "0.4", "1.0", "0.4"])
        return hf, db_ids, q_ids

    def test_public_pack_matches_the_pack_contract(self):
        import gzip
        import json
        hf, db_ids, q_ids = self._public_source()
        out = self.root / "pack"
        with patch.multiple(prepare_data_public, N_TARGET=len(db_ids), N_QUERY=len(q_ids)):
            prepare_data_public.main(["--hf-dir", str(hf), "--out", str(out)])
        manifest = json.loads((out / "provenance.json").read_text())
        self.assertEqual(manifest["query_ids"], q_ids)
        self.assertEqual((manifest["queries"], manifest["targets"]), (len(q_ids), len(db_ids)))
        self.assertEqual(manifest["reference_method"], "S-PLM-TM")
        # run_full.py reads reference_model; it must name the released files.
        self.assertIn("splm_tm_backbone.pth", manifest["reference_model"]["checkpoint"])
        for name in ("target.fasta", "query.fasta", "truth.jsonl.gz", "reference_per_query.tsv"):
            self.assertTrue((out / name).is_file(), name)
        with gzip.open(out / "truth.jsonl.gz", "rt") as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual([r["query"] for r in records], q_ids)
        self.assertEqual(records[0]["positive"], list(range(len(db_ids))))
        self.assertEqual(records[0]["low_identity"], list(range(len(db_ids))))
        self.assertIsNone(records[0]["self_column"])
        # Only the A/B rows of the requested method are copied.
        rows = (out / "reference_per_query.tsv").read_text().splitlines()
        self.assertEqual(len(rows), 1 + len(q_ids) * 2)
        self.assertNotIn("TM-Vec", "\n".join(rows))
        self.assertNotIn("bin_0-10", "\n".join(rows))

    def test_public_pack_reports_a_wrong_method_name(self):
        hf, db_ids, q_ids = self._public_source()
        with patch.multiple(prepare_data_public, N_TARGET=len(db_ids), N_QUERY=len(q_ids)):
            with self.assertRaisesRegex(ValueError, "method_labels.tsv"):
                prepare_data_public.main(["--hf-dir", str(hf), "--out", str(self.root / "a"),
                                          "--method", "SPLM full [some__full]"])

    def _embedding_dir(self, db_ids, q_ids, dimension=4):
        rng = np.random.default_rng(1)
        directory = self.root / "emb"
        directory.mkdir()
        unit = lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)
        db = unit(rng.standard_normal((len(db_ids), dimension)))
        queries = unit(rng.standard_normal((len(q_ids), dimension)))
        np.save(directory / "db_emb.npy", db)
        np.save(directory / "query_scope.npy", queries)
        (directory / "db_ids.txt").write_text("\n".join(db_ids) + "\n")
        (directory / "query_ids.txt").write_text("\n".join(q_ids) + "\n")
        return directory, db, queries

    def test_scoring_from_embeddings_matches_the_frozen_definition(self):
        import json
        hf, db_ids, q_ids = self._public_source(n_db=12, n_q=3)
        pack = self.root / "pack"
        with patch.multiple(prepare_data_public, N_TARGET=len(db_ids), N_QUERY=len(q_ids)):
            prepare_data_public.main(["--hf-dir", str(hf), "--out", str(pack)])
        directory, db, queries = self._embedding_dir(db_ids, q_ids)
        work = self.root / "work"
        with patch.object(sfe, "DIMENSION", db.shape[1]):
            sfe.main(["--data", str(pack), "--embeddings", str(directory),
                      "--work", str(work), "--top-k", "2"])
        unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), np.float32(1e-12))
        expected = (((unit(queries) @ unit(db).T).astype(np.float64) + 1.) / 2.).astype(np.float32)
        np.testing.assert_array_equal(np.load(work / "scores.npy"), expected)
        sidecar = json.loads((work / "scores.npy.json").read_text())
        self.assertEqual(sidecar["score_mode"], "benchmark_tm")
        self.assertEqual((sidecar["query_ids"], sidecar["target_ids"]), (q_ids, db_ids))
        self.assertTrue((work / "hits.tsv").is_file())
        # A second run must not quietly overwrite a score matrix.
        with self.assertRaises(FileExistsError):
            with patch.object(sfe, "DIMENSION", db.shape[1]):
                sfe.main(["--data", str(pack), "--embeddings", str(directory), "--work", str(work)])

    def test_scoring_rejects_misaligned_ids_and_unnormalized_vectors(self):
        hf, db_ids, q_ids = self._public_source(n_db=12, n_q=3)
        pack = self.root / "pack"
        with patch.multiple(prepare_data_public, N_TARGET=len(db_ids), N_QUERY=len(q_ids)):
            prepare_data_public.main(["--hf-dir", str(hf), "--out", str(pack)])
        directory, db, _ = self._embedding_dir(db_ids, q_ids)
        run = lambda name: sfe.main(["--data", str(pack), "--embeddings", str(directory),
                                     "--work", str(self.root / name)])
        (directory / "db_ids.txt").write_text("\n".join(["wrong"] + db_ids[1:]) + "\n")
        with patch.object(sfe, "DIMENSION", db.shape[1]):
            with self.assertRaisesRegex(ValueError, "does not match the data pack"):
                run("w1")
        (directory / "db_ids.txt").write_text("\n".join(db_ids) + "\n")
        np.save(directory / "db_emb.npy", (db * 2).astype(np.float32))
        with patch.object(sfe, "DIMENSION", db.shape[1]):
            with self.assertRaisesRegex(ValueError, "not L2-normalized"):
                run("w2")
        # A missing id list points at the published download rather than failing opaquely.
        (directory / "query_ids.txt").unlink()
        np.save(directory / "db_emb.npy", db)
        with patch.object(sfe, "DIMENSION", db.shape[1]):
            with self.assertRaisesRegex(FileNotFoundError, "S-PLM-TM-benchmarks"):
                run("w3")

    def test_frozen_index_restores_order_and_original_scores(self):
        class Encoder:
            identity=dict(backend="test",mode="tm")
            def __init__(self): self.calls=[]
            def encode(self,seqs,ids,**kwargs):
                self.calls.append(list(ids))
                return np.array([[len(s),s.count("A"),1.] for s in seqs],dtype=np.float32)
        ids=["long","short","mid","tiny"];seqs=["A"*12,"C","DAFG","AW"]
        encoder=Encoder()
        index=FrozenSearchIndex.build(encoder,seqs,ids,self.root/"db",split_length=5,shard_size=2,verbose=False)
        self.assertEqual(encoder.calls,[["short","mid"],["tiny"],["long"]])
        self.assertEqual(index.ids,ids)
        q=np.array([[3,4,0],[1,2,3]],dtype=np.float32)
        result=index.search_embeddings(q,["q","long"],top_k=4,exclude_self=True,score_matrix=self.root/"s.npy")
        unit=lambda a:a/np.linalg.norm(a,axis=1,keepdims=True)
        expected=(((unit(q)@unit(np.asarray(index.embeddings)).T).astype(np.float64)+1.)/2.).astype(np.float32)
        np.testing.assert_array_equal(np.load(self.root/"s.npy"),expected)
        self.assertNotIn("long",[h["target"] for h in result.hits[1]])
        with self.assertRaisesRegex(ValueError,"benchmark_tm"):
            index.search_embeddings(q,["q","long"],score_mode="inner")
        del index

    def test_frozen_build_resumes_recorded_shards(self):
        class Encoder:
            identity=dict(backend="test",mode="tm")
            def __init__(self,fail_after=None): self.calls=[];self.fail_after=fail_after
            def encode(self,seqs,ids,**kwargs):
                if self.fail_after is not None and len(self.calls)>=self.fail_after:
                    raise RuntimeError("simulated interruption")
                self.calls.append(list(ids))
                return np.array([[len(s),1.] for s in seqs],dtype=np.float32)
        ids=["a","b","c","d","e"];seqs=["A","CC","DDD","EEEE","FFFFF"]
        build=lambda encoder,names=ids:FrozenSearchIndex.build(encoder,seqs,names,self.root/"db",
                                                               split_length=None,shard_size=2,verbose=False)
        with self.assertRaisesRegex(RuntimeError,"interruption"):
            build(Encoder(fail_after=1))
        encoder=Encoder()
        index=build(encoder)
        self.assertEqual(encoder.calls,[["c","d"],["e"]])
        np.testing.assert_array_equal(index.embeddings[:,0],[1,2,3,4,5])
        del index
        with self.assertRaisesRegex(ValueError,"different"):
            build(Encoder(),["a","b","c","d","x"])


if __name__=="__main__":
    unittest.main()
