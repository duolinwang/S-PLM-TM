"""CPU checks for SCOPe metrics, the Pfam rule and the case pipeline."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_scope  # noqa: E402
import prepare_scope_public  # noqa: E402
import run_scope  # noqa: E402


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_soft_negatives_and_bin_denominators(self):
        (self.root/"scope_target.tsv").write_text(
            "target\tsccs\tsequence\nq\ta.1.1.1\tAA\np\ta.1.1.1\tAC\nsoft\ta.1.1.2\tAD\nneg\tb.1.1.1\tWW\n")
        for name in ("scope_fam_query.tsv","scope_supfam_query.tsv"):
            (self.root/name).write_text("query\tsccs\nq\ta.1.1.1\n")
        (self.root/"identity.tsv").write_text("query\ttarget\tidentity\nq\tq\t100\nq\tp\t5\nq\tsoft\t15\n")
        tm=np.asarray([[1,0],[.8,.2],[.9,.1],[0,1]],dtype=np.float32)
        actual=run_scope.evaluate(self.root,["q","p","soft","neg"],tm,prefilter="full")
        self.assertEqual(actual["tasks"]["family"]["summary"]["Top1"],.5)
        self.assertEqual(actual["tasks"]["family"]["summary"]["AUC-1FP"],1)
        self.assertEqual(actual["binned"]["family|0-10"]["aucs"],[1.0])
        self.assertEqual(actual["binned"]["superfamily|10-20"]["query_ids"],["q"])
        (self.root/"identity.tsv").write_text("query\ttarget\tidentity\nq\tq\t100\n")
        with self.assertRaisesRegex(ValueError,"Missing"):
            run_scope.evaluate(self.root,["q","p","soft","neg"],tm,prefilter="full")

    def test_pfam_keeps_self_and_preserves_special_fallback(self):
        ids=["q","t","n"];ix={q:i for i,q in enumerate(ids)}
        row=np.array([1.,.8,.9],dtype=np.float32)
        np.testing.assert_array_equal(run_scope.scope_pfam_row(row,"q",ids,ix,{"q":{"q"}}),row)
        filtered=run_scope.scope_pfam_row(row,"q",ids,ix,{"q":{"t"}})
        self.assertEqual(filtered[0],1.)
        self.assertEqual(filtered[1],float(row[1]))
        self.assertTrue(np.isneginf(filtered[2]))
        # Below the original .999 threshold, a singleton self does not fall back.
        below=run_scope.scope_pfam_row(np.array([.9,.8,.7]),"q",ids,ix,{})
        self.assertTrue(np.isneginf(below[1]))

    def test_pfam_preserves_original_accession_keys(self):
        (self.root/"Pfam-A.clans.tsv").write_text("PF00001\tCL001\tx\nPF00002\t\tx\n")
        run_scope.write_json(self.root/"query_pfam.json",{"q":["PF00001.2"],"plain":["PF00001"],"unassigned":["PF00002"]})
        run_scope.write_json(self.root/"target_pfam.json",{"versioned":["PF00001.2"],"clan":["CL001"],"same":["PF00002"]})
        self.assertEqual(run_scope.overlap(self.root),{"q":{"versioned"},"plain":{"clan"},"unassigned":{"same"}})

    def test_export_requires_explicit_profile(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_scope.main(["--research-repo",str(self.root/"r"),"--source-root",str(self.root/"s"),
                                "--out",str(self.root/"pack")])
        self.assertFalse((self.root/"pack").exists())

    def _public_inputs(self):
        """A miniature stand-in for the published scope/ directory."""
        hf = self.root / "hf"
        hf.mkdir()
        (hf / "scope_target.tsv").write_text(
            "target\tsccs\tsequence\nq\ta.1.1.1\tAA\np\ta.1.1.1\tAC\n")
        for name in ("scope_fam_query.tsv", "scope_supfam_query.tsv"):
            (hf / name).write_text("query\tsccs\tsequence\nq\ta.1.1.1\tAA\n")
        run_scope.write_json(hf / "query_pfam.json", {"q": ["PF00001"]})
        run_scope.write_json(hf / "target_pfam.json", {"q": ["PF00001"], "p": ["PF00001"]})
        (hf / "identity.tsv").write_text("query\ttarget\tidentity\nq\tq\t100\nq\tp\t5\n")
        (self.root / "clans.tsv").write_text("PF00001\tCL001\tx\n")
        return hf

    def test_public_pack_carries_inputs_and_no_reference(self):
        hf = self._public_inputs()
        out = self.root / "pack"
        with unittest.mock.patch.multiple(prepare_scope_public, N_TARGET=2,
                                          N_FAMILY=1, N_SUPERFAMILY=1):
            prepare_scope_public.main(["--hf-dir", str(hf), "--out", str(out),
                                       "--clans-tsv", str(self.root / "clans.tsv"),
                                       "--max-length", "0", "--batch-size", "32",
                                       "--max-tokens", "8192"])
        manifest = run_scope.verify_pack(out)
        self.assertEqual(manifest["case"], "scope_retrieval")
        self.assertEqual(manifest["prefilter"], "pfamclan")
        self.assertEqual(manifest["profile"],
                         dict(max_length=None, batch_size=32, max_tokens=8192))
        # No references: the runner must be told to skip the comparison.
        self.assertEqual(manifest["reference_vectors"], {})
        self.assertNotIn("reference.json", manifest["files"])
        self.assertFalse((out / "reference.json").exists())
        # target.fasta is derived from the target table, in its order.
        ids, seqs = run_scope.read_fasta(out / "target.fasta")
        self.assertEqual((ids, seqs), (["q", "p"], ["AA", "AC"]))

    def test_public_pack_rejects_unexpected_size_and_missing_annotations(self):
        hf = self._public_inputs()
        argv = ["--hf-dir", str(hf), "--out", str(self.root / "a"),
                "--clans-tsv", str(self.root / "clans.tsv"),
                "--max-length", "0", "--batch-size", "32", "--max-tokens", "8192"]
        # Guards against a pack that is not this published benchmark.
        with self.assertRaisesRegex(ValueError, "Unexpected SCOPe benchmark size"):
            prepare_scope_public.main(argv)
        run_scope.write_json(hf / "target_pfam.json", {"q": ["PF00001"]})
        argv[3] = str(self.root / "b")
        with unittest.mock.patch.multiple(prepare_scope_public, N_TARGET=2,
                                          N_FAMILY=1, N_SUPERFAMILY=1):
            with self.assertRaisesRegex(ValueError, "no Pfam annotation"):
                prepare_scope_public.main(argv)

    def test_public_pack_requires_profile_and_a_separate_output(self):
        hf = self._public_inputs()
        base = ["--hf-dir", str(hf), "--clans-tsv", str(self.root / "clans.tsv")]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_scope_public.main(base + ["--out", str(self.root / "c")])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_scope_public.main(base + ["--out", str(hf / "inside"), "--max-length", "0",
                                              "--batch-size", "32", "--max-tokens", "8192"])

    def test_incomplete_or_other_data_pack_is_rejected(self):
        path=self.root/"input.txt";path.write_text("original")
        run_scope.write_json(self.root/"manifest.json",dict(case="scope_retrieval",files=["input.txt"]))
        self.assertEqual(run_scope.verify_pack(self.root)["case"],"scope_retrieval")
        path.unlink()
        with self.assertRaisesRegex(ValueError,"missing"):
            run_scope.verify_pack(self.root)

    def test_exact_comparison_rejects_small_differences(self):
        run_scope.write_json(self.root/"reference.json",dict(mean=.5))
        args=argparse.Namespace(data=self.root,out=self.root/"out",skip_reference=False,embeddings=None)
        with self.assertRaises(SystemExit):
            run_scope.finish(args,dict(mean=.5+1e-12),{})
        self.assertEqual(json.loads((args.out/"comparison.json").read_text())["status"],"different")


if __name__=="__main__":
    unittest.main()
