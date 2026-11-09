"""CPU checks for CATH-S20 metrics, export guards and the case pipeline."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_cath  # noqa: E402
import run_cath  # noqa: E402


class CathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_excludes_singletons_and_uses_original_auc_endpoint(self):
        ids=["a","b","c"]
        labels={"a":["1","1","1","1"],"b":["1","1","1","1"],"c":["2","1","1","1"]}
        tm=np.array([[1,0],[.99,.01],[0,1]],dtype=np.float32)
        actual=run_cath.evaluate(ids,tm,labels,chunk=2)
        for level,value in actual["summary"].items():
            self.assertEqual(value["f1max"],1.)
            self.assertEqual(value["queries_with_positives"],2)
            # Original trapz x=[0,1/2], not a mean or integration to x=1.
            self.assertEqual(value["auc_sensitivity_1fp"],.5)
        self.assertIsNone(actual["per_query"][2]["levels"]["Topology"])

    def test_f1_is_full_depth_even_when_sensitivity_is_shortlisted(self):
        ids=["q","neg","pos"]
        labels={"q":["1"]*4,"pos":["1"]*4,"neg":["2"]*4}
        tm=np.array([[1,0],[1,0],[.5,.5]],dtype=np.float32)
        q=run_cath.evaluate(ids,tm,labels,topk=1)["per_query"][0]["levels"]["Class"]
        self.assertEqual(q["f1max"],2/3)
        self.assertEqual(q["sensitivity_1fp"],0.)

    def test_original_summaries_reject_changed_metric_or_count(self):
        f1=self.root/"f1.txt";auc=self.root/"auc.txt"
        f1.write_text("Class  mean F1-max = 0.5927   (n=15043)\n")
        auc.write_text("Class AUC-sens1FP = 0.0135\n")
        value=dict(f1max=.592723,queries_with_positives=15043,auc_sensitivity_1fp=.013512)
        prepare_cath.check_cath_summaries(dict(summary={"Class":value}),f1,auc)
        with self.assertRaisesRegex(ValueError,"differ"):
            prepare_cath.check_cath_summaries(dict(summary={"Class":dict(value,f1max=.6001)}),f1,auc)
        with self.assertRaisesRegex(ValueError,"differ"):
            prepare_cath.check_cath_summaries(dict(summary={"Class":dict(value,queries_with_positives=15042)}),f1,auc)

    def test_export_refuses_nonempty_output(self):
        out=self.root/"pack";out.mkdir();(out/"x").write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_cath.main(["--source-root",str(self.root/"source"),"--out",str(out)])
        self.assertEqual((out/"x").read_text(),"keep")

    def test_incomplete_or_other_data_pack_is_rejected(self):
        path=self.root/"input.txt";path.write_text("original")
        run_cath.write_json(self.root/"manifest.json",dict(case="cath_s20",files=["input.txt"]))
        self.assertEqual(run_cath.verify_pack(self.root)["case"],"cath_s20")
        path.unlink()
        with self.assertRaisesRegex(ValueError,"missing"):
            run_cath.verify_pack(self.root)
        run_cath.write_json(self.root/"manifest.json",dict(case="go_functional",files=[]))
        with self.assertRaisesRegex(ValueError,"another benchmark"):
            run_cath.verify_pack(self.root)

    def test_exact_comparison_rejects_small_differences_and_reports_failure(self):
        self.assertTrue(run_cath.differences({"x":.5+1e-12},{"x":.5}))
        run_cath.write_json(self.root/"reference.json",dict(per_query=[dict(ap=.5)]))
        args=argparse.Namespace(data=self.root,out=self.root/"out",skip_reference=False,embeddings=None)
        with self.assertRaises(SystemExit):
            run_cath.finish(args,dict(per_query=[dict(ap=.5+1e-12)]),{})
        report=json.loads((args.out/"comparison.json").read_text())
        self.assertEqual(report["status"],"different")
        self.assertTrue(report["differences"])

    def test_encoding_pins_profile_and_cached_provenance(self):
        import torch
        calls=[]
        def backbone_encode(seqs,ids,**kwargs):
            calls.append((seqs,ids,kwargs))
            return np.array([[len(seq),1] for seq in seqs],dtype=np.float32)
        model=types.SimpleNamespace(backbone=types.SimpleNamespace(encode=backbone_encode),
                                    tm_head=torch.nn.Identity(),head_device="cpu",device="cpu")
        loaded=(model,dict(config="c",checkpoint="w",tm_head="h"))
        manifest=dict(case="cath_s20",profile=dict(prepare_cath.PROFILE))
        (self.root/"input.fasta").write_text(">second\nAA\n>first\nA\n")
        args=argparse.Namespace(data=self.root,out=self.root/"out",config=None,checkpoint=None,
                                tm_head=None,device="cpu",embeddings=None)
        with patch.object(run_cath,"load_tm_model",return_value=loaded):
            arrays,provenance=run_cath.encode(args,manifest,["input.fasta"])
        self.assertEqual(arrays["ids_0"].tolist(),["second","first"])
        self.assertEqual(calls[0][2],dict(batch_size=64,max_tokens=16384,max_length=512,verbose=True))
        self.assertIn("runtime",provenance)
        args.embeddings=args.out/"embeddings.npz"
        restored,_=run_cath.encode(args,manifest,["input.fasta"])
        np.testing.assert_array_equal(restored["tm_0"],arrays["tm_0"])
        manifest["profile"]["max_length"]=1024
        with self.assertRaisesRegex(ValueError,"profile"):
            run_cath.encode(args,manifest,["input.fasta"])


if __name__=="__main__":
    unittest.main()
