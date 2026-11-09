"""CPU checks for GO retrieval, BMA scoring, export helpers and the case pipeline."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_go  # noqa: E402
import run_go  # noqa: E402


class GoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bma_annotation_filtering_and_bidirectional_average(self):
        annotations={"q":["a","b","stale"],"t":["a"],"missing":["stale"]}
        result=run_go.score_view({"q":["t","missing"]},annotations,{"a":1,"b":1},lambda a,b:1. if a==b else .2)
        # q->t=(1+.2)/2=.6; t->q=1; BMA=.8.
        self.assertEqual(result["pairs"],[dict(query="q",target="t",go_similarity="0.800000")])
        self.assertEqual(result["per_query_means"]["q"],.8)
        self.assertEqual(result["query_count"],1)

    def test_retrieval_preserves_original_partition_ties(self):
        ids=["q"]+[f"t{i}" for i in range(12)]
        emb=np.ones((13,2),dtype=np.float32)
        sim=emb[:1]@emb[1:].T
        top=np.argpartition(-sim,9,axis=1)[0,:10]
        columns=top[np.argsort(-sim[0,top])]
        result=run_go.retrieve(ids,emb,["q"],ids[1:])
        self.assertEqual(result["q"],[ids[1:][i] for i in columns])

    def test_historical_search_preserves_order_and_repeated_queries(self):
        path=self.root/"search.tsv"
        path.write_text("q.pdb\tb.pdb\t0.9\nq.pdb\ta.pdb\t0.9\nr\tc\t0.8\n")
        self.assertEqual(prepare_go.read_hits(path),{"q":["b","a"],"r":["c"]})

    def test_incomplete_or_other_data_pack_is_rejected(self):
        path=self.root/"input.txt";path.write_text("original")
        run_go.write_json(self.root/"manifest.json",dict(case="go_functional",files=["input.txt"]))
        self.assertEqual(run_go.verify_pack(self.root)["case"],"go_functional")
        path.unlink()
        with self.assertRaisesRegex(ValueError,"missing"):
            run_go.verify_pack(self.root)
        run_go.write_json(self.root/"manifest.json",dict(case="cath_s20",files=[]))
        with self.assertRaisesRegex(ValueError,"another benchmark"):
            run_go.verify_pack(self.root)

    def test_exact_comparison_rejects_small_differences(self):
        run_go.write_json(self.root/"reference.json",dict(mean=.5))
        args=argparse.Namespace(data=self.root,out=self.root/"out",skip_reference=False,embeddings=None)
        with self.assertRaises(SystemExit):
            run_go.finish(args,dict(mean=.5+1e-12),{})
        self.assertEqual(json.loads((args.out/"comparison.json").read_text())["status"],"different")

    def test_encoding_uses_go_profile(self):
        import torch
        calls=[]
        def backbone_encode(seqs,ids,**kwargs):
            calls.append(kwargs)
            return np.array([[len(seq),1] for seq in seqs],dtype=np.float32)
        model=types.SimpleNamespace(backbone=types.SimpleNamespace(encode=backbone_encode),
                                    tm_head=torch.nn.Identity(),head_device="cpu",device="cpu")
        manifest=dict(case="go_functional",profile=dict(prepare_go.PROFILE))
        (self.root/"all.fasta").write_text(">b\nAA\n>a\nA\n")
        args=argparse.Namespace(data=self.root,out=self.root/"out",config=None,checkpoint=None,
                                tm_head=None,device="cpu",embeddings=None)
        with patch.object(run_go,"load_tm_model",return_value=(model,{})):
            arrays,_=run_go.encode(args,manifest,["all.fasta"])
        self.assertEqual(arrays["ids_0"].tolist(),["b","a"])
        self.assertEqual(calls,[dict(batch_size=32,max_tokens=16384,max_length=512,verbose=True)])


if __name__=="__main__":
    unittest.main()
