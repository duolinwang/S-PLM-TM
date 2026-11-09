"""End-to-end public API contracts using synthetic vectors."""
import contextlib
import gc
import io
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from splm_tm import Index,QueryEmbeddings
from splm_tm import cli,pfam
from splm_tm.index import _unit


class Encoder:
    def __init__(self, vectors=None, mode="tm", version="one"):
        self.identity=dict(backend="test",mode=mode,version=version)
        self.vectors=vectors or {}
        self.calls=[]

    def encode(self,seqs,ids,**kwargs):
        self.calls.append((list(ids),dict(kwargs)))
        return np.array([self.vectors.get(key,[len(s),s.count("A"),1.,2.]) for key,s in zip(ids,seqs)],dtype=np.float32)


class PublicSearchTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/"code/tests"/("_tmp_"+uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        gc.collect()
        assert self.root.parent == ROOT/"code/tests" and self.root.name.startswith("_tmp_")
        shutil.rmtree(self.root)

    def build(self,values,mode="tm",**kwargs):
        ids=[f"t{i}" for i in range(len(values))]
        encoder=Encoder(dict(zip(ids,values)),mode)
        index=Index.build(encoder,["AAA"]*len(ids),ids,self.root/"index",verbose=False,**kwargs)
        return index,encoder

    def vectors(self,index,values,ids=None,length=None):
        ids=ids if ids is not None else [f"q{i}" for i in range(len(values))]
        return QueryEmbeddings(ids,np.asarray(values,dtype=np.float32),index.meta["setup"]["encoder"],length)

    def test_fasta_sequence_mapping_and_vector_inputs_share_one_entry(self):
        fasta=self.root/"db.fasta";fasta.write_text(">a\nAAA\n>b\nAA\n")
        encoder=Encoder()
        index=Index.build(fasta,self.root/"index",encoder=encoder,max_length=512,verbose=False)
        queries=self.root/"q.fasta";queries.write_text(">q\nAC\n")
        a=index.search(queries)
        b=index.search(["AC"],ids=["q"])
        c=index.search({"q":"AC"})
        d=index.search(self.vectors(index,a.query_embeddings,["q"],512))
        self.assertEqual(a.hits,b.hits);self.assertEqual(a.hits,c.hits);self.assertEqual(a.hits,d.hits)
        self.assertEqual(a.metadata["max_length"],512)
        self.assertEqual(len(index),2)
        self.assertIs(index.meta,index.metadata)
        self.assertEqual(index.meta["n"],2)
        self.assertEqual(index.meta["dim"],4)
        self.assertEqual(index.meta["max_length"],512)
        self.assertEqual(a.hits["q"],[(h["target"],h["score"],h["meta"]) for h in a.hits[0]])
        self.assertIsInstance(Index.load(self.root/"index",encoder=encoder).embeddings,np.memmap)
        self.assertNotIsInstance(Index.load(self.root/"index",mmap=False,encoder=encoder).embeddings,np.memmap)
        del index

    def test_model_and_length_fail_before_encoding(self):
        index,encoder=self.build([[1,0,0,0]],max_length=512)
        encoder.calls.clear()
        for length in (None,0,1024,False,0.0):
            with self.subTest(length=length), self.assertRaises(ValueError):
                index.search({"q":"A"},max_length=length)
        self.assertFalse(encoder.calls)
        with self.assertRaisesRegex(ValueError,"does not match"):
            Index.load(self.root/"index",encoder=Encoder(version="wrong"))
        with self.assertRaisesRegex(ValueError,"does not match"):
            index.search({"q":"A"},encoder=Encoder(version="wrong"))
        with self.assertRaisesRegex(ValueError,"identity"):
            index.search(np.ones((1,4)),ids=["q"],max_length=512)
        with self.assertRaisesRegex(ValueError,"max_length"):
            index.search(np.ones((1,4)),ids=["q"],identity=encoder.identity)
        wrong=QueryEmbeddings(["q"],np.ones((1,4)),dict(version="wrong"),512)
        with self.assertRaises(ValueError):
            index.search(wrong)
        del index

    def test_target_blocks_match_dense_scores_without_full_target_normalization(self):
        from splm_tm import retrieval
        rng=np.random.default_rng(14)
        target=rng.normal(size=(153,16)).astype(np.float32)
        q=rng.normal(size=(13,16)).astype(np.float32)
        index,_=self.build(target)
        for metric in ("inner","cosine","euclidean"):
            with self.subTest(metric=metric):
                normalized=[]
                real_unit=retrieval._unit
                def unit(a):
                    normalized.append(len(a))
                    return real_unit(a)
                path=self.root/(metric+".npy")
                with patch.object(retrieval,"_unit",side_effect=unit):
                    result=index.search(self.vectors(index,q),metric=metric,topk=17,chunk_size=19,
                                        query_block_size=5,exact=True,score_matrix=path)
                if metric == "cosine":
                    expected=_unit(q)@_unit(target).T
                    self.assertLessEqual(max(normalized),19)
                elif metric == "inner":
                    expected=q@target.T
                    self.assertEqual(normalized,[])
                else:
                    expected=-np.linalg.norm(q[:,None,:]-target[None,:,:],axis=-1)
                np.testing.assert_allclose(np.load(path),expected,rtol=2e-5,atol=2e-5)
                order=np.argsort(-expected,axis=1,kind="stable")[:,:17]
                self.assertEqual([[h["target_index"] for h in row] for row in result.hits],order.tolist())
                self.assertEqual(result.metadata["max_score_block"],[5,19])
        del index

    def test_boundary_ties_self_exclusion_singleton_and_large_topk(self):
        index,_=self.build(np.ones((31,4),dtype=np.float32))
        for chunk in (1,3,11,100):
            result=index.search(self.vectors(index,[[1,1,1,1]],["t2"]),topk=100,chunk_size=chunk,exclude_self=True)
            self.assertEqual([h["target_index"] for h in result.hits[0]],[i for i in range(31) if i != 2])
        del index
        singleton=Index.build(Encoder(),["A"],["x"],self.root/"single",verbose=False)
        result=singleton.search({"x":"A"},topk=100,exclude_self=True,chunk_size=1)
        self.assertEqual(result.hits,[[]])
        del singleton
        with self.assertRaises(ValueError):
            Index.build(Encoder(),[],[],self.root/"empty")

    def test_pfam_family_clan_and_global_fallback_match_existing_mask(self):
        target=np.arange(28,dtype=np.float32).reshape(7,4)+1
        annotations={"t0":["PF1.8"],"t1":["PF2"],"t2":["PF1"],"t4":["PF3"]}
        clans=self.root/"clans.tsv";clans.write_text("PF1\tCL1\nPF2\tCL1\nPF3\t\n")
        index,_=self.build(target,pfam=annotations,clan_tsv=clans)
        queries={"q0":["PF1.99"],"q1":[],"q2":["PF999"],"q3":["PF3"]}
        q=np.arange(16,dtype=np.float32).reshape(4,4)+1
        for mode in ("pfamfam","pfamclan"):
            qp={key:{"CL1" if mode=="pfamclan" and a.split('.')[0] in ("PF1","PF2") else a.split('.')[0]
                     for a in values} for key,values in queries.items()}
            tp={key:{"CL1" if mode=="pfamclan" and a.split('.')[0] in ("PF1","PF2") else a.split('.')[0]
                     for a in values} for key,values in annotations.items()}
            overlap=pfam.build_pfam_overlap_map(qp,tp,index.ids)
            expected=pfam.apply_pfam_mask(q@target.T,list(queries),index.ids,overlap,verbose=False)
            path=self.root/(mode+".npy")
            result=index.search(self.vectors(index,q,list(queries)),prefilter=mode,query_pfam=queries,
                                topk=100,chunk_size=2,score_matrix=path)
            np.testing.assert_array_equal(np.load(path),expected)
            for i,row in enumerate(result.hits):
                valid=np.flatnonzero(np.isfinite(expected[i]))
                order=valid[np.argsort(-expected[i,valid],kind="stable")]
                self.assertEqual([h["target_index"] for h in row],order.tolist())
            self.assertEqual(result.metadata["pfam_fallback_queries"],2)
        result=index.search(self.vectors(index,[[1,1,1,1]],["t4"]),query_pfam={"t4":["PF3"]},
                            prefilter="pfamfam",exclude_self=True)
        self.assertEqual(result.hits,[[]])
        del index

    def test_metadata_roundtrip_empty_frames_and_tampering(self):
        metadata=self.root/"metadata.tsv";metadata.write_text("id\tec\torganism\nt0\t1.2.3.4\thuman\n")
        index,_=self.build([[1,0,0,0],[0,1,0,0]],metadata=metadata)
        result=index.search(self.vectors(index,[[1,0,0,0]]),topk=2)
        self.assertEqual(result.hits["q0"][0]["meta"],dict(ec="1.2.3.4",organism="human"))
        self.assertEqual(result.hits["q0"][1]["meta"],dict(ec="",organism=""))
        path=result.to_tsv(self.root/"hits.tsv")
        self.assertEqual(path.read_text().splitlines()[0],"query\trank\ttarget\tscore\tec\torganism")
        self.assertEqual(result.to_dataframe()["ec"].tolist(),["1.2.3.4",""])
        empty=type(result)(["q0"],[[]],result.score_mode,metadata=result.metadata)
        self.assertEqual(empty.to_dataframe().columns.tolist(),["query","rank","target","score","ec","organism"])
        self.assertTrue(empty.to_dataframe().empty)
        self.assertEqual(Index.load(self.root/"index",encoder=Encoder()).target_metadata,index.target_metadata)
        del index
        (self.root/"index/metadata.tsv").write_text("id\tec\torganism\nt0\twrong\thuman\n")
        with self.assertRaisesRegex(ValueError,"annotations have changed"):
            Index.load(self.root/"index")

    def test_query_vectors_roundtrip_and_cli_without_model_loading(self):
        index,_=self.build([[1,0,0,0],[0,1,0,0]])
        vectors=self.vectors(index,[[1,0,0,0]])
        path=self.root/"queries.npz";vectors.save(path)
        with patch("splm_tm.cli._encoder",side_effect=AssertionError("unexpected model loading")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["index","info","--index",str(index.directory)]),0)
            self.assertEqual(cli.main(["search","--index",str(index.directory),"--query-vectors",str(path),
                                       "--out",str(self.root/"cli.tsv"),"--chunk-size","1"]),0)
        self.assertTrue((self.root/"cli.tsv").is_file())
        self.assertEqual(QueryEmbeddings.load(path).ids,["q0"])
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError,"provenance sidecar"):
            QueryEmbeddings.load(path)
        del index

    def test_cli_build_and_query_use_the_same_saved_model_and_length(self):
        fasta=self.root/"db.fasta";fasta.write_text(">t\nAAA\n")
        query=self.root/"query.fasta";query.write_text(">q\nAA\n")
        encoder=Encoder()
        with patch("splm_tm.cli._encoder",return_value=encoder), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["index","build","--fasta",str(fasta),"--out",str(self.root/"db"),"--max-length","512"]),0)
            self.assertEqual(cli.main(["search","--query",str(query),"--index",str(self.root/"db"),
                                       "--out",str(self.root/"hits.tsv"),"--save-query-embeddings",str(self.root/"q.npz")]),0)
        self.assertEqual(encoder.calls[-1][1]["max_length"],512)
        self.assertEqual(QueryEmbeddings.load(self.root/"q.npz").max_length,512)

    def test_search_is_always_exhaustive_and_exact_flag_is_compatible(self):
        rng=np.random.default_rng(2)
        index,_=self.build(_unit(rng.normal(size=(80,16)).astype(np.float32)),pfam={"t0":["PF1"]})
        q=self.vectors(index,np.asarray(index.embeddings[:3]).copy())
        default=index.search(q)
        self.assertTrue(default.metadata["exact"])
        self.assertEqual(index.search(q,exact=True).hits,default.hits)
        with self.assertRaisesRegex(ValueError,"always exhaustive"):
            index.search(q,exact=False)
        filtered=index.search(q,prefilter="pfamfam",query_pfam={"q0":["PF1"]})
        self.assertEqual(filtered.hits[0][0]["target"],"t0")
        self.assertIsInstance(Index.load(index.directory,encoder=Encoder()).embeddings,np.memmap)
        del index


if __name__ == "__main__":
    unittest.main()
