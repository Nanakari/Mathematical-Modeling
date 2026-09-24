import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from aligned_dataset import AlignedDataset
from pipeline import MaskedStandardizer, PreparedDataset, make_plan, apply_plan, nested_intervals
from scenarios import build_library, load_scenario
from torch_training import TokenBaseline, train_epoch, predict, save_checkpoint, restore_checkpoint


def fixture(n=7, split="train", labeled=True):
    rng = np.random.default_rng(7)
    tokens = np.zeros((n, 3, 50))
    tokens[:, 0, :12] = np.arange(1, 13)
    tokens[:, 1, :12] = 1
    a, v = np.zeros((n, 50, 74)), np.zeros((n, 50, 35))
    a[:, :12] = rng.normal(size=(n,12,74))
    v[:, :12] = rng.normal(size=(n,12,35))
    v[0] = 0  # Existing entirely unavailable modality.
    return AlignedDataset({"id": [f"{split}_{i}" for i in range(n)], "text_bert": tokens,
        "audio": a, "vision": v, "classification_labels": np.arange(n)%3,
        "regression_labels": (np.arange(n)%3-1).astype(float)}, labeled=labeled, source=f"attachment2/{split}")


class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.train = fixture()
        self.norm = MaskedStandardizer().fit(self.train, split="train")

    def test_train_only_stats_and_serialization(self):
        expected = self.train.part["audio"][:,:12].reshape(-1,74).mean(0)
        np.testing.assert_allclose(self.norm.stats["audio"]["mean"], expected, atol=1e-7)
        with self.assertRaises(ValueError):
            MaskedStandardizer().fit(fixture(split="valid"), split="train")
        with self.assertRaises(ValueError):
            self.norm.fit(self.train, split="train")
        x = self.norm.transform(self.train[0])
        self.assertTrue((x["audio"][12:] == 0).all())
        self.assertTrue((x["vision"] == 0).all())
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/"norm.json"
            self.norm.save(file)
            reload = MaskedStandardizer.load(file)
            np.testing.assert_array_equal(x["audio"], reload.transform(self.train[0])["audio"])

    def test_nested_intervals_and_modality_isolation(self):
        s = self.train[1]
        for pos in ("start", "middle", "end", "random"):
            spans = nested_intervals(s, "text", [.1,.3,.5], pos, 17)
            prev = None
            for span in spans.values():
                if prev:
                    self.assertLessEqual(span[0],prev[0]); self.assertGreaterEqual(span[1],prev[1])
                prev = span
        plan = make_plan(s, ("text",), .5, "middle", 17)
        x = apply_plan(s, plan)
        np.testing.assert_array_equal(x["extent_mask"], s["extent_mask"])
        np.testing.assert_array_equal(x["audio_observed"], s["audio_observed"])
        np.testing.assert_array_equal(x["audio"], s["audio"])
        self.assertTrue((x["input_ids"][x["text_missing"]] == 0).all())
        self.assertTrue((x["attention_mask"][x["text_missing"]] == 0).all())
        self.assertTrue((s["input_ids"][:12] != 0).all())

    def test_library_reuse_and_wrong_sample(self):
        ds = fixture(2,split="valid")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"scenarios.jsonl.gz"
            meta=build_library(ds,path,seeds=(17,))
            self.assertEqual(meta["record_count"],170)
            plans=load_scenario(path,"17|random|text+audio+vision|0.3")
            a=apply_plan(ds[0],plans[ds[0]["sample_id"]])
            b=apply_plan(ds[0],plans[ds[0]["sample_id"]])
            np.testing.assert_array_equal(a["input_ids"],b["input_ids"])
            with self.assertRaises(ValueError):
                apply_plan(ds[1],plans[ds[0]["sample_id"]])

    def test_loader_shapes_last_batch_and_unlabeled(self):
        ds=PreparedDataset(self.train,self.norm)
        batches=list(DataLoader(ds,batch_size=3))
        self.assertEqual([len(b["sample_id"]) for b in batches],[3,3,1])
        self.assertEqual(batches[0]["input_ids"].dtype,torch.int64)
        self.assertEqual(batches[0]["audio"].dtype,torch.float32)
        self.assertEqual(tuple(batches[0]["audio"].shape),(3,50,74))
        unlabeled=PreparedDataset(fixture(2,split="special",labeled=False),self.norm)
        self.assertNotIn("class_label",next(iter(DataLoader(unlabeled,batch_size=2))))
        self.assertEqual(len(predict(TokenBaseline(vocab_size=32),unlabeled)),2)

    def test_masked_input_invariance_and_finite_gradients(self):
        ds=PreparedDataset(self.train,self.norm,augment=True,seed=17)
        b=next(iter(DataLoader(ds,batch_size=3)))
        model=TokenBaseline(vocab_size=32)
        before=model(b)
        altered={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in b.items()}
        mask=~(b["attention_mask"].bool() & b["text_pool_mask"])
        altered["input_ids"][mask]=999999  # Must be masked before embedding lookup.
        for m in ("audio","vision"):
            altered[m][~b[m+"_observed"]]=float('nan')
        after=model(altered)
        for a,z in zip(before,after): torch.testing.assert_close(a,z,rtol=0,atol=0)
        for m in ("text_observed","text_pool_mask","audio_observed","vision_observed"):
            altered[m].zero_()
        logits,r=model(altered)
        (logits.sum()+r.sum()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_resume_matches_uninterrupted(self):
        ds=PreparedDataset(self.train,self.norm,augment=True,seed=17)
        model=TokenBaseline(vocab_size=32)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        log=train_epoch(model,ds,opt,epoch=0,seed=88,batch_size=3)
        self.assertEqual(log["sample_count"],7)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"state.pt"
            save_checkpoint(path,model,opt,1,seed=88,preprocessing_sha256="test")
            train_epoch(model,ds,opt,epoch=1,seed=88,batch_size=3)
            other=TokenBaseline(vocab_size=32)
            other_opt=torch.optim.Adam(other.parameters(),lr=.001)
            epoch,seed=restore_checkpoint(path,other,other_opt,preprocessing_sha256="test")
            train_epoch(other,ds,other_opt,epoch=epoch,seed=seed,batch_size=3)
            for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(predict(model,ds),predict(other,ds))
            with self.assertRaises(ValueError):
                restore_checkpoint(path,other,other_opt,preprocessing_sha256="wrong")


if __name__ == "__main__":
    unittest.main()
