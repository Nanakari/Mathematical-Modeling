import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from components import ModularModel,construct,FUSIONS,LocalBert,JointLoss
from pipeline import MaskedStandardizer,PreparedDataset
from workflow_data import MixedMissingDataset,SelectionScore,DraftExporter
from torch_training import predict
from test_pipeline import fixture


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.data=fixture();self.norm=MaskedStandardizer().fit(self.data,split='train')

    def config(self,fusion,sequence):
        return {'hidden':8,'text':{'name':'embedding','vocab_size':32},
                'sequence':{'name':sequence},'fusion':{'name':fusion}}

    def test_swap_components_finite_backward_and_mask_invariance(self):
        batch=next(iter(DataLoader(PreparedDataset(self.data,self.norm),batch_size=3)))
        for fusion in ['concat','gated']:
            for sequence in ['linear','temporal_conv']:
                model=ModularModel(self.config(fusion,sequence))
                before=model(batch)
                changed={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                changed['input_ids'][~changed['text_observed']]=99999
                for m in ['audio','vision']:changed[m][~changed[m+'_observed']]=float('nan')
                after=model(changed)
                for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
                for m in ['text_observed','text_pool_mask','audio_observed','vision_observed']:changed[m].zero_()
                logits,reg=model(changed);(logits.sum()+reg.sum()).backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_extension_points_and_unverified_bert(self):
        layer=construct({'name':'components:ConcatFusion','hidden':8},FUSIONS)
        self.assertEqual(layer.output_dim,24)
        with self.assertRaises(ValueError):LocalBert(8,'unused',tokenizer_verified=False)
        with self.assertRaises(ValueError):JointLoss(class_weight=-1)
        self.assertAlmostEqual(SelectionScore()([{'mae':.5,'f1_macro':.6}]),.9)

    def test_mixed_augmentation_and_unlabeled_export(self):
        a=MixedMissingDataset(self.data,self.norm,seed=77,probability=1.)
        first=a[1];again=a[1]
        np.testing.assert_array_equal(first['input_ids'],again['input_ids'])
        self.assertTrue(any(first[m+'_missing'].any() for m in ['text','audio','vision']))
        clean=MixedMissingDataset(self.data,self.norm,seed=77,probability=0.)[1]
        self.assertFalse(any(clean[m+'_missing'].any() for m in ['text','audio','vision']))
        model=ModularModel(self.config('gated','linear'))
        unlabeled=fixture(3,split='special',labeled=False)
        rows=predict(model,PreparedDataset(unlabeled,self.norm))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(DraftExporter().export(Path(tmp)/'pred.csv',rows),3)
            with self.assertRaises(ValueError):DraftExporter().export(Path(tmp)/'bad.csv',rows+rows[:1])


if __name__=='__main__':unittest.main()
