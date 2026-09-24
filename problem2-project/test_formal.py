import unittest
import numpy as np
from test_pipeline import fixture
from pipeline import make_plan,apply_plan
from formal_features import token_ngrams,temporal_summary,DescriptorScaler,FeatureMap,SemanticEncoder,combine
from formal_model import DualModel,ensemble


class FormalTest(unittest.TestCase):
    def test_ngram_does_not_bridge_missing_interval(self):
        sample=fixture()[1]
        altered=apply_plan(sample,make_plan(sample,('text',),.3,'middle',17))
        for term in token_ngrams(altered):
            if term.startswith('b'):
                a,b=map(int,term[1:].split('_'));self.assertEqual(b-a,1)
        altered['input_ids'][altered['text_missing']]=99999
        self.assertFalse(any('99999' in t for t in token_ngrams(altered)))

    def test_descriptor_masks_ignore_unobserved_values(self):
        s=fixture()[1];a=temporal_summary(s,'audio')
        s['audio'][~s['audio_observed']]=999999
        np.testing.assert_array_equal(a,temporal_summary(s,'audio'))
        z=fixture()[0];v,m=temporal_summary(z,'vision',True)
        self.assertFalse(m.any());self.assertTrue((v==0).all())
        scaler=DescriptorScaler().fit(np.array([[1.,100.],[3.,200.]]),np.array([[True,False],[True,False]]))
        values=scaler.transform(np.array([[1.,999.]]),np.array([[True,False]]))
        np.testing.assert_array_equal(values,[[-1.,0.]])

    def test_feature_map_dual_heads_finite_and_reloadable(self):
        ds=fixture(9);samples=[ds[i] for i in range(len(ds))]
        sem=np.random.default_rng(1).normal(size=(9,128))
        mapper=FeatureMap(min_df=1).fit(samples,sem);blocks=mapper.blocks(samples,sem)
        self.assertTrue(np.isfinite(combine(blocks).data).all())
        model=DualModel({},1,1).fit(blocks,ds.part['classification_labels'],ds.part['regression_labels'])
        p,r=ensemble([model,model],blocks)
        np.testing.assert_allclose(p.sum(1),1)
        self.assertTrue(np.all(abs(r)<=3))

    def test_semantic_encoder_masks_and_empty_pool(self):
        encoder=SemanticEncoder('models/bert_tiny',threads=2)
        s=fixture()[1];a=apply_plan(s,make_plan(s,('text',),.3,'middle',17))
        b={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in a.items()}
        b['input_ids'][~b['text_observed']]=999999
        np.testing.assert_array_equal(encoder.encode([a]),encoder.encode([b]))
        b['text_pool_mask'][:]=False
        np.testing.assert_array_equal(encoder.encode([b]),np.zeros((1,128)))


if __name__=='__main__':unittest.main()
