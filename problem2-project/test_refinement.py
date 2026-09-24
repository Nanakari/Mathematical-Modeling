import unittest
import numpy as np
from test_pipeline import fixture
from formal_features import FeatureMap
from refinement_model import view_weights,adjusted_probabilities,calibrated_intensity,RefinedModel

class RefinementTests(unittest.TestCase):
    def test_original_sample_weight_conservation(self):
        for mass in [0.25,.6,1.]:
            w=view_weights(9,mass).reshape(4,9)
            np.testing.assert_allclose(w.sum(0),1.)
            np.testing.assert_allclose(w[0],mass)
        with self.assertRaises(ValueError):view_weights(2,1.1)

    def test_neutral_rule_changes_decision_without_mutating_input(self):
        p=np.array([[.4,.3,.3],[.1,.2,.7]])
        q=adjusted_probabilities(p,1.5)
        self.assertEqual(q[0].argmax(),1);self.assertEqual(q[1].argmax(),2)
        np.testing.assert_allclose(q.sum(1),1.)
        np.testing.assert_array_equal(p,[[.4,.3,.3],[.1,.2,.7]])

    def test_calibration_clips_and_preserves_order(self):
        z=calibrated_intensity([-10,-1,0,1,10],1.2,-.1)
        np.testing.assert_allclose(z,[-3,-1.3,-.1,1.1,3])

    def test_separate_task_dimensions_absolute_error_head_and_reload(self):
        import joblib,tempfile
        from pathlib import Path
        ds=fixture(12);ss=[ds[i] for i in range(len(ds))]
        sem=np.random.default_rng(7).normal(size=(12,128))
        mapper=FeatureMap(min_df=1).fit(ss,sem);b=mapper.blocks(ss*4,np.tile(sem,(4,1)))
        model=RefinedModel(dict(fusion='text',clean_mass=1.,C=.5),
            dict(fusion='av03_plain',clean_mass=.6,kind='absolute_svr',C=.1),1.25,1.,0.)
        model.fit(b,ds.part['classification_labels'],ds.part['regression_labels'])
        self.assertNotEqual(model.classifier.n_features_in_,model.regressor.n_features_in_)
        p,z=model.predict(mapper.blocks(ss,sem));self.assertTrue(np.isfinite(z).all())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.joblib';joblib.dump(model,path)
            pp,zz=joblib.load(path).predict(mapper.blocks(ss,sem))
            np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(z,zz)

if __name__=='__main__':unittest.main()
