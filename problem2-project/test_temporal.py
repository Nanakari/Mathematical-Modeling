import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from pipeline import MaskedStandardizer, make_plan, apply_plan
from temporal_model import TemporalDual, MaskedTemporal, tensors, predict
from test_pipeline import fixture


class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(7)
        self.ds=fixture(3)
        self.norm=MaskedStandardizer().fit(self.ds,split='train')
        self.samples=[self.ds[i] for i in range(3)]

    def test_unobserved_values_cannot_change_temporal_outputs(self):
        model=TemporalDual('models/bert_mini','mini_mag').eval()
        samples=[apply_plan(s,make_plan(s,('text','audio','vision'),.3,'middle',17)) for s in self.samples]
        batch=tensors(samples,self.norm); before=predict(model,batch)
        batch['ids'][~batch['attention']]=999999
        batch['segments'][~batch['attention']]=999999
        for modality in ['audio','vision']:
            batch[modality][~batch[modality+'_mask']]=float('nan')
        for a,b in zip(before,predict(model,batch)):
            np.testing.assert_array_equal(a,b)
            self.assertTrue(np.isfinite(b).all())

    def test_missing_text_does_not_remove_available_audio(self):
        model=TemporalDual('models/bert_mini','mini_mag').eval()
        batch=tensors(self.samples,self.norm)
        batch['attention'][:]=False; batch['pool'][:]=False
        with torch.no_grad(): before=model(**batch)[0]
        batch['audio_mask'][:]=False
        with torch.no_grad(): after=model(**batch)[0]
        self.assertGreater(float((before-after).abs().max()),1e-7)
        batch['vision_mask'][:]=False
        p,z=predict(model,batch)
        self.assertTrue(np.isfinite(p).all() and np.isfinite(z).all())

    def test_text_control_independent_of_audio_video(self):
        model=TemporalDual('models/bert_mini','mini_text')
        batch=tensors(self.samples,self.norm); a=predict(model,batch)
        batch['audio'][:]=999; batch['vision'][:]=-999
        batch['audio_mask'][:]=False; batch['vision_mask'][:]=True
        for x,y in zip(a,predict(model,batch)): np.testing.assert_array_equal(x,y)

    def test_masked_layers_preserve_gaps_and_all_missing_zero(self):
        encoder=MaskedTemporal(4,8,0.).eval()
        mask=torch.tensor([[True,False,True,True],[False]*4])
        x=torch.randn(2,4,4)
        with torch.no_grad(): y=encoder(x,mask)
        self.assertTrue(torch.equal(y[~mask],torch.zeros_like(y[~mask])))

    def test_saved_optimizer_rng_continues_same_training(self):
        model=TemporalDual('models/bert_tiny','tiny_text',dropout=.2)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        batch=tensors(self.samples,self.norm)
        def step(m,o):
            m.train(); o.zero_grad(); logits,z=m(**batch)
            (logits.square().mean()+z.square().mean()).backward(); o.step()
        step(model,opt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pt'
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()},path)
            step(model,opt); expected=predict(model,batch)
            reloaded=TemporalDual('models/bert_tiny','tiny_text',dropout=.2)
            other=torch.optim.AdamW(reloaded.parameters(),lr=1e-4)
            saved=torch.load(path,weights_only=True)
            reloaded.load_state_dict(saved['model']); other.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['rng'])
            step(reloaded,other)
            for a,b in zip(expected,predict(reloaded,batch)): np.testing.assert_array_equal(a,b)


if __name__=='__main__': unittest.main()
