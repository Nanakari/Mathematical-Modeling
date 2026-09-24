"""Data, augmentation, model selection and export extension points."""
import csv
import hashlib
import itertools
import pickle
from pathlib import Path
import numpy as np
from aligned_dataset import AlignedDataset
from pipeline import PreparedDataset, make_plan, apply_plan, MODALITIES


class AlignedAdapter:
    def __init__(self, feature_path, special_folder, train_limit, valid_limit, special_token_ids=()):
        self.feature_path,self.special_folder = Path(feature_path),Path(special_folder)
        self.train_limit,self.valid_limit = train_limit,valid_limit
        self.special_token_ids = special_token_ids

    def load(self, seed):
        with self.feature_path.open("rb") as f: data=pickle.load(f)
        datasets={}
        for split,limit,offset in (("train",self.train_limit,0),("valid",self.valid_limit,1)):
            p=data[split]
            if limit<3: raise ValueError("Sample limit must be at least 3")
            indices=np.sort(np.random.default_rng(seed+offset).choice(len(p["id"]),min(limit,len(p["id"])),replace=False))
            selected={k:np.asarray(v)[indices] for k,v in AlignedDataset._select(p).items()}
            datasets[split]=AlignedDataset(selected,labeled=True,source=f"attachment2/{split}",special_token_ids=self.special_token_ids)
        if set(datasets['train'].part['id']) & set(datasets['valid'].part['id']):
            raise ValueError("Overlapping split IDs")
        return datasets["train"],datasets["valid"]

    def special(self):
        return AlignedDataset.from_attachment3(self.special_folder,special_token_ids=self.special_token_ids)


class MixedMissingDataset(PreparedDataset):
    def __init__(self, dataset, standardizer, seed, probability=.5, rates=(.1,.3,.5)):
        super().__init__(dataset,standardizer,augment=True,seed=seed)
        if not 0<=probability<=1 or not rates or any(not 0<r<1 for r in rates):
            raise ValueError("Invalid training corruption settings")
        self.probability,self.rates=probability,list(rates)
        self.combinations=[c for n in range(1,4) for c in itertools.combinations(MODALITIES,n)]

    def __getitem__(self,index):
        sample=self.dataset[index]
        token=f"{self.seed}|{self.epoch}|{sample['sample_id']}".encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'))
        if rng.random()<self.probability:
            combo=self.combinations[int(rng.integers(len(self.combinations)))]
            rate=float(rng.choice(self.rates))
            position=str(rng.choice(['start','middle','end','random']))
            plan=make_plan(sample,combo,rate,position,self.seed+self.epoch,rates=sorted(self.rates))
        else:
            plan=make_plan(sample,(),0.,"start",self.seed)
        return self.standardizer.transform(apply_plan(sample,plan))


class SelectionScore:
    def __init__(self,mae_weight=1.,f1_penalty_weight=1.):
        if min(mae_weight,f1_penalty_weight)<0 or mae_weight+f1_penalty_weight<=0:
            raise ValueError("Invalid selection weights")
        self.mw,self.fw=mae_weight,f1_penalty_weight

    def __call__(self,scores):
        return float(np.mean([self.mw*s['mae']+self.fw*(1-s['f1_macro']) for s in scores]))


class DraftExporter:
    """Replace once official columns and sample-ID mapping are confirmed."""
    def __init__(self): pass

    def export(self,path,predictions):
        seen=set();rows=[]
        for p in predictions:
            sid=p['sample_id']
            if sid in seen:raise ValueError("Duplicate output sample ID")
            seen.add(sid)
            intensity=float(p['pred_intensity']);c=int(p['pred_class'])
            if not np.isfinite(intensity) or not -3<=intensity<=3 or c not in (0,1,2):
                raise ValueError("Invalid prediction")
            probs=[p[f'prob_{i}'] for i in range(3)]
            if not np.isfinite(probs).all() or min(probs)<0 or max(probs)>1 or not np.isclose(sum(probs),1,atol=1e-6):
                raise ValueError("Invalid probabilities")
            rows.append({'sample_id':sid,'polarity':['Negative','Neutral','Positive'][c],
                         'intensity':intensity,**{f'prob_{i}':v for i,v in enumerate(probs)}})
        with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return len(rows)
