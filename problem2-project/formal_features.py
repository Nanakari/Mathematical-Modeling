"""Train-only multimodal feature map with local pretrained semantic encoding.

Inputs always use the observed token stream. No lookup of complete text by ID.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler,normalize
from transformers import AutoModel


def token_ngrams(sample):
    ids=sample['input_ids'];mask=sample['text_pool_mask'] & sample['text_observed'] & sample['attention_mask'].astype(bool)
    terms=[]
    for i in np.flatnonzero(mask):
        terms.append(f'u{int(ids[i])}')
        if i>0 and mask[i-1]:terms.append(f'b{int(ids[i-1])}_{int(ids[i])}')
    return terms


def temporal_summary(sample,modality,return_mask=False):
    x=np.asarray(sample[modality],dtype=np.float64);mask=sample[modality+'_observed']
    extent=sample['extent_mask'];n=int(extent.sum());dim=x.shape[1]
    def mean(part):return x[part].mean(0) if part.any() else np.zeros(dim)
    count=int(mask.sum());avg=mean(mask)
    std=x[mask].std(0) if count else np.zeros(dim)
    bins=[];bin_valid=[]
    for a,b in zip([0,n//3,2*n//3],[n//3,2*n//3,n]):
        part=mask.copy();part[:a]=False;part[b:]=False;bins.append(mean(part));bin_valid.append(part.any())
    adjacent=mask[1:] & mask[:-1]
    difference=np.abs(np.diff(x,axis=0)[adjacent]).mean(0) if adjacent.any() else np.zeros(dim)
    values=np.concatenate([avg,std,*bins,difference]).astype(np.float32)
    valid=np.repeat([count>0,count>0,*bin_valid,adjacent.any()],dim)
    return (values,valid) if return_mask else values


class DescriptorScaler:
    def fit(self,values,valid):
        count=valid.sum(0).clip(1)
        self.mean=(values*valid).sum(0)/count
        variance=((values-self.mean)**2*valid).sum(0)/count
        self.scale=np.where(variance>1e-12,np.sqrt(variance),1.)
        return self

    def transform(self,values,valid):
        return np.where(valid,(values-self.mean)/self.scale,0.)


class SemanticEncoder:
    def __init__(self,model_dir,threads=4):
        torch.set_num_threads(threads)
        provenance=Path(model_dir)/'provenance.json'
        if provenance.exists():
            for name,record in json.loads(provenance.read_text(encoding='utf-8'))['files'].items():
                if hashlib.sha256((Path(model_dir)/name).read_bytes()).hexdigest()!=record['sha256']:
                    raise ValueError('Pretrained artifact hash mismatch: '+name)
        self.model=AutoModel.from_pretrained(str(model_dir),local_files_only=True).eval()
        self.model.requires_grad_(False)
        self.cache={}

    @staticmethod
    def key(sample):
        attention=sample['attention_mask'].astype(bool)&sample['text_observed']
        ids=np.where(attention,sample['input_ids'],0).astype(np.int64)
        segments=np.where(attention,sample['token_type_ids'],0).astype(np.int64)
        pool=sample['text_pool_mask']&attention
        return hashlib.sha256(ids.tobytes()+attention.tobytes()+segments.tobytes()+pool.tobytes()).hexdigest()

    @torch.inference_mode()
    def encode(self,samples,batch_size=128):
        keys=[self.key(s) for s in samples]
        pending={k:s for k,s in zip(keys,samples) if k not in self.cache}
        items=list(pending.items())
        for start in range(0,len(items),batch_size):
            batch=items[start:start+batch_size]
            attention=np.stack([s['attention_mask'].astype(bool)&s['text_observed'] for _,s in batch])
            ids=np.stack([np.where(a,s['input_ids'],0) for a,(_,s) in zip(attention,batch)])
            segments=np.stack([np.where(a,s['token_type_ids'],0) for a,(_,s) in zip(attention,batch)])
            pool=np.stack([s['text_pool_mask']&a for a,(_,s) in zip(attention,batch)])
            safe=attention.copy();safe[~safe.any(1),0]=True
            h=self.model(input_ids=torch.as_tensor(ids,dtype=torch.long),
                         attention_mask=torch.as_tensor(safe,dtype=torch.long),
                         token_type_ids=torch.as_tensor(segments,dtype=torch.long)).last_hidden_state
            m=torch.as_tensor(pool).unsqueeze(-1)
            vectors=(torch.where(m,h,0.).sum(1)/m.sum(1).clamp_min(1)).numpy()
            for (key,_),vector in zip(batch,vectors):self.cache[key]=vector
        return np.stack([self.cache[k] for k in keys])


class FeatureMap:
    def __init__(self,max_features=12000,min_df=2):
        self.lexical=TfidfVectorizer(analyzer=token_ngrams,lowercase=False,min_df=min_df,max_features=max_features,sublinear_tf=True,dtype=np.float64)
        self.semantic_scaler=StandardScaler()
        self.av_scalers={m:DescriptorScaler() for m in ['audio','vision']}

    def fit(self,samples,semantics):
        self.lexical.fit(samples)
        self.semantic_scaler.fit(semantics)
        for m in self.av_scalers:
            pairs=[temporal_summary(s,m,True) for s in samples]
            self.av_scalers[m].fit(np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs]))
        return self

    def blocks(self,samples,semantics):
        blocks={'lexical':self.lexical.transform(samples)}
        blocks['semantic']=normalize(self.semantic_scaler.transform(semantics)).astype(np.float64)
        fractions=np.zeros((len(samples),3),dtype=np.float64)
        for i,s in enumerate(samples):
            den=max(1,int(s['extent_mask'].sum()))
            fractions[i]=[np.sum(s['text_pool_mask']&s['text_observed'])/den,
                          np.sum(s['audio_observed'])/den,np.sum(s['vision_observed'])/den]
        for m in ['audio','vision']:
            pairs=[temporal_summary(s,m,True) for s in samples]
            values=self.av_scalers[m].transform(np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs]))
            # Each modality has a comparable overall scale despite dimensions.
            values=np.clip(values,-5,5)/np.sqrt(values.shape[1])
            j=1 if m=='audio' else 2
            values[fractions[:,j]==0]=0
            blocks[m]=values
        blocks['semantic'][fractions[:,0]==0]=0
        blocks['fractions']=fractions
        return blocks


def combine(blocks,av_weight=.1,semantic_weight=1.,reliability=True,modalities=('text','audio','vision'),semantic=True):
    result=[];q=blocks['fractions']
    if 'text' in modalities:
        factor=np.sqrt(q[:,0]) if reliability else np.ones(len(q))
        result.append(blocks['lexical'].multiply(factor[:,None]))
        if semantic:result.append(sparse.csr_matrix(blocks['semantic']*factor[:,None]*semantic_weight))
    for i,m in enumerate(['audio','vision'],1):
        if m in modalities:
            factor=np.sqrt(q[:,i]) if reliability else np.ones(len(q))
            result.append(sparse.csr_matrix(blocks[m]*factor[:,None]*av_weight))
    if reliability:
        # Explicit availability is useful even when missing content equals zero.
        include=[i for i,m in enumerate(['text','audio','vision']) if m in modalities]
        result.append(sparse.csr_matrix(q[:,include]*.1))
    return sparse.hstack(result,format='csr')
