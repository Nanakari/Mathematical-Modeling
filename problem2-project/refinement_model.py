"""Independent task fusion, clean/augmented mass, and replaceable decision rules."""
import warnings
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.exceptions import ConvergenceWarning
from formal_features import combine
from formal_model import fit_classifier,fit_regressor

FUSIONS={
    'text':dict(modalities=('text',),reliability=False),
    'av03_plain':dict(av_weight=.3,reliability=False),
    'av03_quality':dict(av_weight=.3,reliability=True),
    'av1_plain':dict(av_weight=1.,reliability=False)}

def view_weights(n,clean_mass,views=3):
    if not 0<=clean_mass<=1:raise ValueError('Invalid clean mass')
    return np.concatenate([np.full(n,clean_mass),np.full(n*views,(1-clean_mass)/views)])

def adjusted_probabilities(probabilities,neutral_weight=1.):
    if neutral_weight<=0:raise ValueError('Neutral weight must be positive')
    p=np.asarray(probabilities,dtype=float).copy();p[:,1]*=neutral_weight
    return p/p.sum(1,keepdims=True)

def calibrated_intensity(raw,scale=1.,offset=0.):
    return np.clip(scale*np.asarray(raw)+offset,-3,3)

def fit_head(blocks,y,spec,task):
    x=combine(blocks,**FUSIONS[spec['fusion']]);n=len(y)
    weights=view_weights(n,spec['clean_mass']);target=np.tile(y,4)
    use=weights>0;x=x[use];target=target[use];weights=weights[use]
    if task=='classification':return fit_classifier(x,target,spec['C'],'balanced',weights)
    if spec['kind']=='ridge':return fit_regressor(x,target,spec['alpha'],weights)
    if spec['kind']!='absolute_svr':raise ValueError(spec['kind'])
    # epsilon=0: absolute prediction error plus L2 regularization.
    # Fit a regularized intercept; this differs slightly from Ridge's intercept.
    for iterations in [5000,20000]:
        model=LinearSVR(C=spec['C'],epsilon=0.,loss='epsilon_insensitive',
                        dual=True,tol=1e-5,max_iter=iterations,random_state=1729)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(x,target,sample_weight=weights)
        if not any(issubclass(w.category,ConvergenceWarning) for w in caught):return model
    raise RuntimeError('Absolute-error SVR did not converge')

class RefinedModel:
    def __init__(self,class_spec,reg_spec,neutral_weight=1.,scale=1.,offset=0.):
        self.class_spec=dict(class_spec);self.reg_spec=dict(reg_spec)
        self.neutral_weight=neutral_weight;self.scale=scale;self.offset=offset
    def fit(self,blocks,classes,intensities):
        self.classifier=fit_head(blocks,classes,self.class_spec,'classification')
        self.regressor=fit_head(blocks,intensities,self.reg_spec,'regression')
        return self
    def predict(self,blocks):
        p=self.classifier.predict_proba(combine(blocks,**FUSIONS[self.class_spec['fusion']]))
        r=self.regressor.predict(combine(blocks,**FUSIONS[self.reg_spec['fusion']]))
        return adjusted_probabilities(p,self.neutral_weight),calibrated_intensity(r,self.scale,self.offset)
