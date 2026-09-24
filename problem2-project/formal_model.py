"""Regularized dual-task model with observed-content feature fusion."""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.exceptions import ConvergenceWarning
from formal_features import combine


def fit_classifier(x,y,C,class_weight='balanced',sample_weight=None):
    for max_iter in (500,2000):
        clf=LogisticRegression(C=C,class_weight=class_weight,solver='lbfgs',max_iter=max_iter,tol=1e-5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            clf.fit(x,y,sample_weight=sample_weight)
        if not any(issubclass(w.category,ConvergenceWarning) for w in caught):return clf
    raise RuntimeError('Logistic regression did not converge; do not report unconverged fits')


def fit_regressor(x,y,alpha,sample_weight=None):
    return Ridge(alpha=alpha,solver='lsqr',tol=1e-6,max_iter=5000).fit(x,y,sample_weight=sample_weight)


class DualModel:
    def __init__(self,feature_config,C,alpha,class_weight='balanced'):
        self.feature_config=feature_config;self.C=C;self.alpha=alpha;self.class_weight=class_weight

    def fit(self,blocks,classes,intensities,sample_weight=None):
        x=combine(blocks,**self.feature_config)
        self.classifier=fit_classifier(x,classes,self.C,self.class_weight,sample_weight)
        self.regressor=fit_regressor(x,intensities,self.alpha,sample_weight)
        return self

    def predict(self,blocks):
        x=combine(blocks,**self.feature_config)
        p=self.classifier.predict_proba(x)
        r=np.clip(self.regressor.predict(x),-3,3)
        return p,r


def ensemble(models,blocks):
    predictions=[m.predict(blocks) for m in models]
    return np.mean([p for p,r in predictions],axis=0),np.mean([r for p,r in predictions],axis=0)
