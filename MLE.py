import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, input_cols=[]):
        self.input_cols = input_cols
        self.classes_ = []
        self.in_cols_range = range(len(self.input_cols))

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.classes_ = [np.unique(X[:, i]) for i in self.input_cols]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        xcp = X.copy()
        classes = [np.unique(X[:, i]) for i in self.input_cols]
        for i in self.in_cols_range:
            if len(np.intersect1d(classes[i], self.classes_[i])) < len(classes[i]):
                diff = np.setdiff1d(classes[i], self.classes_[i])
                raise ValueError("X[%d] contains new labels: %s" % (i, str(diff)))
            else:
                xcp[:, self.input_cols[i]] = np.searchsorted(self.classes_[i], X[:, self.input_cols[i]])
        return pd.DataFrame(xcp)


