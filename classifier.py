from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from scattering import ScatteringTransform


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = OneVsRestClassifier(LinearSVC(random_state=0))
        self.scat = ScatteringTransform(level=1)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        X = X[:, :, 5:-5, 5:-5]
        return np.array([self._transform(x) for x in X])

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def _transform(self, x):
        _rpz = [self.scat.fit(x[i], layer=2) for i in range(3)]
        rpz = []
        for rr in _rpz:
            for r in rr:
                rpz += r
        return rpz

    def fit(self, X, y):
        X = self.preprocess(X)
        print('Start fitting')
        self.clf.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        X = self.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.clf.predict_proba(X)
