from sklearn.base import ClassifierMixin, BaseEstimator
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
class GaussClassifaer(BaseEstimator, ClassifierMixin):

    def __init__(self, up_bound=0.5, method=None):
        self.up_bound=up_bound
        self.rv = None
        self.maxB=None
        self.method = method

    def fit(self, X,y):
        standardized_X = X
        clf_1 = KMeans(n_clusters=1, random_state=0).fit(standardized_X)
        if self.method!=None:
            mu1, cov1 = clf_1.cluster_centers_, self.method.fit(standardized_X)
            self.rv = multivariate_normal(cov1.location_, cov1.covariance_)
        else:
            mu1, cov1 = clf_1.cluster_centers_, np.cov(standardized_X.T)
            self.rv = multivariate_normal(mu1[0], cov1)
        self.maxB = np.max(self.rv.pdf(standardized_X))
        return self

    def getMaxBound(self):
        return self.maxB

    def transform(self, X, y=None):
        pass


    def predict(self, X, y=None):
        standardized_X = X
        test_p = self.rv.pdf(standardized_X)
        r = np.array([j > self.up_bound for j in test_p])*1
        return r

    def getTestZ(self,X):
        normalized_X = preprocessing.normalize(X)
        standardized_X = preprocessing.scale(normalized_X)
        test_p = self.rv.pdf(standardized_X)
        return test_p



