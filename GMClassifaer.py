from sklearn.mixture import GaussianMixture
from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np
class GaussMixtureClassifaer(BaseEstimator, ClassifierMixin):

    def __init__(self, up_bound=0.5,n_components=1,covariance_type='full',init_params='kmeans',random_state=1):
        self.up_bound=up_bound
        self.n_components=n_components
        self.covariance_type=covariance_type
        self.init_params=init_params
        self.random_state=random_state
        self.log_probs_train = None
        self.clf=None
    def fit(self, X,y):
        self.clf = GaussianMixture(n_components=self.n_components,covariance_type=self.covariance_type,init_params=self.init_params,
                              random_state=self.random_state)
        self.clf.fit(X)
        return self


    def transform(self, X, y=None):
        pass


    def predict(self, X, y=None):

        log_probs_test = self.clf.score_samples(X)
        predict = [1 if prob >= self.up_bound else 0 for prob in log_probs_test]

        return predict


