from sklearn.base import TransformerMixin, BaseEstimator
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.grid_search import ParameterGrid


class OneClassCV(TransformerMixin,BaseEstimator):
    def __init__(self,classifaer,parameter):
        self.classifaer = classifaer
        self.parameter = parameter


    def putNoiseInData(self, X, count=10):
        arrayNoise = np.zeros(X.shape)
        arrayNoise = np.std(X) * np.random.randn(arrayNoise.shape[0] * count, arrayNoise.shape[1]) + np.mean(X)
        return arrayNoise

    def selectBestParameterComb(self, X, noiseTest):
        model = self.classifaer#.fit(X)
        print (model)

        for k, v in self.parameter.items():
            print (k,v)
            for val in v:
                clf = model['oneclasscv'].set_params(**{k: val})
                print(clf)

        return

    def fit(self, X, y):
        noiseTest = self.putNoiseInData(X)
        self.selectBestParameterComb(X,noiseTest)
        return self

    def transform(self, X, y=None):
        return

    def predict(self):
        return