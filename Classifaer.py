from sklearn.base import TransformerMixin, BaseEstimator,ClassifierMixin
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class OneClassCV(TransformerMixin,BaseEstimator,ClassifierMixin):
    def __init__(self,classifaer,parameter):
        self.classifaer = classifaer
        self.parameter = parameter
        self.verbose = True
        self.clf = None

    def putNoiseInData(self, X, count=10):
        arrayNoise = np.zeros(X.shape)
        arrayNoise = np.std(X) * np.random.randn(arrayNoise.shape[0] * count, arrayNoise.shape[1]) + np.mean(X)

        noiseTarget = [1]*len(arrayNoise)
        return arrayNoise, noiseTarget

    def selectBestParameterComb(self, X, noiseTest,y, noiseY):
        model = self.classifaer
        scoreTrain = []
        scoreNoise = []
        gridPar = list(ParameterGrid(self.parameter))
        for par in gridPar:
            clf = model['oneclasscv'].set_params(**par)
            clf.fit(X,y)
            predictOnTrain = clf.predict(X)
            predictOnNoise = clf.predict(noiseTest)
            scoreTrain.append(accuracy_score(y,predictOnTrain))
            scoreNoise.append(accuracy_score(noiseY,predictOnNoise))
        if self.verbose==True:
            bestParInd = np.argmax(np.array(scoreTrain)-np.array(scoreNoise))
            print (gridPar[bestParInd], "- лучшие параметры")
            print (np.array(scoreTrain)[bestParInd], "- точность на обучении")
            print(np.array(scoreNoise)[bestParInd], "- точность на тесте")
            print ((np.array(scoreTrain)-np.array(scoreNoise))[bestParInd], "- разница")

        self.clf = model['oneclasscv'].set_params(**gridPar[bestParInd])
        # pca = PCA(n_components=2)
        # pca.fit_transform(np.vstack([X, noiseTest]))
        # plt.scatter(np.vstack([X, noiseTest])[:len(X), 0], np.vstack([X, noiseTest])[:len(X), 1])
        # plt.scatter(np.vstack([X, noiseTest])[len(X):, 0], np.vstack([X, noiseTest])[len(X):, 1])
        # plt.show()
        return

    def fit(self, X, y):
        noiseTest, noiseTarget = self.putNoiseInData(X)
        self.selectBestParameterComb(X,noiseTest,y,noiseTarget)
        self.clf.fit(X,y)
        return self

    def transform(self, X, y=None):
        return

    def predict(self,X):
        predict = self.clf.predict(X)
        return predict