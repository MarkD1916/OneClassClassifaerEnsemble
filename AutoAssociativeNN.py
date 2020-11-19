from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import DistanceMetric
import numpy as np
from sklearn.cluster import KMeans
class AANN(BaseEstimator, ClassifierMixin):

    def __init__(self,up_bound=0,hidden_layer_sizes=0,solver='adam',alpha=0):
        self.up_bound=up_bound
        self.hidden_layer_sizes = hidden_layer_sizes
        self.clf=None
        self.xTrain = None
        self.xPredict = None
        self.solver=solver
        self.alpha=alpha
    def fit(self, X,y):
        self.clf = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,random_state=1,solver=self.solver,
                                alpha=self.alpha)
        y = X
        self.clf.fit(X,y)
        clf_1 = KMeans(n_clusters=1, random_state=1).fit(X)
        self.xTrain = clf_1.cluster_centers_
        return self


    def transform(self, X, y=None):
        pass


    def predict(self, X, y=None):
        self.xPredict = self.clf.predict(X) # трансформированная выборка
        dist = DistanceMetric.get_metric('euclidean')
        # d = [dist.pairwise([sTr,sP])[0][1] for sTr,sP in zip(X,self.xPredict)]
        # print(d)
        # print (np.min(d))
        # print (np.max(d))
        # print(np.var(d))
        # print(len([dist.pairwise([sTr,sP])[0][1] for sTr,sP in zip(self.xTrain,self.xPredict)]))
        predicted=[0 if dist.pairwise([self.xTrain[0],sP])[0][1]>self.up_bound else 1 for sP in self.xPredict]
        return predicted