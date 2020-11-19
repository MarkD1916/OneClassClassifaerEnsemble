from SearcherFile import Searcher
from Loader import RatDataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Preprocessing import Features
from Classifaer import OneClassCV
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from GaussianClassifaer import GaussClassifaer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.covariance import OAS, MinCovDet,LedoitWolf,EmpiricalCovariance
#from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from GMClassifaer import GaussMixtureClassifaer
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression

mainDir = '/home/dmv/Desktop/Rat'
day = ["2020_09_09"] # если пустой массив - выбирает все дни, дни задаются датой как в названии папки с экспериментом
year = '2020'
rat=['n51']
drugs=[]
Search = Searcher(mainDir=mainDir,year=year,rat=rat,verbose=False,metaDataByRatNum={},day=day,drugs=drugs)
Search.fileSearch()
jsonData = Search.parsJson()
L=RatDataLoader(metaDataByRatNum=jsonData,mainDir=mainDir,year=year,rat=rat,verbose=False,day=day,drugs=drugs)
Data = L.loadData()
# загрузка и предобработка данных
Y_data_train1 = Data[rat[0]][day[0]]['1']['trainEvent']
X_data_train1 = Data[rat[0]][day[0]]['1']['train']
Y_data_test1 = Data[rat[0]][day[0]]['1']['testEvent']
X_data_test1 = Data[rat[0]][day[0]]['1']['test']
Y_data_test2 = Data[rat[0]][day[0]]['2']['testEvent']
X_data_test2 = Data[rat[0]][day[0]]['2']['test']
Y_data_train2 = Data[rat[0]][day[0]]['2']['trainEvent']
X_data_train2 = Data[rat[0]][day[0]]['2']['train']
#Y_data_test2 = Data[rat[0]][day[0]]['2']['trainEvent']
#X_data_test2 = Data[rat[0]][day[0]]['2']['train']
Xnoise = np.vstack([X_data_train1[Y_data_train1!=1],X_data_test1[Y_data_test1!=1],X_data_train2[Y_data_train2!=1],X_data_test2[Y_data_test2!=1]])
Ynoise = np.concatenate([Y_data_train1[Y_data_train1!=1],Y_data_test1[Y_data_test1!=1],Y_data_test2[Y_data_test2!=1],Y_data_train2[Y_data_train2!=1]])
X_data_train1 = X_data_train1[Y_data_train1==1]
Y_data_train1 = Y_data_train1[Y_data_train1==1]
X_data_test1=X_data_test1[Y_data_test1==1]
Y_data_test1=Y_data_test1[Y_data_test1==1]
Ynoise[Ynoise!=1]=1

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# print ("OneClassSVM")
# paramGrid1={'nu':np.arange(0.01,1,0.03), 'gamma': [(10**(x*(-1)))/4 for x in range(1,20)],'kernel':['rbf']}
# model1 =OneClassSVM()
# mem1 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
#                  ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model1)]),paramGrid1))])
# mem1.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem1.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
# #print(mem1.predict(Xnoise[:,[5,7,12],:]))
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# print ("IsolationForest")
# paramGrid2={'n_estimators':[x for x in range(100,500,50)], 'contamination':np.arange(0.01,0.5,0.03),'random_state':[0]}
# model2 =IsolationForest()
# mem2 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
#                  ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model2)]),paramGrid2))])
# mem2.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem2.predict(Xnoise[:,[5,7,12],:])))
# print(mem2.predict(Xnoise[:,[5,7,12],:]), "- точность на веществах")
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# print ("Gaussian")
# paramGrid3={'up_bound':np.arange(0.01,0.16,0.01), 'method':[EmpiricalCovariance(),LedoitWolf(),OAS(),MinCovDet(random_state=0),None]}
# model3 =GaussClassifaer()
# mem3 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
#                  ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model3)]),paramGrid3))])
# mem3.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem3.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
# #print(mem3.predict(Xnoise[:,[5,7,12],:]))
#
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# print ("LOF")
# paramGrid4={'n_neighbors':[x for x in range(2, 30, 2)], 'contamination':np.arange(0.01, 0.5, 0.03),'novelty':[True]}
# model4 =LocalOutlierFactor()
# mem4 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
#                  ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model4)]),paramGrid4))])
# mem4.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem4.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
#mem4.predict(Xnoise[:,[5,7,12],:])

print ("GaussMixture")
paramGrid5={'up_bound':np.arange(0.1,9,0.5)*(-1),'n_components':range(1,3), 'covariance_type':['full'],'init_params':['kmeans']}
model5 =GaussMixtureClassifaer()
mem5 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model5)]),paramGrid5))])
mem5.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
print (mem5.predict(Xnoise[:,[5,7,12],:]))
print (accuracy_score(Ynoise,mem5.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
print (mem5.predict(X_data_train1[:,[5,7,12],:]))
print (accuracy_score(Y_data_train1,mem5.predict(X_data_train1[:,[5,7,12],:])), "- точность на ТНТ")
