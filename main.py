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
from AutoAssociativeNN import AANN
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.covariance import OAS, MinCovDet,LedoitWolf,EmpiricalCovariance
from GMClassifaer import GaussMixtureClassifaer
import data
from data import load_data

# mainDir = '/home/dmv/Desktop/Rat'
# day = ["2020_09_09"] # если пустой массив - выбирает все дни, дни задаются датой как в названии папки с экспериментом
# year = '2020'
# rat=['n51']
# drugs=[]
# Search = Searcher(mainDir=mainDir,year=year,rat=rat,verbose=False,metaDataByRatNum={},day=day,drugs=drugs)
# Search.fileSearch()
# jsonData = Search.parsJson()
# L=RatDataLoader(metaDataByRatNum=jsonData,mainDir=mainDir,year=year,rat=rat,verbose=False,day=day,drugs=drugs)
# Data = L.loadData()
# # загрузка и предобработка данных
# Y_data_train1 = Data[rat[0]][day[0]]['1']['trainEvent']
# X_data_train1 = Data[rat[0]][day[0]]['1']['train']
# Y_data_test1 = Data[rat[0]][day[0]]['1']['testEvent']
# X_data_test1 = Data[rat[0]][day[0]]['1']['test']
# Y_data_test2 = Data[rat[0]][day[0]]['2']['testEvent']
# X_data_test2 = Data[rat[0]][day[0]]['2']['test']
# Y_data_train2 = Data[rat[0]][day[0]]['2']['trainEvent']
# X_data_train2 = Data[rat[0]][day[0]]['2']['train']
# #Y_data_test2 = Data[rat[0]][day[0]]['2']['trainEvent']
# #X_data_test2 = Data[rat[0]][day[0]]['2']['train']
# Xnoise = np.vstack([X_data_train1[Y_data_train1!=1],X_data_test1[Y_data_test1!=1],X_data_train2[Y_data_train2!=1],X_data_test2[Y_data_test2!=1]])
# Ynoise = np.concatenate([Y_data_train1[Y_data_train1!=1],Y_data_test1[Y_data_test1!=1],Y_data_test2[Y_data_test2!=1],Y_data_train2[Y_data_train2!=1]])
# X_data_train1 = X_data_train1[Y_data_train1==1]
# Y_data_train1 = Y_data_train1[Y_data_train1==1]
# X_data_test1=X_data_test1[Y_data_test1==1]
# Y_data_test1=Y_data_test1[Y_data_test1==1]
# Ynoise[Ynoise!=1]=1

trainFileName = ["08.10.2020.FI118.1.plx"]
testFileName = ["08.10.2020.FI118.2.plx"]

numCh = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
data.NUM_EEG_CHANNELS = len(numCh)
data.NUM_ALL_CHANNELS = data.NUM_EEG_CHANNELS + 2
data.IDXS_EEG_CHANNELS = slice(0, data.NUM_EEG_CHANNELS)
data.IDX_BREATH_CHANNEL = data.NUM_EEG_CHANNELS
data.ALTER_STIM_CHANNEL = True
piece = 5
data.DECIMATE=1

data.SAMPLE_RATE = 1000
data.LEN_STIMUL_SECS = 5
data.LEN_STIMUL = piece * data.SAMPLE_RATE
data.BEFORE_STIMUL = data.LEN_STIMUL
print (data.BEFORE_STIMUL)
train = load_data('odor',
                  dir="/home/dmv/Desktop/Rat/2020/2020_10_08-i118-5odor-18chan-dtb",
                  series=trainFileName,
                  categs=[1,2,4,8,16], take_signal='stimul', take_ecg=False, det_breath=2, take_air=False,
                  piece=5, take_spikes=False, verbose=False,add_breath=False)


test = load_data('odor',
              dir="/home/dmv/Desktop/Rat/2020/2020_10_08-i118-5odor-18chan-dtb",
              series=testFileName,
              categs=[1,2,4,8,16], take_signal='stimul', take_ecg=False, det_breath=2, take_air=False,
              piece=5, take_spikes=False, verbose=False,add_breath=False)



Target_drug = [1]

X_data_train = train[0].get('').get('').get(trainFileName[0])[0]

Y_data_train = train[0].get('').get('').get(trainFileName[0])[1]

X_data_train = X_data_train[Y_data_train==Target_drug[0]]

Y_data_train = Y_data_train[Y_data_train == Target_drug[0]]

X_data_test = test[0].get('').get('').get(testFileName[0])[0]

Y_data_test = test[0].get('').get('').get(testFileName[0])[1]

Y_data_test[Y_data_test!=Target_drug[0]] = 0
#Test_nature_noise = np.vstack([X_data_test[Y_data_test!=1]])

#Test_nature_noise_target =np.concatenate([Y_data_test[Y_data_test!=1]])
# trainEvent = train[0].get('').get('').get(trainFileName)[1]
# trainData = train[0].get('').get('').get(trainFileName)[0]
# print (trainData.shape)
#
# trainTestCatages = [trainCategs, testCategs]
# if testFileName != "":
#     for i, j in zip(trainTestCatages[0], trainTestCatages[1]): # переобозначаю метки, если клапана менялись
#         if i != j:
#             testEvent[testEvent == j] = i * -1
#     testEvent[testEvent < 0] = testEvent[testEvent < 0] * -1



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#print ("OneClassSVM")
paramGrid1={'nu':np.arange(0.01,1,0.03), 'gamma': [(10**(x*(-1)))/4 for x in range(1,20)],'kernel':['rbf']}
model1 =OneClassSVM()
mem1 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model1)]),paramGrid1,1))])
# mem1.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem1.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
# #print(mem1.predict(Xnoise[:,[5,7,12],:]))
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#print ("IsolationForest")
paramGrid2={'n_estimators':[x for x in range(100,500,50)], 'contamination':np.arange(0.01,0.5,0.03),'random_state':[0]}
model2 =IsolationForest()
mem2 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model2)]),paramGrid2,1))])
# mem2.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem2.predict(Xnoise[:,[5,7,12],:])))
# print(mem2.predict(Xnoise[:,[5,7,12],:]), "- точность на веществах")
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# print ("Gaussian")
paramGrid3={'up_bound':np.arange(0.01,0.16,0.01), 'method':[EmpiricalCovariance(),LedoitWolf(),OAS(),MinCovDet(random_state=0),None]}
model3 =GaussClassifaer()
mem3 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model3)]),paramGrid3,1))])
# mem3.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem3.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
# #print(mem3.predict(Xnoise[:,[5,7,12],:]))
#
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# print ("LOF")
paramGrid4={'n_neighbors':[x for x in range(2, 30, 2)], 'contamination':np.arange(0.01, 0.5, 0.03),'novelty':[True]}
model4 =LocalOutlierFactor()
mem4 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model4)]),paramGrid4,1))])
# mem4.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (accuracy_score(Ynoise,mem4.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
#mem4.predict(Xnoise[:,[5,7,12],:])
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# print ("GaussMixture")
paramGrid5={'up_bound':np.arange(0.1,9,0.5)*(-1),'n_components':range(1,3), 'covariance_type':['full'],'init_params':['kmeans']}
model5 =GaussMixtureClassifaer()
mem5 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model5)]),paramGrid5,1))])
# mem5.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (mem5.predict(Xnoise[:,[5,7,12],:]))
# print (accuracy_score(Ynoise,mem5.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
# print (mem5.predict(X_data_train1[:,[5,7,12],:]))
# print (accuracy_score(Y_data_train1,mem5.predict(X_data_train1[:,[5,7,12],:])), "- точность на ТНТ")
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# print ("AANN")#np.arange(0.1,2.5,0.1)
paramGrid6={'up_bound':np.arange(0.1,2.5,0.1),'hidden_layer_sizes':[(1)],'solver':['lbfgs'],'alpha':[(10**(x*(-1))) for x in range(-3,3)]}
model6 =AANN()
mem6 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',model6)]),paramGrid6,2))])

# mem6.fit(X_data_train1[:,[5,7,12],:],Y_data_train1)
# print (mem6.predict(Xnoise[:,[5,7,12],:]))
# print (accuracy_score(Ynoise,mem6.predict(Xnoise[:,[5,7,12],:])), "- точность на веществах")
# print (mem6.predict(X_data_train1[:,[5,7,12],:]))
# print (accuracy_score(Y_data_train1,mem6.predict(X_data_train1[:,[5,7,12],:])), "- точность на ТНТ")
members=[mem1,mem2,mem3,mem4,mem5,mem6]
def Ensemble(members,xTrain,yTrain,xTest,verbose=True):
    results=[]
    for m in members:
        m.fit(xTrain,yTrain)
        predicted = m.predict(xTest)
        predicted=np.array(predicted)
        predicted[predicted<0]=0
        results.append(predicted)
    boundary = len(members)//2

    ensembleResult = [1 if r>boundary else 0 for r in np.sum(results,axis=0)]
    if verbose==True:
        print(np.array(results))
        print (ensembleResult)
    return ensembleResult

ensembleResult = Ensemble(members,X_data_train[:,:,:],Y_data_train,X_data_test[:,:,:])

print (Y_data_test, "естественный шум")
print (ensembleResult, "результат классификации")
print(accuracy_score(Y_data_test,ensembleResult),"Точность ансамбля на веществах")

# ensembleResult = Ensemble(members,X_data_train1[:,[5,7,12],:],Y_data_train1,X_data_train1[:,[5,7,12],:])
#
# print(accuracy_score(Y_data_train1,ensembleResult),"Точность ансамбля на ТНТ")