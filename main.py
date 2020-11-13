from SearcherFile import Searcher
from Loader import RatDataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Preprocessing import Features
from Classifaer import OneClassCV
from sklearn.svm import OneClassSVM
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
Y_data_train = Data[rat[0]][day[0]]['1']['trainEvent']
X_data_train = Data[rat[0]][day[0]]['1']['train']
print (X_data_train.shape, "<<<<<<")
 #OneClassSVM An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors

paramGrid={'nu': [1,2,3,4,5,6], 'gamma': [1,2,3,4,5,6]}

mem1 = Pipeline([('features', Features(function="fft", band=[[1,3]])),('scaler', StandardScaler()),
                 ('oneclasscv', OneClassCV(Pipeline([('oneclasscv',OneClassSVM())]),paramGrid))])
mem1.fit(X_data_train,Y_data_train)