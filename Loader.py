# coding=utf-8
from SearcherFile import Searcher
import data
from data import load_data, load_file
import os
import numpy as np
from itertools import chain, zip_longest

class RatDataLoader(Searcher):



    def getRatDate(self):
        allDate = {}
        for ratNum in self.rat:
            allDate[ratNum] = list(self.metaDataByRatNum[ratNum].keys())
        return allDate



    def loadData(self):
        eegDataRatNum = {}
        for ratNum in self.rat:
            eegDataDate = {}
            for session in list(self.metaDataByRatNum[ratNum].keys()):
                eegDataSer = {}
                for ser in self.metaDataByRatNum[ratNum][session].keys():

                    eegData={}
                    numCh = self.metaDataByRatNum[ratNum][session][ser]["numCh"]
                    trainFileName = self.metaDataByRatNum[ratNum][session][ser]["trainFileName"]
                    testFileName = self.metaDataByRatNum[ratNum][session][ser]["testFileName"]
                    trainCategs = self.metaDataByRatNum[ratNum][session][ser]["trainCategs"]
                    testCategs = self.metaDataByRatNum[ratNum][session][ser]["testCategs"]
                    dirName = self.metaDataByRatNum[ratNum][session][ser]["sessionDirName"]
                    sampleRate = self.metaDataByRatNum[ratNum][session][ser]["sampleRate"]

                    data.NUM_EEG_CHANNELS = len(numCh)
                    data.NUM_ALL_CHANNELS = data.NUM_EEG_CHANNELS + 2
                    data.IDXS_EEG_CHANNELS = slice(0, data.NUM_EEG_CHANNELS)
                    data.IDX_BREATH_CHANNEL = data.NUM_EEG_CHANNELS
                    data.ALTER_STIM_CHANNEL = True
                    piece = 5
                    if trainFileName[0].split(".")[-1] == 'plx':

                        data.DECIMATE = 10
                    if trainFileName[0].split(".")[-1] == 'dat':
                        data.DECIMATE = 1
                    data.DECIMATE=1

                    data.SAMPLE_RATE = 1000
                    data.LEN_STIMUL_SECS = 5
                    data.LEN_STIMUL = piece * sampleRate
                    data.BEFORE_STIMUL = data.LEN_STIMUL
                    print (data.BEFORE_STIMUL)
                    train = load_data('odor',
                                      dir=dirName,
                                      series=[trainFileName],
                                      categs=trainCategs, take_signal='stimul', take_ecg=False, det_breath=2, take_air=False,
                                      piece=5, take_spikes=False, verbose=False,add_breath=False)

                    print (testFileName)
                    if testFileName!="":
                        test = load_data('odor',
                                      dir=dirName,
                                      series=[testFileName],
                                      categs=testCategs, take_signal='stimul', take_ecg=False, det_breath=2, take_air=False,
                                      piece=5, take_spikes=False, verbose=False,add_breath=False)
                        testData = test[0].get('').get('').get(testFileName)[0]
                        testEvent = test[0].get('').get('').get(testFileName)[1]
                    else:
                        testData=[]
                        testEvent=[]
                    trainEvent = train[0].get('').get('').get(trainFileName)[1]
                    print(trainEvent)
                    trainData = train[0].get('').get('').get(trainFileName)[0]
                    print (trainData.shape)

                    trainTestCatages = [trainCategs, testCategs]
                    if testFileName != "":
                        for i, j in zip(trainTestCatages[0], trainTestCatages[1]): # переобозначаю метки, если клапана менялись
                            if i != j:
                                testEvent[testEvent == j] = i * -1
                        testEvent[testEvent < 0] = testEvent[testEvent < 0] * -1
                    #testDataEeg = testData[:,:-1,5000:]
                    #testDataBreath = testData[:,[-1],:5000]
                    #print (testDataBreath[23])
                    #testData = np.concatenate([testDataEeg,testDataBreath],axis=1)
                    #print (testData[23,-1,:])
                    #trainDataEeg = trainData[:, :-1, 5000:]
                    #trainDataBreath = trainData[:, [-1], :5000]
                    #print (trainDataEeg.shape,trainDataBreath.shape)
                    #trainData = np.concatenate([trainDataEeg, trainDataBreath],axis=1)
                    eegData['test'] = testData
                    eegData['train'] = trainData[:,:,:]
                    eegData['trainEvent'] = trainEvent
                    eegData['testEvent'] = testEvent
                    eegData['trainLength'] = len(trainEvent)
                    eegData['testLength'] = len(testEvent)

                    eegDataSer[ser]=eegData
                eegDataDate[session]=eegDataSer
            eegDataRatNum[ratNum]=eegDataDate

        return eegDataRatNum

    def loadRatDrugsList(self):
        ratNamesByDrugs=[]
        for ratNum in self.rat:

            for session in list(self.metaDataByRatNum[ratNum].keys()):

                for ser in self.metaDataByRatNum[ratNum][session].keys():
                    #print (self.metaDataByRatNum[ratNum][session][ser]['drugsName'])
                    if self.metaDataByRatNum[ratNum][session][ser]['drugsName'] is not None:
                        for d in self.metaDataByRatNum[ratNum][session][ser]['drugsName']:
                                #print (session,ratNum)
                                if d in self.drugs:
                                    ratNamesByDrugs.append(session+'-'+ratNum+'-'+ser)

        return ratNamesByDrugs

    def loadFile(self):
        for ratNum in self.rat:
            for session in list(self.metaDataByRatNum[ratNum].keys()):
                for ser in self.metaDataByRatNum[ratNum][session].keys():
                    numCh = self.metaDataByRatNum[ratNum][session][ser]["numCh"]
                    trainFileName = self.metaDataByRatNum[ratNum][session][ser]["trainFileName"]
                    sampleRate = self.metaDataByRatNum[ratNum][session][ser]["sampleRate"]

                    data.NUM_EEG_CHANNELS = len(numCh)
                    data.NUM_ALL_CHANNELS = data.NUM_EEG_CHANNELS + 2
                    data.IDXS_EEG_CHANNELS = slice(0, data.NUM_EEG_CHANNELS)
                    data.IDX_BREATH_CHANNEL = data.NUM_EEG_CHANNELS
                    data.ALTER_STIM_CHANNEL = False
                    piece = 5
                    if trainFileName[0].split(".")[-1] == 'plx':
                        data.DECIMATE = 10
                    if trainFileName[0].split(".")[-1] == 'dat':
                        data.DECIMATE = 1
                    data.DECIMATE = 1

                    data.SAMPLE_RATE = sampleRate
                    data.LEN_STIMUL_SECS = piece
                    data.LEN_STIMUL = piece * sampleRate
                    data.BEFORE_STIMUL = data.LEN_STIMUL
                    sessionDirName = self.metaDataByRatNum[ratNum][session][ser]["sessionDirName"]
                    trainFileName = self.metaDataByRatNum[ratNum][session][ser]["trainFileName"]
                    testFileName = self.metaDataByRatNum[ratNum][session][ser]["testFileName"]
                    s = '.'

                    trainData = load_file(os.path.join(sessionDirName,trainFileName))
                    testData = load_file(os.path.join(sessionDirName,testFileName))

                    NSamplingsTrain = trainData[0][:,0].shape[0]
                    NSamplingsTest = testData[0][:, 0].shape[0]
                    #print(trainData[0].shape)
                    #print (trainData[2][trainData[2]==1].shape)
                    concatSignalTrain = np.array(
                        ([ele for ele in chain.from_iterable(
                            zip_longest(trainData[0][:],trainData[1][0],
                                        trainData[2])) if ele is not None]))
                    # import matplotlib.pyplot as plt
                    # plt.plot(trainData[2])
                    # plt.show()
                    #print (np.unique(trainData[2]))
                    newTrainArray = []
                    for i in concatSignalTrain:
                        if isinstance(i,np.ndarray):
                            for j in i:
                                newTrainArray.append(j)
                        else:
                            newTrainArray.append(i)
                    mean = 30
                    std = 1250
                    num_samples = testData[0][:, [8]].shape[0]
                    samples = np.random.normal(mean, std, size=num_samples)
                    mean = 15
                    std = 2239
                    num_samples = testData[0][:, [8]].shape[0]
                    samples2 = np.random.normal(mean, std, size=num_samples)
                    mean = 26
                    std = 189
                    num_samples = testData[0][:, [8]].shape[0]
                    samples3 = np.random.normal(mean, std, size=num_samples)
                    newCh1 =np.zeros(testData[0][:, [8]].shape)
                    newCh2 = np.zeros(testData[0][:, [8]].shape)
                    newCh3 = np.zeros(testData[0][:, [8]].shape)
                    for i,j,num in zip(np.array(testData[0][:, [8]]),samples,range(len(samples))):
                        newCh1[num]=i+j
                    for i,j,num in zip(np.array(testData[0][:, [8]]),samples2,range(len(samples2))):
                        newCh2[num]=i+j
                    for i,j,num in zip(np.array(testData[0][:, [8]]),samples3,range(len(samples3))):
                        newCh3[num]=i+j
                    newChannel = np.concatenate([testData[0][:, [0,1,2,3,4,5,6,7,8]],newCh1,
                                                 newCh2,newCh3,testData[0][:, [12,13,14]]],axis=1)
                    #print (newChannel[:20,8:16])

                    concatSignalTest = np.array(
                        ([ele for ele in chain.from_iterable(
                            zip_longest(newChannel, testData[1][0],
                                        testData[2])) if ele is not None]))

                    newTestArray = []
                    for i in concatSignalTest:
                        if isinstance(i, np.ndarray):
                            for j in i:
                                newTestArray.append(j)
                        else:
                            newTestArray.append(i)
                    newTrainFileName = s.join(trainFileName.split('.')[:-1])+'-'+'art'+'.dat'
                    newTestFileName = s.join(testFileName.split('.')[:-1]) + '-' + 'art' + '.dat'
                    np.array(newTrainArray).astype('int16').tofile(newTrainFileName)
                    np.array(newTestArray).astype('int16').tofile(newTestFileName)

                    def create_inf(path_to_res, nNSamplings):
                        with open(path_to_res + '.inf', 'w') as f:
                            f.write(
                                "[Version]\nDevice=Smell-ADC\n\n[Object]\nFILE=\"\"\n\n[Format]\nType=binary\n\n[Parameters]\nNChannels={0}\nNSamplings={1}\nSamplingFrequency={2}\n\n[ChannelNames]\n{3}"
                                    .format(17, nNSamplings, 1000,
                                            "\n".join(map(lambda x: str(x) + "=" + str(x + 1), range(17)))))
                    create_inf(s.join(trainFileName.split('.')[:-1])+"-art",NSamplingsTrain)
                    create_inf(s.join(testFileName.split('.')[:-1])+"-art", NSamplingsTest)
        return


