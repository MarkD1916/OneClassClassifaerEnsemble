import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
class Handler():
    def __init__(self,lengthData,date,verbose,window,step):
        self.breathData = []
        self.lengthData = lengthData
        self.date = date
        self.verbose = verbose
        self.eegData =[]
        self.featureData = []
        self.window = window
        self.step = step

    def getAllLength(self,trTeL=True,serL=True,dateL=True):
        sticksTrTe = []
        sticksSer = []
        sticksDate = []
        for ratNum in self.date.keys():
            for date in list(self.date[ratNum]):
                sticksDate_ = []
                for ser in list(self.lengthData[ratNum][date].keys()):
                    trainLength = self.lengthData[ratNum][date][ser]['trainLength']
                    testLength = self.lengthData[ratNum][date][ser]['testLength']
                    if self.verbose==True:
                        print (date,'date')
                        print (ser,'ser')
                        print (self.lengthData[ratNum][date][ser]['trainLength'],'train')
                        print (self.lengthData[ratNum][date][ser]['testLength'],'test')
                    if trTeL==True:
                        sticksTrTe.append([trainLength,testLength])
                    if serL==True:
                        sticksSer.append(sum([trainLength,testLength]))
                    sticksDate_.append(sum([trainLength,testLength]))
                if dateL==True:
                    sticksDate.append(sum(sticksDate_))




        resultLength = [sticksDate,sticksSer,sticksTrTe]
        return resultLength

    def getAllDate(self):
        labelDate = []
        for ratNum in list(self.date.keys()):
            for l in self.date[ratNum]:
                labelDate.append(l + " " + ratNum)
        return labelDate



    def getDataBreath(self):
        allBreath = []
        for ratNum in self.date.keys():
            for date in list(self.date[ratNum]):
                for ser in list(self.lengthData[ratNum][date].keys()):
                    trainBreath = self.lengthData[ratNum][date][ser]['train'][:,-1,:]
                    if len(self.lengthData[ratNum][date][ser]['test'])!=0:
                        testBreath = self.lengthData[ratNum][date][ser]['test'][:,-1,:]
                    #print (np.vstack([trainBreath,testBreath]).shape)
                        allBreath.append(np.vstack([trainBreath,testBreath]))
                    else:
                        allBreath.append(trainBreath)
        self.breathData = np.vstack(allBreath)

    def getDataEeg(self):
        allEeg = []
        for ratNum in self.date.keys():
            for date in list(self.date[ratNum]):
                for ser in list(self.lengthData[ratNum][date].keys()):
                    trainEeg = self.lengthData[ratNum][date][ser]['train'][:,:-1,:]

                    if len(self.lengthData[ratNum][date][ser]['test'])!=0:
                        testEeg = self.lengthData[ratNum][date][ser]['test'][:, :-1, :]
                        allEeg.append(np.vstack([trainEeg,testEeg]))
                    else:
                        allEeg.append(trainEeg)
        self.eegData = np.vstack(allEeg)
        return self.eegData



    def getFreqBreath(self):
        f= self.fft(data = self.breathData,band=[[1,3]])
        return f

    def fft(self,data, freq=1000, band=[], start=1, stop=120, step=10,
            fit=30):  # подготавливает выборку признаков fft
        if len(band) == 0:
            freq_band = [[i, i + fit] for i in range(start, stop, step)]
        else:
            freq_band = sorted(band, key=lambda x: x[0])

        def fft(data, freq, dia_from, dia_to):
            sfreq = freq

            N = data.shape[1]
            spectrum = np.abs((np.fft.rfft(data)))
            freqs = np.fft.rfftfreq(N, 1. / sfreq)
            #print (freqs)
            mask_signal = np.all([[(freqs >= dia_from)], [(freqs <= dia_to)]], axis=0)[0]
            data = np.array(
                [np.mean(spectrum[event][mask_signal])  for event in
                 range(data.shape[0])])
            return data

        start = fft(data, freq, freq_band[0][0], freq_band[0][1])
        if len(freq_band) > 1:
            for i in range(1, len(freq_band)):
                end = fft(data, freq, freq_band[i][0], freq_band[i][1])
                array_stack = np.vstack((start, end))
                start = array_stack
            return array_stack
        else:
            return start

    def maxFreq(self,data,sfreq=1000):
        N = data.shape[1]
        print (N)
        print (data.shape)
        spectrum = np.abs((np.fft.rfft(data)))
        freqs = np.fft.rfftfreq(N, 1. / sfreq)
        print (freqs)
        mask_signal = np.all([[(freqs >= 1)], [(freqs <= 3)]], axis=0)[0]

        data = np.array(
            [np.argmax(spectrum[event][mask_signal]) for event in
             range(data.shape[0])])
        freqArray = np.zeros(data.shape)
        for i,inum in zip(data,range(len(data))):
            freqArray[inum] = freqs[mask_signal][i]
        return freqArray

    def getMaxFreqBreth(self):
        fMax = self.maxFreq(data = self.breathData)
        #print (self.breathData.shape)
        return fMax

    def meanCorr(self, data):
        print (data.shape)
        coef = [
            stats.trim_mean(np.triu(np.asmatrix((np.corrcoef(data[i]))), 1)[np.triu(np.asmatrix((np.corrcoef(data[i]))), 1) != 0],0.2)
            for i in range(len(data))]
        coefMean = np.array(coef)
        return coefMean

    def getMeanCorr(self):
        print(self.eegData.shape)
        corrMean = self.meanCorr(self.eegData[:,:,:])

        return corrMean

    def corrFeature(self,data):

        coef = [
                np.triu(np.asmatrix((np.corrcoef(data[i]))), 1)[np.triu(np.asmatrix((np.corrcoef(data[i]))), 1) != 0]
            for i in range(len(data))]
        coef = np.array(coef)
        return coef

    def getCorrFeature(self):
        corrF = self.corrFeature(self.eegData)

        return corrF

    def transformDataByWindow(self):
        transformedFeature = []
        lenStimul=15000
        for ratNum in self.date.keys():
            for date in list(self.date[ratNum]):
                for ser in list(self.lengthData[ratNum][date].keys()):
                    trainEeg = self.lengthData[ratNum][date][ser]['train']
                    testEeg = self.lengthData[ratNum][date][ser]['test']
                    trainEvent = self.lengthData[ratNum][date][ser]['trainEvent']
                    testEvent = self.lengthData[ratNum][date][ser]['testEvent']
                    trainEegRavel = np.concatenate(trainEeg,axis=1)
                    if len(testEeg)!=0:
                        testEegRavel = np.concatenate(testEeg,axis=1)

                    slice_train = [[i, i + self.window] for i in
                              range(0, trainEegRavel.shape[1] - self.window + self.step, self.step)]
                    if len(testEeg) != 0:
                        slice_test = [[i, i + self.window] for i in
                                       range(0, testEegRavel.shape[1] - self.window + self.step, self.step)]

                    self.lengthData[ratNum][date][ser]['trainLength'] = len(slice_train)
                    if len(testEeg) != 0:
                        self.lengthData[ratNum][date][ser]['testLength'] = len(slice_test)
                    transformedEegDataTrain = []
                    transformedEegDataTest = []
                    transformedEventTrain = []
                    transformedEventTest = []
                    transformedFTrain = []
                    transformedFTest = []

                    for s in slice_train:
                        transformedEventTrain.append(trainEvent[int(s[0] / lenStimul):int(s[1] / lenStimul)])
                        transformedEegDataTrain.append(trainEegRavel[:,s[0]:s[1]])
                        #transformedFTrain.append(trainF[int(s[0] / lenStimul):int(s[1] / lenStimul)])
                    if len(testEeg) != 0:
                        for s in slice_test:
                            transformedEventTest.append(testEvent[int(s[0] / lenStimul):int(s[1] / lenStimul)])
                            transformedEegDataTest.append(testEegRavel[:, s[0]:s[1]])
                        #transformedFTest.append(testF[int(s[0] / lenStimul):int(s[1] / lenStimul)])
                    self.lengthData[ratNum][date][ser]['trainEvent'] = np.array(transformedEventTrain)
                    self.lengthData[ratNum][date][ser]['testEvent'] = np.array(transformedEventTest)
                    self.lengthData[ratNum][date][ser]['train'] = np.array(transformedEegDataTrain)
                    self.lengthData[ratNum][date][ser]['test'] = np.array(transformedEegDataTest)

                    #self.featureData.append(transformedFTrain)
                    #self.featureData.append(transformedFTest)
        print (np.array(transformedEegDataTrain).shape)
        return

    def getCenter(self):
        allEvent=[]
        self.window=int(self.window/15000)
        self.step=int(self.step/15000)

        for ratNum in self.date.keys():
            for date in list(self.date[ratNum]):
                for ser in list(self.lengthData[ratNum][date].keys()):
                    trainEvent = self.lengthData[ratNum][date][ser]['trainEvent']
                    testEvent = self.lengthData[ratNum][date][ser]['testEvent']
                    allEvent.append(trainEvent)
                    allEvent.append(testEvent)
        slice = [[i, i + self.window] for i in
                              range(0, np.concatenate(allEvent).shape[0] - self.window + self.step, self.step)]

        allCenter=[]

        for s in slice:
            windowCenter=[]
            for nClass,sliceE in zip(np.unique(np.concatenate(allEvent)),np.concatenate(allEvent)):
                #print (sliceE)
                #print (nClass)
                kmeans = KMeans(n_clusters=1, random_state=0).fit(self.featureData[s[0]:s[1]])

                windowCenter.append(kmeans.cluster_centers_)
            allCenter.append(np.mean(windowCenter))


        return allCenter
    