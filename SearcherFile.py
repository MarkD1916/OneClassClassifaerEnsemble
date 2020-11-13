# coding=utf-8
import numpy as np
import os
from glob import glob
import json
import re

class Searcher():
    def __init__(self,mainDir,rat,year,verbose,metaDataByRatNum,day,drugs):
        self.mainDir = mainDir
        self.rat = rat
        self.year = year
        self.verbose = verbose
        self.sessionDirs = {}
        self.fileMetaData = {}
        self.metaDataByRatNum = metaDataByRatNum
        self.day=day
        self.drugs=drugs


    def fileSearch(self):

        for rat in self.rat:
            top_level_name = np.array(
                [name for name in os.listdir(self.mainDir) if
                 os.path.isdir(os.path.join(self.mainDir, name))])  # внешние папки на диске

            year_level = self.mainDir + '/' + top_level_name[top_level_name == self.year][0]  # не очень удачная конструкция, нельзя выбрать несколько лет сразу же

            def find_setting(year_level, name):  # поиск файла настроек в указанной папке
                for root, dirs, files in os.walk(year_level):
                    if name in files:
                        return os.path.join(root, name)
                print ('no config file')
                return

            with open(find_setting(year_level, 'setting' + '_' + self.year + '.json')) as json_data_file:
                all_settings = json.load(json_data_file)

            subdir_mask = all_settings.get('data').get('subdir_mask').format(rat)
            sessions_names = [name for name in glob(os.path.join(year_level, subdir_mask))]
            sessions_names = sorted(sessions_names)
            if len(sessions_names)!=0:
                self.sessionDirs[rat] = sessions_names
            else:
                if self.verbose==True:
                    print ("Крыса "+rat + " не найдено экспериментов")
        if self.verbose==True:
            print (self.sessionDirs)
        return

    def serchFileByDrugs(self):

        top_level_name = np.array(
            [name for name in os.listdir(self.mainDir) if
             os.path.isdir(os.path.join(self.mainDir, name))])  # внешние папки на диске

        year_level = self.mainDir + '/' + top_level_name[top_level_name == self.year][0]  # не очень удачная конструкция, нельзя выбрать несколько лет сразу же

        sessions_names = []
        rat=[]
        for expDay in os.walk(year_level).__next__()[1]:
            try:

                if '_' not in expDay.split('-')[1]:
                    #print(expDay.split('-'))
                    #print(expDay)
                    rat.append(expDay.split('-')[1])
                    sessions_names.append([os.path.join(year_level, expDay)])
            except:
                pass



        for i,r in zip(sessions_names,rat):

            self.sessionDirs[r] = i
        #print (self.sessionDirs)
        self.rat = rat
        return

    def parsJson(self): #

        for ratNumber in self.rat: # перебор крыс по номерам
            metaDataByRatDate = {}
            for sessionDir in self.sessionDirs[ratNumber]: # перебор крыс по датам

                sessionDate = re.findall('[0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]',
                                         os.path.basename(os.path.normpath(sessionDir)))[0]

                self.fileMetaData = {}
                if sessionDate in self.day or len(self.day)==0:
                    for file in sorted(glob(os.path.join(sessionDir, 'settings*.json'))): # перебор крыс по сериям
                        with open(file) as json_data_file:
                            try:
                                serNumber = re.findall('\d',os.path.basename(os.path.normpath(file)))[0]
                            except IndexError:
                                serNumber = '1'

                            self.fileMetaData[serNumber]={}
                            sessionSettings = json.load(json_data_file)
                            self.fileMetaData[serNumber]["sessionDirName"] = sessionDir
                            self.fileMetaData[serNumber]["trainFileName"] = sessionSettings.get('data').get('train_names')[0]
                            self.fileMetaData[serNumber]["testFileName"] = sessionSettings.get('data').get('test_names')[0]
                            self.fileMetaData[serNumber]["sampleRate"] = sessionSettings.get('data').get('final_sample_rate')
                            self.fileMetaData[serNumber]["trainCategs"] = sessionSettings.get('categ').get('categs')
                            self.fileMetaData[serNumber]["testCategs"] = sessionSettings.get('categ').get('test_categs')
                            self.fileMetaData[serNumber]["numCh"] = sessionSettings.get('channel').get('eeg_idxs')
                            self.fileMetaData[serNumber]["drugsName"] = sessionSettings.get('categ').get('names')

                    metaDataByRatDate[sessionDate] = self.fileMetaData
                self.metaDataByRatNum[ratNumber] = metaDataByRatDate
        if self.verbose==True:
            print (self.metaDataByRatNum)
        return self.metaDataByRatNum



