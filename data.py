# coding=utf-8

import os
import sys
if sys.version[0] == 2:
    import cPickle as pickle
else:
    import pickle
import numpy as np
import pandas as pd
from glob import glob
from scipy import signal
from sklearn import preprocessing
import array
from neo import PlexonIO, SpikeTrain
import h5py
import mne
from collections import Counter
#from common.utils import *
#from eeg.utils import *

# подсчет крыс и экспериментальных серий для них
MIN_NUM_SERIES = 2#10 # не менее 5 т.к. при 20% в валид. выборку попадает 0 серий
MAX_NUM_SERIES = 20
MIN_NUM_DAYS = 2

def get_rat_file_names(dir_mask, file_ext):
    # выборка крыс
    dir_names = glob('./%s'%dir_mask)#glob('./*')
    # вырезаем символы между двумя дефисами
    rat_names = []
    for dir_name in dir_names:
        #if (dir_name.find('-') != -1) and (dir_name.find('control') == -1) and (dir_name.find('problem') == -1):
        if (dir_name.count('.') == 1) and len(dir_name.split('-')) > 1: # у файла 2
            rat_names.append(dir_name.split('-')[1])
    rat_names = set(rat_names)#rat_names = list(set(rat_names))

    # выборка серий и запоминание директорий
    rats_series = dict()
    rats_dirs = dict()
    print('rat\t\tday_count\t\tseries_count')
    for rat_name in rat_names:
        #file_names = glob('./%s%s*/*.%s' %(dir_mask,rat_name,file_ext))#glob('./*%s*/*.%s' %(rat_name,file_ext))#file_names = glob('./*%s*/*.txt' %(rat_name))
        if dir_mask == '*odor*':
            file_names = glob('./*-%s-%s/*.%s' %(rat_name,dir_mask,file_ext))#file_names = glob('./*%s*/*.txt' %(rat_name))
            series = []
            days = []
            for file_name in file_names:
                if (file_name.find('control') == -1) and (file_name.find('problem') == -1):
                    series.append(file_name)
                    days.append(file_name.split('-')[0])
        elif dir_mask == '*onco*':
            series_0 = glob('./*-%s-%s/0/*.%s' %(rat_name,dir_mask,file_ext))
            series_1 = glob('./*-%s-%s/1/*.%s' %(rat_name,dir_mask,file_ext))
            min_count = min(len(series_0), len(series_1))
            series = series_0[:min_count] + series_1[:min_count] # сделаем series минимальной из 0 и 1 папок
            days = []
            for file_name in series:
                days.append(file_name.split('-')[0])#days = ['onco']
        #elif dir_mask == '*anes*':
        #    series = glob('./*%s-anes*/*.%s' % (rat_name, file_ext))
        #    days = ['anes']
        else:
            raise Exception('Undefined dir mask!')

        if (len(series) >= MIN_NUM_SERIES) and (len(series) <= MAX_NUM_SERIES) and (len(set(days)) >= MIN_NUM_DAYS):
            rats_series[rat_name] = series
            rats_dirs[rat_name] = glob('./*-%s-%s' %(rat_name,dir_mask))#glob('./*%s*' %(rat_name)) # запоминаем директории крысы
            rat_name = rat_name + '*' # отметка что крыса отобрана
        print('%s\t\t%d\t\t%d' %(rat_name, len(set(days)), len(series)))
    print('Selected %d rats with days_count >= %d, series_count >= %d and <= %d\n' %(len(rats_series), MIN_NUM_DAYS, MIN_NUM_SERIES, MAX_NUM_SERIES))
    return rats_series, rats_dirs

#INIT_SAMPLE_RATE = 1000
SAMPLE_RATE = 1000
LEN_STIMUL = 3000
LEN_STIMUL_SECS = 3

DECIMATE = 10#5#1
AIR_LABEL = 64

NUM_EEG_CHANNELS = 8
NUM_ALL_CHANNELS = 10
ADD_STIM_CHANNEL = False
ALTER_STIM_CHANNEL = False

IDXS_EEG_CHANNELS = slice(0,NUM_EEG_CHANNELS)
IDX_BREATH_CHANNEL = 8
IDX_ECG_CHANNEL = 9
SPIKE_CHANNELS = []#{}

def load_file(file_name, take_ecg=False, take_spikes=False, take_breath=True): # по требованию (ADD_STIM_CHANNEL) добавляется дополнительный канал (для стимулов) (NUM_ALL_CHANNELS+1)
    breath = None
    ecg = None
    spikes = None # данные
    ext = file_name.split('.')[-1]
    if ext == 'txt':
        df = pd.read_csv(file_name, sep=' ', header=None)
        last_chan = NUM_ALL_CHANNELS - 1
        if ADD_STIM_CHANNEL: # запись без стимула, последний канал - дыхание, формально раздваиваем его - увеличивается количество каналов на 1
            return np.float32(df.values[::DECIMATE, :last_chan]), np.int16(df.values[::DECIMATE, last_chan]), np.int16(df.values[::DECIMATE, last_chan])  # просто повторяем последний канал, чтобы формально был канал стимула
        else: # считается, что последний канал - стимулы, предпоследний - дыхание
            return np.float32(df.values[::DECIMATE, :last_chan-1]), np.int16(df.values[::DECIMATE, last_chan-1]), np.int16(df.values[::DECIMATE, last_chan])
    elif ext == 'dat':
        with open(file_name, 'rb') as file_handler:
            arr = array.array('h') #int16 - signed short int
            file_size = os.path.getsize(file_name)
            num_values = file_size//2
            num_samples = file_size//2//NUM_ALL_CHANNELS
            arr.fromfile(file_handler, num_values)
            arr = np.asarray(arr,np.float32).reshape((num_samples,NUM_ALL_CHANNELS))
            #return np.int16(arr[::DECIMATE,:8]), np.int16(arr[::DECIMATE,8]), np.int16(arr[::DECIMATE,9])
            last_chan = NUM_ALL_CHANNELS - 1
            if ADD_STIM_CHANNEL: # запись без стимула, последний канал - дыхание, формально раздваиваем его - увеличивается количество каналов на 1
                return arr[::DECIMATE, IDXS_EEG_CHANNELS], (arr[::DECIMATE, last_chan],ecg), arr[::DECIMATE, last_chan] #:last_chan # просто повторяем последний канал, чтобы формально был канал стимула
            elif ALTER_STIM_CHANNEL: # импульсные метки переделать в пролонгированные стимулы
                # читаем sampling_rate
                found = False

                for line in open(file_name[:-3] + 'inf', 'r'):

                    if line.find('SamplingFrequency') != -1:
                        sampling_rate = int(line[line.find('=') + 1:])
                        found = True
                        break
                if not found: raise Exception('Import .dat file error! No .inf file!')
                stimul_ = arr[:, last_chan]
                stimul_idxs = np.where(stimul_ != 0)[0]
                # заполняем стимулы
                stimul = np.zeros(stimul_.shape[0], dtype=np.int16)
                len_stimul = sampling_rate * LEN_STIMUL_SECS #INIT_SAMPLE_RATE
                for stimul_idx in stimul_idxs:
                    if stimul_[stimul_idx] != AIR_LABEL:
                        stimul[stimul_idx:stimul_idx+len_stimul] = stimul_[stimul_idx]
                # заполняем межстимулы
                stimul[stimul == 0] = AIR_LABEL
                #from collections import Counter
                #count = Counter(stimul)#[::DECIMATE]
                return arr[::DECIMATE, IDXS_EEG_CHANNELS], (arr[::DECIMATE, last_chan-1],ecg,spikes), stimul[::DECIMATE] #:last_chan-1
            else: # считается, что последний канал - стимулы, предпоследний - дыхание
                return arr[::DECIMATE, IDXS_EEG_CHANNELS], (arr[::DECIMATE, last_chan-1],ecg,spikes), arr[::DECIMATE, last_chan] #:last_chan-1
    elif (ext == 'plx') | (ext == 'hdf'):
        def load_plx(breath, ecg, spikes):
            segm = PlexonIO(file_name).read_segment()#bool(take_spikes)load_spike_waveform=False

            # загрузка ЭЭГ каналов
            # из общего набора (128 каналов) выбираем заполненные каналы сигнала
            signal = [asig for asig in segm.analogsignals if len(asig) > 0]
            signal = np.asarray(np.multiply(np.concatenate(signal, axis=1), 1000), np.float32)  # переводим из миливольт в микровольты
            # из общего набора (30 каналов) отбираем длинные массивы событий и из них отбираем несколько равных по длине
            all_valves_events = [ev for ev in segm.events if len(ev) > 7]#7
            if len(all_valves_events) > 0:
                # отбираем массивы равные первому
                len_first_arr = len(all_valves_events[0])
                valves_events = [ev for ev in all_valves_events if len(ev) == len_first_arr]
                # если таких массивов один, т.е. первый был массив включения воздуха, то сравниваем с длинной второго
                # if len(valves_events) == 1:
                #    len_second_arr = len(all_valves_events[1])
                #    valves_events = [ev for ev in all_valves_events if len(ev) == len_second_arr]
                # альтернативная реализация, устойчивая к разному количеству стимулов в каналах
                # если таких массивов один, т.е. первый был массив включения воздуха, то просто берем все кроме первого
                if len(valves_events) == 1:
                    valves_events = all_valves_events[1:]
                else:
                    raise Exception('Import .plx file error!')
            else:
                valves_events = []
            sampling_rate = int(segm.analogsignals[0].sampling_rate)#64

            # загрузка Спайковых каналов
            if take_spikes:
                global SPIKE_CHANNELS
                spikes = segm.spiketrains
                #print([(st.annotations['Name'], st.size) for st in spikes])
                # if take_spikes == 'sum':
                #     spike_idxs = range(1, NUM_EEG_CHANNELS + 1)
                #     cur_name = spikes[0].annotations['Name']
                #     present = [int(cur_name[-2:])]
                #     res = {cur_name: spikes[0]}
                #     for i in range(1, len(spikes)):
                #         if spikes[i].annotations['Name'] == cur_name:
                #             res[cur_name].t_stop, spikes[i].t_stop = [max(res[cur_name].t_stop, spikes[i].t_stop)] * 2
                #             res[cur_name] = res[cur_name].merge(spikes[i])
                #         else:
                #             cur_name = spikes[i].annotations['Name']
                #             if int(cur_name[-2:]) > NUM_EEG_CHANNELS:
                #                 break
                #             present.append(int(cur_name[-2:]))
                #             res[cur_name] = spikes[i]
                #     spikes = [res[key] for key in sorted(res.keys())]
                #     missing = list(set(spike_idxs) - set(present))
                #     for nn in sorted(missing):
                #         name = 'sig0' + '0' * int(nn < 10) + str(nn)
                #         spikes.insert(nn - 1, SpikeTrain(times=[0.001], units='sec', t_stop=1., Name=name))
                #
                # elif take_spikes == 'sort':
                #warnings.warn('take_spikes=\'sort\' works correctly only with neo 0.7.2, not 0.5.2., other versions were not tested', UserWarning)
                # neo v. 0.7.2
                chs_ns = {str(id): i for i, id in enumerate(tuple([st.annotations['id'] for st in spikes]))
                          if ((not id.endswith('#0')) and (int(id.split('#')[0][2:]) <= NUM_EEG_CHANNELS))}
                spikes = [spikes[i] for i in chs_ns.values()]
                if SPIKE_CHANNELS == []:
                    SPIKE_CHANNELS = chs_ns.keys()
                else:
                    missing, extra = [], []
                    positions = range(len(SPIKE_CHANNELS))
                    for el in SPIKE_CHANNELS:
                        if el not in chs_ns.keys():
                            missing.append(el)
                    for el in chs_ns.keys():
                        if el not in SPIKE_CHANNELS:
                            extra.append(el)
                    to_del = []
                    for i, st in enumerate(spikes):
                        if st.annotations['id'] in extra:
                            to_del.append(i)
                    for idx in sorted(to_del, reverse=True):
                        del spikes[idx]
                    for pos, el in enumerate(sorted(missing)):
                        spikes.insert(positions[pos], SpikeTrain(times=[0.001], units='sec', t_stop=1., id=el))

                len_signal = signal[::DECIMATE].shape[0]
                spike_arr = np.zeros((len_signal, len(spikes)), dtype=np.int8)
                for i in range(len(spikes)):
                    neuron = (np.array(spikes[i]) * (sampling_rate / DECIMATE)).astype(int).flatten()
                    neuron = neuron[neuron < len_signal]
                    spike_arr[neuron, i] = 1

            # формируем канал стимулов
            num_cutoffs = signal.shape[0]
            stimul = np.zeros(num_cutoffs, dtype=np.int16)
            for i_valve, events in enumerate(valves_events):  # по клапанам
                for event in events:  # по событиям
                    begin = int(event * sampling_rate)
                    stimul[begin:begin + LEN_STIMUL_SECS * sampling_rate] = 2 ** i_valve
                # for i, event in enumerate(events):  # по событиям
                #     begin = int(event * sampling_rate)
                #     stimul[begin:begin + LEN_STIMUL_SECS * sampling_rate] = i%2+1
                # break

            # заполним межстимульные интервалы AIR_LABEL
            stimul[stimul == 0] = AIR_LABEL
            # сохраним в HDF5
            with h5py.File(file_hdf, 'w') as f:
                eeg = np.asarray(f.create_dataset('eeg', data=signal[::DECIMATE, IDXS_EEG_CHANNELS]), np.float32)  #:NUM_EEG_CHANNELS
                stimul = np.asarray(f.create_dataset('stimul', data=stimul[::DECIMATE]), np.int16)
                if take_breath:
                    breath = np.asarray(f.create_dataset('breath', data=signal[::DECIMATE, IDX_BREATH_CHANNEL]), np.float32)
                if take_ecg:
                    ecg = np.asarray(f.create_dataset('ecg', data=signal[::DECIMATE, IDX_ECG_CHANNEL]), np.float32)
                if take_spikes:
                    spikes_ds = f.create_dataset('spikes', data=spike_arr)
                    # сохранение идентификаторов спайковых каналов
                    spikes_ds.attrs['neuron_id'] = sorted(chs_ns.keys())
                    spikes = np.asarray(spikes_ds, np.int8)

            return eeg, (breath, ecg, spikes), stimul

        # загрузим данные из HDF5, если он есть
        file_hdf = file_name[:-3] + 'hdf'
        if os.path.exists(file_hdf):
            #with h5py.File(file_hdf, 'r') as f:
            f = h5py.File(file_hdf, 'r')
            eeg = np.asarray(f['eeg'], np.float32)
            stimul = np.asarray(f['stimul'], np.int16)
            if take_breath or take_ecg or take_spikes:
                try:
                    if take_breath:
                        breath = np.asarray(f['breath'], np.float32)
                    if take_ecg:
                        ecg = np.asarray(f['ecg'], np.float32)
                    if take_spikes:
                        spikes = np.asarray(f['spikes'], np.float32)
                        ids = f['spikes'].attrs['neuron_id']
                        global SPIKE_CHANNELS
                        if SPIKE_CHANNELS == []:#dict():
                            SPIKE_CHANNELS = list(ids)
                        else:
                            #extra = np.array(map(lambda (i, x): i if x in SPIKE_CHANNELS else None, enumerate(sorted(ids)))) tuple parameter unpacking is not supported in python 3
                            def ix4extra(ix):
                                i, x = ix
                                return i if x in SPIKE_CHANNELS else None
                            extra = np.array(map(ix4extra, enumerate(sorted(ids))))
                            extra = np.argwhere(extra == None).flatten()
                            #missing = np.array(map(lambda (i, x): i if x in ids else None, enumerate(sorted(SPIKE_CHANNELS)))) tuple parameter unpacking is not supported in python 3
                            def ix4missing(ix):
                                i, x = ix
                                return i if x in ids else None
                            missing = np.array(map(ix4missing, enumerate(sorted(SPIKE_CHANNELS))))
                            missing = np.argwhere(missing == None).flatten()
                            cols = np.arange(spikes.shape[1])
                            cols = cols[cols != extra] if len(extra)>0 else cols
                            spikes = np.squeeze(spikes[:, cols])
                            for i in range(len(missing)):
                                np.insert(spikes, missing[i], 0, axis=1)
                    f.close()
                except:
                    f.close()
                    eeg, (breath, ecg, spikes), stimul = load_plx(breath, ecg, spikes) # заново загружаем файл
        else:
            eeg, (breath, ecg, spikes), stimul = load_plx(breath, ecg, spikes)

        return eeg, (breath, ecg, spikes), stimul


#EXCLUDE_STIMUL_VALUES = [0,96] # 65-воздух, 66-ТНТ, 68-ТНТТабак, 72-Кока, 80-КокаТабак
INCLUDE_STIMUL_VALUES = [66,72] # 65-воздух, 66-ТНТ, 68-ТНТТабак, 72-Кока, 80-КокаТабак, 85-неизвестное
#INCLUDE_STIMUL_VALUES = [66,68,72,80,85]
BEFORE_STIMUL = 3000

# extract eeg under stimuls
def extract_stimuls(eeg, bio, stimul, before=False, after=False, continuous=True, take_air=False, take_ecg=False, take_spikes=False, cut=True):
    breath, ecg, spikes = bio
    # забор воздуха делаем перед стимулом первого клапана (для равенства кол-ва примеров)
    if take_air:
        # сдвигаем на BEFORE_STIMUL раньше
        stimul_air = np.delete(stimul, np.s_[:BEFORE_STIMUL])
        stimul_air = np.append(stimul_air, np.zeros((BEFORE_STIMUL)))
        # берем только первый клапан
        mask_air = stimul_air == INCLUDE_STIMUL_VALUES[0]

    if before or after:
        # сдвигаем на BEFORE_STIMUL раньше
        if before:
            stimul_before = np.delete(stimul, np.s_[:BEFORE_STIMUL])
            stimul_before = np.append(stimul_before, np.zeros((BEFORE_STIMUL)))
        else:
            stimul_before = np.delete(stimul, np.s_[-BEFORE_STIMUL:])
            stimul_before = np.append(np.zeros((BEFORE_STIMUL)), stimul_before)

        if len(INCLUDE_STIMUL_VALUES) == 6:
            if continuous:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])], [(stimul == INCLUDE_STIMUL_VALUES[4])], [(stimul == INCLUDE_STIMUL_VALUES[5])],
                               [(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])], [(stimul_before == INCLUDE_STIMUL_VALUES[3])], [(stimul_before == INCLUDE_STIMUL_VALUES[4])], [(stimul_before == INCLUDE_STIMUL_VALUES[5])]], axis=0)[0] # [0] because of 2d result array
            else:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])], [(stimul == INCLUDE_STIMUL_VALUES[4])], [(stimul == INCLUDE_STIMUL_VALUES[5])]], axis=0)[0]
                mask_before = np.any([[(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])], [(stimul_before == INCLUDE_STIMUL_VALUES[3])], [(stimul_before == INCLUDE_STIMUL_VALUES[4])], [(stimul_before == INCLUDE_STIMUL_VALUES[5])]], axis=0)[0]
        elif len(INCLUDE_STIMUL_VALUES) == 5:
            if continuous:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])], [(stimul == INCLUDE_STIMUL_VALUES[4])],
                               [(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])], [(stimul_before == INCLUDE_STIMUL_VALUES[3])], [(stimul_before == INCLUDE_STIMUL_VALUES[4])]], axis=0)[0] # [0] because of 2d result array
            else:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])], [(stimul == INCLUDE_STIMUL_VALUES[4])]], axis=0)[0]
                mask_before = np.any([[(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])], [(stimul_before == INCLUDE_STIMUL_VALUES[3])], [(stimul_before == INCLUDE_STIMUL_VALUES[4])]], axis=0)[0]
        elif len(INCLUDE_STIMUL_VALUES) == 4:
            if continuous:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])],
                               [(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])], [(stimul_before == INCLUDE_STIMUL_VALUES[3])]], axis=0)[0] # [0] because of 2d result array
            else:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])]], axis=0)[0]
                mask_before = np.any([[(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])], [(stimul_before == INCLUDE_STIMUL_VALUES[3])]], axis=0)[0]
        elif len(INCLUDE_STIMUL_VALUES) == 3:
            if continuous:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])],
                               [(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])]], axis=0)[0] # [0] because of 2d result array
            else:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])]], axis=0)[0]
                mask_before = np.any([[(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])], [(stimul_before == INCLUDE_STIMUL_VALUES[2])]], axis=0)[0]
        elif len(INCLUDE_STIMUL_VALUES) == 2:
            if continuous:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])],
                               [(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])]], axis=0)[0] # [0] because of 2d result array
            else:
                mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])]], axis=0)[0]
                mask_before = np.any([[(stimul_before == INCLUDE_STIMUL_VALUES[0])], [(stimul_before == INCLUDE_STIMUL_VALUES[1])]], axis=0)[0]
        elif len(INCLUDE_STIMUL_VALUES) == 1:
            if continuous:
                mask = stimul == INCLUDE_STIMUL_VALUES[0]
            else:
                mask = stimul == INCLUDE_STIMUL_VALUES[0]
                mask_before = stimul_before == INCLUDE_STIMUL_VALUES[0]
        if not continuous:
            #assert LEN_STIMUL == BEFORE_STIMUL
            sample_size = LEN_STIMUL

            if take_air:
                mask_before = np.add(mask_before, mask_air)

            eeg_before = eeg[mask_before]
            breath_before = breath[mask_before]
            stimul_before = stimul[mask_before]
            if take_ecg:
                ecg_before = ecg[mask_before]
            if take_spikes:
                spikes_before = spikes[:, mask_before]
            a = float(len(stimul_before)) % BEFORE_STIMUL#sample_size #LEN_STIMUL #np.ceil(float(len(stimul)) / LEN_STIMUL)
            n = len(stimul_before) // BEFORE_STIMUL#sample_size #LEN_STIMUL
            if a != 0: # обрежем массив
                b = n*BEFORE_STIMUL#sample_size #LEN_STIMUL
                eeg_before = eeg_before[:b,:]
                breath_before = breath_before[:b]
                stimul_before = stimul_before[:b]
                if take_ecg:
                    ecg_before = ecg_before[:b]
                if take_spikes:
                    spikes_before = spikes[:b, :]
        else:
            sample_size = LEN_STIMUL + BEFORE_STIMUL
    else:
        if len(INCLUDE_STIMUL_VALUES) == 6:
            mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])], [(stimul == INCLUDE_STIMUL_VALUES[4])], [(stimul == INCLUDE_STIMUL_VALUES[5])]], axis=0)[0] # [0] because of 2d result array
        elif len(INCLUDE_STIMUL_VALUES) == 5:
            mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])], [(stimul == INCLUDE_STIMUL_VALUES[4])]], axis=0)[0] # [0] because of 2d result array
        elif len(INCLUDE_STIMUL_VALUES) == 4:
            mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])], [(stimul == INCLUDE_STIMUL_VALUES[3])]], axis=0)[0] # [0] because of 2d result array
        elif len(INCLUDE_STIMUL_VALUES) == 3:
            mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])], [(stimul == INCLUDE_STIMUL_VALUES[2])]], axis=0)[0] # [0] because of 2d result array
        elif len(INCLUDE_STIMUL_VALUES) == 2:
            mask = np.any([[(stimul == INCLUDE_STIMUL_VALUES[0])], [(stimul == INCLUDE_STIMUL_VALUES[1])]], axis=0)[0] # [0] because of 2d result array
        elif len(INCLUDE_STIMUL_VALUES) == 1:
            mask = stimul == INCLUDE_STIMUL_VALUES[0]
        sample_size = LEN_STIMUL

    if take_air:
        mask = np.add(mask,mask_air)

    if cut: # вырезаем стимулы
        eeg = eeg[mask]
        breath = breath[mask]
        stimul = stimul[mask]
        if take_ecg:
            ecg = ecg[mask]
        if take_spikes:
            spikes = spikes[mask]
    else:  # определяем канал стимулов
        stimul = np.multiply(stimul, mask.astype(np.int))
    # делаем длину кратной стимулу для будущего
    a = float(len(stimul)) % sample_size #LEN_STIMUL #np.ceil(float(len(stimul)) / LEN_STIMUL)
    n = len(stimul) // sample_size #LEN_STIMUL
    if a != 0: # обрежем массив
        b = n*sample_size #LEN_STIMUL
        eeg = eeg[:b,:]
        breath = breath[:b]
        stimul = stimul[:b]
        if take_ecg:
            ecg = ecg[:b]
        if take_spikes:
            spikes = spikes[:b, :]

    if (before or after) and (not continuous):
        #print(len(eeg_before)//BEFORE_STIMUL, len(eeg)//LEN_STIMUL)
        eeg = [eeg_before, eeg]#np.vstack((eeg_before, eeg))
        breath = [breath_before, breath]#np.hstack((breath_before, breath))
        stimul = [stimul_before, stimul]#np.hstack((stimul_before, stimul))
        if take_ecg:
            ecg = [ecg_before, ecg]
        if take_spikes:
            spikes = [spikes_before, spikes]


    return eeg, (breath, ecg, spikes), stimul, sample_size

def gen_stimuls(eeg, breath, spikes, stimul, sample_size, gen_len=0, gen_num=1):
    eeg_gen = None
    breath_gen = None
    stimul_gen = None
    n = len(stimul) // sample_size
    if gen_num == 1: # расположим вначале
        step = (sample_size-gen_len)
    else:
        step = (sample_size-gen_len) // (gen_num-1)
    for i in range(gen_num):
        # маска для одного фрагмента
        mask = np.r_[[0]*i*step, [1]*gen_len, [0]*(sample_size-gen_len-i*step)]
        # маска для всех фрагментов
        mask = np.r_[[mask]*n].flatten().astype(np.bool)
        #mask = np.reshape(mask, (n*sample_size)).astype(np.bool)
        eeg_masked = eeg[mask]
        breath_masked = breath[mask]
        stimul_masked = stimul[mask]
        eeg_gen = eeg_masked if eeg_gen is None else np.vstack((eeg_gen, eeg_masked))
        breath_gen = breath_masked if breath_gen is None else np.vstack((breath_gen, breath_masked))
        if spikes != None:
            spikes_masked = spikes[mask]
            spikes_gen = spikes_masked if spikes_gen is None else np.vstack((spikes_gen, spikes_masked))
        stimul_gen = stimul_masked if stimul_gen is None else np.vstack((stimul_gen, stimul_masked))

    return eeg_gen, breath_gen, spikes_gen, stimul_gen, gen_len

BREATH_AVE_WINDOW_RAW = 450
BREATH_AVE_WINDOW_DER = 200
BREATH_BIN_THRESH = 0
BREATH_LEN_THRESH = 200

BREATH_PROLONG_FORWARD = 100#512
BREATH_PROLONG_BACK = 412#0
BREATH_LEN = 512
BREATH_DETECT = 2 # 1 - по началу вдоха 2 - по началу выдоха

def set_breath_detect(detect=2, length=512, forward=100, back=412):
    global BREATH_PROLONG_FORWARD, BREATH_PROLONG_BACK, BREATH_LEN, BREATH_DETECT
    BREATH_PROLONG_FORWARD = forward
    BREATH_PROLONG_BACK = back
    BREATH_LEN = length
    BREATH_DETECT = detect

def bin_breath(breath, sample_size, cut=True, uncut_stimul=None):
    #breaths = np.reshape(breath, (len(breath)//LEN_STIMUL, LEN_STIMUL))
    if cut:
        breaths = np.reshape(breath, (len(breath)//sample_size, sample_size))
    else:
        breaths = breath[np.newaxis, :] # затычка, чтобы цикл for проитерировал правильно
    window_raw = np.repeat([1.], BREATH_AVE_WINDOW_RAW)
    window_der = np.repeat([1.], BREATH_AVE_WINDOW_DER)
    tot_breath = None
    for breath in breaths: # по предъявлениям
        # ave
        breath = signal.convolve(breath, window_raw, mode='same') / BREATH_AVE_WINDOW_RAW #pd.rolling_mean(breath, 451, min_periods=1, center=True)
        # der, добавим 5 в начало, 5 в конец
        temp = 10*(breath[10:] - breath[:-10]) #10*pd.DataFrame.diff(pd.DataFrame(breath), periods=10, axis=0) # axis=0 - in row
        breath = np.zeros([len(temp)+10], dtype = np.float32)
        breath[5:-5] = temp
        breath[0:5] = temp[0]
        breath[-5:] = temp[len(temp)-1]
        # ave
        breath = signal.convolve(breath, window_der, mode='same') / BREATH_AVE_WINDOW_DER #pd.rolling_mean(breath, 201, min_periods=1, center=True)
        tot_breath = breath if tot_breath is None else np.hstack((tot_breath, breath))
    breath = tot_breath
    # bin
    if BREATH_DETECT==1: # 1 - по началу вдоха
        breath[breath > BREATH_BIN_THRESH] = 1 # 1  - вдох, 0 = - выдох  >
        breath[breath <= BREATH_BIN_THRESH] = 0 # 0 = - вдох, 1 - выдох  <
    else:
        breath[breath >= BREATH_BIN_THRESH] = 0 # 1  - вдох, 0 = - выдох  >
        breath[breath < BREATH_BIN_THRESH] = 1 # 0 = - вдох, 1 - выдох  <
    # del which len < thresh,
    """
    # извлечение без учета выхода за границу LEN_STIMUL параметров BREATH_PROLONG_BACK и BREATH_PROLONG_FORWARD
    idxs = []
    i = 0
    while i < (len(breath)-BREATH_LEN_THRESH): # если сумма в скользящем окне шириной LEN_THRESH равна LEN_THRESH то нашли - запоминаем начало
        if np.sum(breath[i:i+BREATH_LEN_THRESH]) == BREATH_LEN_THRESH:
            if (breath[i-1] == 0) and (breath[i] == 1): # начало
                idxs.append(i);
            i += BREATH_LEN_THRESH
        else:
            i += 1
    print len(idxs), idxs
    # от запомненного индекса берем сигнал сзади и спереди
    breath = np.zeros([len(breath)], dtype = np.float32)
    for idx in idxs:
        breath[idx-BREATH_PROLONG_BACK:idx+BREATH_PROLONG_FORWARD] = 1
    """
    # извлечение с учетом выхода за границу LEN_STIMUL параметров BREATH_PROLONG_BACK и BREATH_PROLONG_FORWARD
    if cut:
        breath = np.reshape(breath, (len(breath) // sample_size, sample_size))
    else:
        breath = breath[np.newaxis,:] # затычка, чтобы цикл for проитерировал правильно
        stimul_idxs = np.add(np.where(np.diff(uncut_stimul) != 0), 1) # +1 корректируем сдвижку из-за разности

    idxs = []
    for j in range(len(breath)): # по предъявлениям
        i = 0
        while i < (breath.shape[1]-BREATH_LEN_THRESH): #len(breath[j]) # если сумма в скользящем окне шириной LEN_THRESH равна LEN_THRESH то нашли - запоминаем начало
            if np.sum(breath[j,i:i+BREATH_LEN_THRESH]) == BREATH_LEN_THRESH:
                if cut:
                    lower_bound, upper_bound = 0, breath.shape[1] # индексы начала и конца стимула, для этого случая тривиальные
                else:
                    if uncut_stimul[i] == 0: # находимся за пределами стимула (предъявления запаха)
                        i += BREATH_LEN_THRESH
                        continue
                    else:                    # находимся внутри стимула, определяем индексы его начала и конца
                        lower_bound = stimul_idxs[stimul_idxs < i][-1] # последний элемент массива границ, которые все меньше текущей позиции
                        upper_bound = stimul_idxs[stimul_idxs > i][0]  # первый элемент массива границ, которые все больше текущей позиции
                if (breath[j,i-1] == 0) and (breath[j,i] == 1) and ((i-BREATH_PROLONG_BACK)>=lower_bound) and ((i+BREATH_PROLONG_FORWARD)<upper_bound): #sample_size # начало
                    idxs.append(j*sample_size+i) # этот код работает и для cut=False т.к. всегда j=0
                i += BREATH_LEN_THRESH
            else:
                i += 1
    #print len(idxs), idxs
    #breath = np.reshape(breath, (len(breath) * LEN_STIMUL))
    if cut:
        n = len(breath) * sample_size # длина развернутого сигнала
    else:
        n = breath.shape[1]
    # от запомненного индекса берем сигнал сзади и спереди
    breath = np.zeros([n], dtype=np.float32)#breath = np.zeros([len(breath)], dtype = np.float32)
    for idx in idxs:
        if (breath[idx-BREATH_PROLONG_BACK] == 0):
            breath[idx-BREATH_PROLONG_BACK:idx+BREATH_PROLONG_FORWARD] = 1

    return breath

# сюда приходят массивы в которых в первой половине, это сигнал before, а вторая половина - это сами стимулы
def extract_inhales(eeg, breath, bined_breath, spikes, stimul, sample_size, before=False):
    if before: # выравниваем количество вдохов в стимуле и before
        def erase_n_breaths(count, breath):
            n = 0
            i = len(breath)-1
            while i > 0:
                if (i == len(breath)-1):
                    if (breath[i] == 1): # вдох прям на конце стумула
                        breath[i-BREATH_LEN+1:i+1] = 0  # обнуляем вдох
                        n += 1
                elif (breath[i+1] == 0) and (breath[i] == 1): # конец вдоха
                    breath[i-BREATH_LEN+1:i+1] = 0 # обнуляем вдох
                    n += 1
                i -= 1
                if n == count:
                    break

        # считаем сколько вдохов запаха в каждом интервале
        nums_breath_before, _ = get_num_breaths_per_stimul(bined_breath[0], BEFORE_STIMUL)
        nums_breath_stimul, _ = get_num_breaths_per_stimul(bined_breath[1], sample_size)
        # затираем несовпадающие по количеству вдохи, начиная с последних
        breath_before = np.reshape(bined_breath[0], (len(bined_breath[0]) // BEFORE_STIMUL, BEFORE_STIMUL))
        breath_stimul = np.reshape(bined_breath[1], (len(bined_breath[1]) // sample_size, sample_size))
        for i in range(len(nums_breath_before)): # по предъявлениям
            if nums_breath_before[i] > nums_breath_stimul[i]:
                erase_n_breaths(nums_breath_before[i]-nums_breath_stimul[i], breath_before[i])
            elif nums_breath_before[i] < nums_breath_stimul[i]:
                erase_n_breaths(nums_breath_stimul[i]-nums_breath_before[i], breath_stimul[i])
        # формируем результат
        eeg[0] = eeg[0][bined_breath[0] == 1]
        eeg[1] = eeg[1][bined_breath[1] == 1]
        breath[0] = breath[0][bined_breath[0] == 1]
        breath[1] = breath[1][bined_breath[1] == 1]
        if spikes != None:
            spikes[0] = spikes[0][bined_breath[0] == 1]
            spikes[1] = spikes[1][bined_breath[1] == 1]
        stimul[0] = stimul[0][bined_breath[0] == 1]
        stimul[1] = stimul[1][bined_breath[1] == 1]
        #eeg = eeg[bined_breath == 1]
        #breath = breath[bined_breath == 1]#breath = breath.flatten()#breath[1]#breaths[1]
        #stimul = stimul[bined_breath == 1]
        '''
        breath_before = bined_breath[0]
        breath_before = np.reshape(breath_before, (len(breath_before) // BEFORE_STIMUL, BEFORE_STIMUL))#(len(breath_before) // sample_size, sample_size))
        nums_breath_before = np.ones((len(breath_before)), dtype=np.int32)
        for j in range(len(breath_before)): # по предъявлениям
            i = BEFORE_STIMUL-2#sample_size-2
            while i > 0:
                if (breath_before[j,i+1] == 0) and (breath_before[j,i] == 1): # конец последнего вдоха
                    breath_before[j,:i-BREATH_LEN+1] = 0 # обнуляем всё с начала стимула до начала последнего вдоха
                    break
                else:
                    i -= 1
            if i==0: # не нашли вдох
                nums_breath_before[j] = 0 # чтобы пропустить запоминаение
                breath_before[j,:BREATH_LEN] = 1 # выставим наличие дыхания формально, поэтому эти примеры заведомо шумовые (чтобы не сбились индексы)

        breath_before = breath_before.flatten()
        # вырезаем последний вдох воздуха перед стимулом
        eeg_before = eeg[0]#eegs[0]
        stimul_before = stimul[0]#stimuls[0]
        eeg_before = eeg_before[breath_before == 1]
        stimul_before = stimul_before[breath_before == 1]
        eeg_before = np.reshape(eeg_before, (len(eeg_before) // BREATH_LEN, BREATH_LEN, NUM_EEG_CHANNELS))
        stimul_before = np.reshape(stimul_before, (len(stimul_before) // BREATH_LEN, BREATH_LEN))

        # делаем копии вдохов воздуха, чтобы для каждого вдоха запаха был свой вдох воздуха
        # получаем выровненные с stimul количиство вдохов воздуха
        # т.к. только один вдох в before, то число вдохов равно число стимулов в stimul
        eeg_before_align = None
        stimul_before_align = None
        for i, n in enumerate(nums_breath):
            #if (nums_breath_before[i] != 0) and (nums_breath[i] != 0):
                for j in range(n):
                    eeg_before_align = eeg_before[i] if eeg_before_align is None else np.vstack((eeg_before_align, eeg_before[i]))
                    stimul_before_align = stimul_before[i] if stimul_before_align is None else np.vstack((stimul_before_align, stimul_before[i]))
        eeg_before_align = np.reshape(eeg_before_align, (len(eeg_before_align) // BREATH_LEN, BREATH_LEN, NUM_EEG_CHANNELS))
        # stimul_before_align уже в форме (len(stimul_before_align) // BREATH_LEN, BREATH_LEN)
        # вставляем последний вдох воздуха перед вдохом запаха, одинаковый вдох воздуха вставляется перед всеми вдохами стимула
        eeg[0] = eeg[0][bined_breath[0] == 1]
        eeg[1] = eeg[1][bined_breath[1] == 1]
        breath = breath[bined_breath == 1]
        stimul = stimul[1]#stimuls[1]
        eeg = eeg[bined_breath == 1]
        breath = breath[bined_breath == 1]#breath = breath.flatten()#breath[1]#breaths[1]
        stimul = stimul[bined_breath == 1]
        eeg = np.reshape(eeg, (len(eeg) // BREATH_LEN, BREATH_LEN, NUM_EEG_CHANNELS))
        stimul = np.reshape(stimul, (len(stimul) // BREATH_LEN, BREATH_LEN))
        #assert eeg.shape[0] == eeg_before_align.shape[0]
        #print(eeg_before_align.shape, eeg.shape)
        eeg = np.hstack((eeg_before_align, eeg))
        stimul = np.hstack((stimul_before_align, stimul))

        # разворачиваем чтобы был совместим с дальнейшей обработкой
        sample_size = 2*BREATH_LEN
        eeg = np.reshape(eeg, (len(eeg) * sample_size, NUM_EEG_CHANNELS))
        stimul = np.reshape(stimul, (len(stimul) * sample_size))
        '''
    else:
        eeg = eeg[bined_breath == 1]
        breath = breath[bined_breath == 1]
        if spikes != None:
            spikes = spikes[bined_breath == 1]
        stimul = stimul[bined_breath == 1]
    sample_size = BREATH_LEN
    return eeg, breath, spikes, stimul, sample_size

VALID_PERCENT = 20
SELECTED_RATS = [] # []'a_88'

# загрузка данных нескольких крыс, только стимулы и отдельно перед стимулами за BEFORE_STUMUL
def load_data_old(mode='inhale', add_breath=False, gen_len=0, gen_num=0):
    rat_file_names, _ = get_rat_file_names()
    lb = preprocessing.LabelBinarizer()
    lb.fit(INCLUDE_STIMUL_VALUES)

    if add_breath:
        num_channels = NUM_EEG_CHANNELS + 1
    else:
        num_channels = NUM_EEG_CHANNELS

    xy_train = {}#None
    xy_valid = {}#None
    print('rat_name\ttrain_series_count\ttrain_examp_count\tvalid_series_count\tvalid_examp_count')
    for rat_name in rat_file_names:
        # выбираем определенных крыс или всех
        if len(SELECTED_RATS) != 0:
            if not (rat_name in SELECTED_RATS):
                continue
            #for selected_rat_name in SELECTED_RATS:
                #if selected_rat_name == rat_name:
                    #found = True
        if os.path.exists('%s.pickle' % rat_name):
            with open('%s.pickle' % rat_name, 'rb') as f:
                xy_train[rat_name], xy_valid[rat_name] = pickle.load(f)
            sample_size = xy_train[rat_name][0].shape[1]
            print('%s\t\t%s\t\t%d\t\t%s\t\t%d' %(rat_name, 'unknown', len(xy_train[rat_name][0]), 'unknown', len(xy_valid[rat_name][0])))
        else:
            xy_train[rat_name] = [None,None]
            #xy_train[rat_name][0] = None
            #xy_train[rat_name][1] = None
            xy_valid[rat_name] = [None,None]
            #xy_valid[rat_name][0] = None
            #xy_valid[rat_name][1] = None

            # определим некоторое кол-во серий в валидационную выборку
            num_series = len(rat_file_names[rat_name]) # кол-во серий (файлов)
            if VALID_PERCENT != 0:
                while 1:
                    valid_idx_series = sorted(np.random.randint(0, num_series, num_series*VALID_PERCENT//100))
                    if len(valid_idx_series) == len(set(valid_idx_series)): # если есть совпадающие, то заново сгенерировать
                        break
                train_idx_series = sorted(set(range(num_series)) - set(valid_idx_series))
            else:
                train_idx_series = range(num_series)
                valid_idx_series = []

            train_examp_count = 0
            valid_examp_count = 0
            for idx, file_name in enumerate(rat_file_names[rat_name]):

                eeg, breath, stimul = load_file(file_name)
                if mode == 'inhale':
                    # извлекаем данные под стимулами
                    eeg, breath, stimul, sample_size = extract_stimuls(eeg, breath, stimul)
                    # бинаризуем вдохи
                    breath = bin_breath(breath, sample_size)
                    # извлекаем данные под вдохами и стимулами
                    eeg, breath, stimul, sample_size = extract_inhales(eeg, breath, stimul, sample_size)
                elif mode == 'inhale_and_before':
                    # извлекаем данные под стимулами
                    eeg, breath, stimul, sample_size = extract_stimuls(eeg, breath, stimul, before=True, continuous = False)
                    # бинаризуем вдохи, массив составлен из двух, т.к. continuous = False
                    #breaths = np.split(breath, 2)
                    breath1 = bin_breath(breath[0], BEFORE_STIMUL)#bin_breath(breaths[0], sample_size)
                    breath2 = bin_breath(breath[1], sample_size)#bin_breath(breaths[1], sample_size)
                    breath = [breath1, breath2]#np.hstack((breath1, breath2))
                    # извлекаем данные под вдохами и стимулами
                    eeg, breath, stimul, sample_size = extract_inhales(eeg, breath, stimul, sample_size, before=True)
                elif mode == 'stimul':
                    # извлекаем данные под стимулами
                    eeg, breath, stimul, sample_size = extract_stimuls(eeg, breath, stimul)
                    # генерируем данные
                    if gen_len != 0:
                        eeg, breath, stimul, sample_size = gen_stimuls(eeg, breath, stimul, sample_size, gen_len=gen_len, gen_num=gen_num)
                elif mode == 'stimul_and_before':
                    # извлекаем данные под стимулами
                    eeg, breath, stimul, sample_size = extract_stimuls(eeg, breath, stimul, before=True, continuous = True)
                    # генерируем данные
                    if gen_len != 0:
                        eeg, breath, stimul, sample_size = gen_stimuls(eeg, breath, stimul, sample_size, gen_len=gen_len, gen_num=gen_num)
                n = len(eeg) // sample_size

                if add_breath:
                    eeg = np.hstack((eeg, np.reshape(breath,(breath.shape[0],1))))

                eeg = (np.reshape(eeg, (n, sample_size, num_channels))).astype(np.float32)
                stimul = np.reshape(stimul, (n, sample_size))[:,-1] # берем последнюю колонку массива в строках которого одинаковые коды запаха
                stimul = (lb.transform(stimul)).astype(np.float32) # перекодируем в вектор 0 и 1
                if idx in train_idx_series:
                    xy_train[rat_name][0] = eeg if xy_train[rat_name][0] is None else np.vstack((xy_train[rat_name][0], eeg))
                    xy_train[rat_name][1] = stimul if xy_train[rat_name][1] is None else np.vstack((xy_train[rat_name][1], stimul))
                    train_examp_count += n
                else:
                    xy_valid[rat_name][0] = eeg if xy_valid[rat_name][0] is None else np.vstack((xy_valid[rat_name][0], eeg))
                    xy_valid[rat_name][1] = stimul if xy_valid[rat_name][1] is None else np.vstack((xy_valid[rat_name][1], stimul))
                    valid_examp_count += n

            with open('%s.pickle' % rat_name, 'wb') as f:
                pickle.dump((xy_train[rat_name], xy_valid[rat_name]), f, -1)

            # посчитаем количество примеров каждого класса
            num_examps = np.sum(xy_train[rat_name][1], axis=0, dtype=np.int32)
            train_numbers = '('
            if len(INCLUDE_STIMUL_VALUES) > 2:
                for n in num_examps:
                    train_numbers += '{} '.format(n)
            else:
                train_numbers += '{} '.format(len(xy_train[rat_name][1])-num_examps[0])
                train_numbers += '{} '.format(num_examps[0])
            train_numbers += ')'
            num_examps = np.sum(xy_valid[rat_name][1], axis=0, dtype=np.int32)
            valid_numbers = '('
            if len(INCLUDE_STIMUL_VALUES) > 2:
                for n in num_examps:
                    valid_numbers += '{} '.format(n)
            else:
                valid_numbers += '{} '.format(len(xy_valid[rat_name][1])-num_examps[0])
                valid_numbers += '{} '.format(num_examps[0])
            valid_numbers += ')'
            print('%s\t%d\t%d%s\t%d\t%d%s' %(rat_name, len(train_idx_series), train_examp_count, train_numbers, len(valid_idx_series), valid_examp_count, valid_numbers))

    return xy_train, xy_valid, sample_size, num_channels

def get_num_breaths_per_stimul(breath, sample_size, first_only=False, cut=True, uncut_stimul=None):
    # считаем сколько вдохов запаха в каждом стимуле
    if cut:
        breath = np.reshape(breath, (len(breath) // sample_size, sample_size))
        nums_breath = np.zeros((len(breath)), dtype=np.int32)
    else:
        breath = breath[np.newaxis,:] # затычка, чтобы цикл for проитерировал правильно
        stimul_idxs = np.add(np.where(np.diff(uncut_stimul) != 0)[0], 1)  # +1 корректируем сдвижку индексов из-за разности, [0] из-за того, что where возвражает двумерный массив
        stimul_idxs = np.asarray([(stimul_idxs[i-1],stimul_idxs[i]) for i in range(1,len(stimul_idxs))])  # массив пар начало-конец стимула
        stimul_values = uncut_stimul[stimul_idxs[:, 0]]  # берем значение стимула
        stimul_idxs = stimul_idxs[stimul_values != 0]  # удаляем все пары начало-конец, которые не стимул
        nums_breath = np.zeros((len(stimul_idxs)), dtype=np.int32)

    for j in range(len(breath)): # по предъявлениям
        i = 0
        while i <= len(breath[j])-BREATH_LEN+2:
            if (i==0):
                if (breath[j,i] == 1): # начало вдоха
                    i += BREATH_LEN
                    if cut:
                        nums_breath[j] += 1
                        if first_only:  # берем только первый вдох
                            breath[j, i:][breath[j, i:]==1] = 0
                            break
                    else:
                        k = np.where(stimul_idxs[:,1] >= i)[0][0] # первый элемент массива границ, которые все больше текущей позиции
                        upper_bound = stimul_idxs[k,1]

                        nums_breath[k] += 1
                        if first_only:  # берем только первый вдох
                            breath[j, i:upper_bound][breath[j, i:upper_bound]==1] = 0
                            i = upper_bound
                else:
                    i += 1
            else:
                if (breath[j,i-1] == 0) and (breath[j,i] == 1): # начало вдоха
                    i += BREATH_LEN
                    if cut:
                        nums_breath[j] += 1
                        if first_only: # берем только первый вдох
                            breath[j, i:][breath[j, i:]==1] = 0
                            break
                    else:
                        k = np.where(stimul_idxs[:,1] >= i)[0][0]  # первый элемент массива границ, которые все больше текущей позиции
                        upper_bound = stimul_idxs[k,1]

                        nums_breath[k] += 1
                        if first_only:  # берем только первый вдох
                            breath[j, i:upper_bound][breath[j, i:upper_bound] == 1] = 0
                            i = upper_bound
                else:
                    i += 1

    return nums_breath, breath.reshape(-1)

def load_file_data(file_name, num_channels=8, take_signal='inhale', take_air=False, take_ecg=False, take_spikes=False, add_breath=False, unite_categs=[], relabel=None, gen_len=0, gen_num=0):
    global BEFORE_STIMUL, BREATH_PROLONG_FORWARD

    eeg, bio, stimul = load_file(file_name, take_ecg, take_spikes)

    if take_spikes:
        n_spike_ch = bio[-1].shape[1]

    if (take_signal == 'inhale') or (take_signal == 'first_inhale'):
        # извлекаем данные под стимулами
        eeg, bio, stimul, sample_size = extract_stimuls(eeg, bio, stimul, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes)
        breath, ecg, spikes = bio
        #raise Exception('ECG processing is required!')
        # бинаризуем вдохи
        bined_breath = bin_breath(breath, sample_size)
        # считаем сколько вдохов запаха в каждом стимуле и, если нужно, выделяем только первый вдох
        if take_signal == 'first_inhale':
            num_breaths, bined_breath = get_num_breaths_per_stimul(bined_breath, sample_size, first_only=True)
        else:
            num_breaths, bined_breath = get_num_breaths_per_stimul(bined_breath, sample_size)
        # извлекаем данные под вдохами и стимулами
        eeg, breath, spikes, stimul, sample_size = extract_inhales(eeg, breath, bined_breath, spikes, stimul, sample_size)
    elif take_signal == 'inhale_and_before':
        # извлекаем данные под стимулами
        eeg, bio, stimul, sample_size = extract_stimuls(eeg, bio, stimul, before=True, continuous=False, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes)
        breath, ecg, spikes = bio
        #raise Exception('ECG processing is required!')
        # бинаризуем вдохи, массив составлен из двух, т.к. continuous = False
        bined_breath_before = bin_breath(breath[0], BEFORE_STIMUL)
        bined_breath_stimul = bin_breath(breath[1], sample_size)
        bined_breath = [bined_breath_before, bined_breath_stimul]
        # считаем сколько вдохов запаха в каждом стимуле
        #num_breaths, _ = get_num_breaths_per_stimul(breath2, sample_size)
        #num_breaths, _ = get_num_breaths_per_stimul(breath1, sample_size)
        # извлекаем данные под вдохами и стимулами
        eeg, breath, spikes, stimul, sample_size = extract_inhales(eeg, breath, bined_breath, spikes, stimul, sample_size, before=True)
        # проверяем равенство вдохов
        num_breaths_before, _ = get_num_breaths_per_stimul(bined_breath[0], BEFORE_STIMUL)
        num_breaths_stimul, _ = get_num_breaths_per_stimul(bined_breath[1], BEFORE_STIMUL) # sample_size уже не тот
        assert np.allclose(num_breaths_before, num_breaths_stimul), 'Still nonequal number of inhales in before and stimul!'
        # конкатенация inhale и before
        n = len(eeg[0]) // sample_size

        eeg = np.hstack((np.reshape(eeg[1], (n, sample_size, num_channels)), np.reshape(eeg[0], (n, sample_size, num_channels))))
        eeg = np.reshape(eeg, (-1, num_channels))
        breath = np.hstack((np.reshape(breath[1], (n, sample_size)), np.reshape(breath[0], (n, sample_size))))
        breath = np.reshape(breath, -1)
        stimul = np.repeat(stimul[1], 2) # размножаем метки стимула, не before
        sample_size = sample_size * 2
        num_breaths = num_breaths_stimul
    elif take_signal == 'inhale_and_breath':
        # извлекаем данные под стимулами
        eeg, bio, stimul, sample_size = extract_stimuls(eeg, bio, stimul, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes)
        breath, ecg, spikes = bio
        #raise Exception('ECG processing is required!')
        # бинаризуем вдохи
        bined_breath = bin_breath(breath, sample_size)
        # считаем сколько вдохов запаха в каждом стимуле и, если нужно, выделяем только первый вдох
        num_breaths, bined_breath = get_num_breaths_per_stimul(bined_breath, sample_size, first_only=(take_signal=='first_inhale'))
        # извлекаем данные под вдохами и стимулами
        eeg, breath, spikes, stimul, sample_size = extract_inhales(eeg, breath, bined_breath, spikes, stimul, sample_size)
        # формируем из одноканального breath многоканальный
        breath = np.repeat(breath[:,np.newaxis], num_channels, axis=1)
        n = len(eeg) // sample_size
        eeg = np.hstack((np.reshape(eeg, (n, sample_size, num_channels)), np.reshape(breath, (n, sample_size, num_channels))))
        eeg = np.reshape(eeg, (-1, num_channels))
        if take_spikes:
            n = len(spikes) // sample_size
            spikes = np.hstack((np.reshape(spikes, (n, sample_size, n_spike_ch)), np.reshape(breath, (n, sample_size, n_spike_ch))))
            spikes = np.reshape(spikes, (-1, n_spike_ch))
        stimul = np.repeat(stimul, 2)
        sample_size = sample_size*2
    elif take_signal == 'inhale_and_exhale':
        # извлекаем данные под стимулами
        old_before_stimul = BEFORE_STIMUL
        BEFORE_STIMUL = BREATH_LEN # будем сдвигать сигнал, чтобы добавить 500 мс после стимула для возможного выдоха
        eeg, bio, stimul, sample_size = extract_stimuls(eeg, bio, stimul, after=True, continuous=True, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes)
        breath, ecg, spikes = bio
        BEFORE_STIMUL = old_before_stimul
        # удлиним вырезание на длину выдоха BREATH_LEN
        old_breath_prolong_forward = BREATH_PROLONG_FORWARD
        BREATH_PROLONG_FORWARD += BREATH_LEN
        bined_breath = bin_breath(breath, sample_size) # check_after=True bin_breath(breaths[1], sample_size)
        BREATH_PROLONG_FORWARD = old_breath_prolong_forward
        # извлекаем выдохи
        #exhale = np.delete(bined_breath, np.s_[-BREATH_LEN:])
        #exhale = np.append(np.zeros((BREATH_LEN)), exhale)
        # считаем сколько вдохов запаха в каждом стимуле
        num_breaths, _ = get_num_breaths_per_stimul(bined_breath, sample_size)
        # извлекаем данные под вдохами и стимулами
        eeg, breath, spikes, stimul, sample_size = extract_inhales(eeg, breath, bined_breath, spikes, stimul, sample_size)
        sample_size = 2*BREATH_LEN # вдох и выдох
        # извлекаем данные под выдохами и стимулами
        #eeg_exhale, breath_exhale, stimul, sample_size = extract_inhales(eeg, breath, exhale, stimul, sample_size)
        #n = len(eeg_inhale) // sample_size
        #eeg = np.hstack((np.reshape(eeg_exhale, (n, sample_size, num_channels)), np.reshape(eeg_inhale, (n, sample_size, num_channels))))
        #eeg = np.reshape(eeg, (-1, num_channels))
        #stimul = np.repeat(stimul, 2)
        #sample_size = sample_size*2
    elif take_signal == 'stimul':
        # извлекаем данные под стимулами
        eeg, bio, stimul, sample_size = extract_stimuls(eeg, bio, stimul, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes)
        breath, ecg, spikes = bio
        num_breaths = np.ones(len(eeg)//sample_size, dtype=np.int8)
        # генерируем данные
        if gen_len != 0:
            raise Exception('ECG processing is required!')
            eeg, breath, spikes, stimul, sample_size = gen_stimuls(eeg, breath, spikes, stimul, sample_size, gen_len=gen_len, gen_num=gen_num)
    elif take_signal == 'stimul_and_before':
        # извлекаем данные под стимулами
        # import matplotlib.pyplot as plt
        # plt.plot(stimul)
        # plt.show()

        eeg, bio, stimul, sample_size = extract_stimuls(eeg, bio, stimul, before=True, continuous = True, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes)

        # plt.plot(stimul)
        # plt.show()

        breath, ecg, spikes = bio
        num_breaths = np.ones(len(eeg)//sample_size, dtype=np.int8)
        # генерируем данные
        if gen_len != 0:
            raise Exception('ECG processing is required!')
            eeg, breath, spikes, stimul, sample_size = gen_stimuls(eeg, breath, spikes, stimul, sample_size, gen_len=gen_len, gen_num=gen_num)
    elif take_signal == 'stimul_and_breath':
        # извлекаем данные под стимулами
        eeg, bio, stimul, sample_size = extract_stimuls(eeg, bio, stimul, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes)
        breath, ecg, spikes = bio
        num_breaths = np.ones(len(eeg)//sample_size, dtype=np.int8)
        # формируем из одноканального breath многоканальный
        breath = np.repeat(breath[:,np.newaxis], num_channels, axis=1)
        n = len(eeg) // sample_size
        eeg = np.hstack((np.reshape(eeg, (n, sample_size, num_channels)), np.reshape(breath, (n, sample_size, num_channels))))
        eeg = np.reshape(eeg, (-1, num_channels))
        if take_spikes:
            n = len(spikes) // sample_size
            spikes = np.hstack((np.reshape(spikes, (n, sample_size, n_spike_ch)), np.reshape(breath, (n, sample_size, n_spike_ch))))
            spikes = np.reshape(spikes, (-1, n_spike_ch))
        stimul = np.repeat(stimul, 2)
        sample_size = sample_size*2

    n = len(eeg) // sample_size

    if add_breath:
        eeg = np.hstack((eeg, np.reshape(breath,(breath.shape[0],1))))
        if take_spikes:
            spikes = np.hstack((spikes, np.reshape(breath,(breath.shape[0],1))))

    eeg = (np.reshape(eeg, (n, sample_size, num_channels))).astype(np.float32)
    if take_spikes:
        n_ = len(spikes) // sample_size
        spikes = (np.reshape(spikes, (n_, sample_size, n_spike_ch))).astype(np.float32)
    stimul = np.reshape(stimul, (n, sample_size))[:,0 if take_signal=='inhale_and_exhale' else -1] #-1 берем последнюю колонку массива в строках которого одинаковые коды запаха

    if relabel:  # для тестовой выборки если нужен relabel
        stimul = np.array(list(map(lambda y: relabel[y], stimul)))  # сначала делаем relabel, а потом объединяем

    # объединим указанные в подсписках категории в отдельные классы с меткой первого класса из подсписка; если остались классы, то они так и остаются отдельными. Пример:  1,[2,4],[8,16] -> 1,2,8
    if unite_categs: # если список не пустой, а состоит из подсписков
        for unite_categ in unite_categs: # по подсписками с объединяемыми категориями
            if len(unite_categ)-1 == 1:
                mask = stimul == unite_categ[1]
            elif len(unite_categ)-1 == 2:
                mask = np.any([[(stimul == unite_categ[1])], [(stimul == unite_categ[2])]], axis=0)[0] # [0] because of 2d result array
            elif len(unite_categ)-1 == 3:
                mask = np.any([[(stimul == unite_categ[1])], [(stimul == unite_categ[2])], [(stimul == unite_categ[3])]], axis=0)[0] # [0] because of 2d result array
            elif len(unite_categ)-1 == 4:
                mask = np.any([[(stimul == unite_categ[1])], [(stimul == unite_categ[2])], [(stimul == unite_categ[3])], [(stimul == unite_categ[4])]], axis=0)[0] # [0] because of 2d result array
            elif len(unite_categ)-1 == 5:
                mask = np.any([[(stimul == unite_categ[1])], [(stimul == unite_categ[2])], [(stimul == unite_categ[3])], [(stimul == unite_categ[4])], [(stimul == unite_categ[5])]], axis=0)[0] # [0] because of 2d result array
            elif len(unite_categ)-1 == 6:
                mask = np.any([[(stimul == unite_categ[1])], [(stimul == unite_categ[2])], [(stimul == unite_categ[3])], [(stimul == unite_categ[4])], [(stimul == unite_categ[5])], [(stimul == unite_categ[6])]], axis=0)[0] # [0] because of 2d result array
            stimul[mask] = unite_categ[0] # объединяемым классам присваиваем метку первого класса
            #stimul_values = list(set(INCLUDE_STIMUL_VALUES)-set(union_classes[1:]))
            #lb.fit(stimul_values)

    return eeg, (breath, ecg, spikes), stimul, num_breaths

def load_train_valid_contr(mode='inhale', add_breath=False, gen_len=0, gen_num=0, load_test=True, saveload=False, union_classes=None, split_pos=12, split_test=True):
    lb = preprocessing.LabelBinarizer()
    lb.fit(INCLUDE_STIMUL_VALUES)#[:-1])
    #le = preprocessing.LabelEncoder()
    #le.fit(INCLUDE_STIMUL_VALUES[:-1])

    if add_breath:
        num_channels = NUM_EEG_CHANNELS + 1
    else:
        num_channels = NUM_EEG_CHANNELS

    # в словаре будем хранить название файла и его данные
    # для обуч выборки исп. 'all':массив всех файлов
    # для тестовой выборки исп. file_name1:массив этого файла, file_name2:массив этого файла...

    # определим названия файлов обуч, валид. и тест. выборки
    file_names = glob('./*.txt')
    train_file_names = []
    test_file_names = []
    test_split_pos = []
    for file_name in file_names:
        #if (predict and (file_name.find('m_') != -1)) or ((not predict) and (file_name.find('t_') != -1)):
        if (file_name.find('m_') != -1) or (file_name.find('t_') != -1):
            test_file_names.append(file_name)
            #if file_name.find('t_') != -1:
            #if os.path.getsize(file_name)//(1024*1024) < 20: # 20Mb
            #    test_split_pos.append(4) # 4 известных стимула (valid), остальные неизвестные (contr)
            #else:
            test_split_pos.append(split_pos)
        else:
            train_file_names.append(file_name)

    print('set\texamp_count')
    #xy_train = {} здесь должны быть как ключеваые слова номера крыс, example: 'a_87'
    #xy_valid = {}
    #xy_contr = {}
    #rat_name = 'curr'
    #rat_xy = {}
    #rat_xy[rat_name] = [None,None,None]
    #num_breaths_contr = {}

    # загрузим данные обуч выборки
    if saveload and os.path.exists('trainset.pickle'):
        with open('trainset.pickle', 'rb') as f:
            xy_train = pickle.load(f)
        sample_size = xy_train[0].shape[1]
        num_classes = max(2,xy_train[1].shape[1])
        #print('%s\t\t%s\t\t%d\t\t%s\t\t%d' %(rat_name, 'unknown', len(xy_train[rat_name][0]), 'unknown', len(xy_valid[rat_name][0])))
    else:
        xy_train = [None,None]
        if not split_test:
            xy_valid = [None,None]

        for file_name in train_file_names:
            eeg, breath, stimul, num_breaths = load_file_data(file_name, lb, num_channels=num_channels, mode=mode, add_breath=add_breath, gen_len=gen_len, gen_num=gen_num, union_classes=union_classes)

            if split_test:
                xy_train[0] = eeg if xy_train[0] is None else np.vstack((xy_train[0], eeg))
                xy_train[1] = stimul if xy_train[1] is None else np.vstack((xy_train[1], stimul))
            else:
                if num_breaths is not None:
                    n = np.sum(num_breaths[:split_pos])
                    num_breaths_valid = num_breaths[split_pos:]
                else: # для стимулов
                    n = split_pos
                    num_breaths_valid = np.ones(eeg.shape[0]-n, dtype=np.int32)
                xy_train[0] = eeg[:n]
                xy_train[1] = stimul[:n]
                xy_valid[0] = eeg[n:]
                xy_valid[1] = stimul[n:]

        sample_size = xy_train[0].shape[1]
        num_classes = max(2,xy_train[1].shape[1])
        if saveload:
            with open('trainset.pickle', 'wb') as f:
                pickle.dump(xy_train, f, -1)
    # посчитаем количество примеров каждого класса
    num_examps = np.sum(xy_train[1], axis=0, dtype=np.int32)
    train_numbers = '('
    if num_classes > 2:
        for n in num_examps:
            train_numbers += '{} '.format(n)
    else:
        train_numbers += '{} '.format(len(xy_train[1])-num_examps[0])
        train_numbers += '{} '.format(num_examps[0])
    train_numbers += ')'
    print('%s\t%d%s' %('trainset', xy_train[0].shape[0], train_numbers))

    if load_test:
        if saveload and os.path.exists('testset.pickle'):
            with open('testset.pickle', 'rb') as f:
                xy = pickle.load(f)
                xy_valid = xy[0]
                xy_contr = xy[1]
        else:
            # загрузим данные валид и контр выборки
            if split_test:
                xy_valid = [None,None]
            xy_contr = [None,None]
            for i, file_name in enumerate(test_file_names): # этот цикл работает только для одного файла
                eeg, breath, stimul, num_breaths = load_file_data(file_name, lb, num_channels=num_channels, mode=mode, add_breath=add_breath, gen_len=gen_len, gen_num=gen_num, union_classes=union_classes)
                #file_name = os.path.split(file_name)[1] # берем только название файла без пути
                if split_test:
                    if num_breaths is not None:
                        n = np.sum(num_breaths[:test_split_pos[i]])
                        num_breaths_contr = num_breaths[test_split_pos[i]:]
                    else: # для стимулов
                        n = test_split_pos[i]
                        num_breaths_contr = np.ones(eeg.shape[0]-n, dtype=np.int32)
                    xy_valid[0] = eeg[:n]
                    xy_valid[1] = stimul[:n]
                    xy_contr[0] = eeg[n:]
                    xy_contr[1] = stimul[n:]
                else:
                    xy_contr[0] = eeg
                    xy_contr[1] = stimul

            if saveload:
                with open('testset.pickle', 'wb') as f:
                    pickle.dump([xy_valid, xy_contr], f, -1)

        # посчитаем количество примеров каждого класса
        num_examps = np.sum(xy_valid[1], axis=0, dtype=np.int32)
        valid_numbers = '('
        if num_classes > 2:
            for n in num_examps:
                valid_numbers += '{} '.format(n)
        else:
            valid_numbers += '{} '.format(len(xy_valid[1])-num_examps[0])
            valid_numbers += '{} '.format(num_examps[0])
        valid_numbers += ')'
        num_examps = np.sum(xy_contr[1], axis=0, dtype=np.int32)
        contr_numbers = '('
        if num_classes > 2:
            for n in num_examps:
                contr_numbers += '{} '.format(n)
        else:
            contr_numbers += '{} '.format(len(xy_contr[1])-num_examps[0])
            contr_numbers += '{} '.format(num_examps[0])
        contr_numbers += ')'
        print('%s\t%d%s' %('validset', xy_valid[0].shape[0], valid_numbers))
        print('%s\t%d%s' %('contrset', xy_contr[0].shape[0], contr_numbers))
        if np.sum(num_examps)==0:
            xy_contr = None
    return xy_train, xy_valid, xy_contr, sample_size, num_channels, num_breaths_contr

# загрузка данных одной крысы текущей директории
def load_rat_files(dir='.', task='odor', take_signal='inhale', take_air=False, take_ecg=False, take_spikes=False, add_breath=False, unite_categs=[], relabel=None, gen_len=0, gen_num=0, saveload=False, select_files=None, file_ext='txt', verbose=True):
    if take_air:
        odors = INCLUDE_STIMUL_VALUES + [AIR_LABEL]
    else:
        odors = INCLUDE_STIMUL_VALUES[:]
    if task=='odor':
        if unite_categs:
            new_odors = []
            for odor in odors:
                found = False
                first = False
                for unite_categ in unite_categs:
                    if odor in unite_categ:
                        found = True
                        first = unite_categ.index(odor) == 0
                        break
                if (not found) or (found and first): # если категория не содержится в подсписках для объединения, то добавляем ее как отдельный класс; если содержится, то добавляем только если это первая категория из подсписка
                    new_odors.append(odor)
            odors = sorted(new_odors) # sorted на случай загрузки тестовых данных, для которых этот список может быть перемешан из-за relabel
    elif task=='onco':
        odors = [0] + odors
    num_categs = len(odors)

    if add_breath:
        num_channels = NUM_EEG_CHANNELS + 1
    else:
        num_channels = NUM_EEG_CHANNELS
    train_or_test = dir.split(os.sep)[-1]  # вырезаем слово train или test или пусто

    # в словаре будем хранить название файла и его данные

    # определим названия файлов
    select_files = [os.path.join(dir, select_file) for select_file in select_files]
    if task=='odor':
        if select_files:
            file_names = [str(select_file) for select_file in select_files]
        else:
            file_names = glob(dir+os.sep+'*.%s' %(file_ext))
    elif task=='onco':
        if select_files:
            file_names_ = select_files
        else:
            #file_names_0 = glob(dir + '\\0\\*.%s' % (file_ext))
            #file_names_1 = glob(dir + '\\1\\*.%s' % (file_ext))  # extend(glob('.\\1\\*.%s' %(file_ext)))
            file_names_ = [glob(dir + os.sep + '%d%s*.%s' % (odor, os.sep, file_ext)) for odor in odors]
        if (train_or_test=='') or (train_or_test=='train'):
            #min_count = min(len(file_names_0), len(file_names_1))
            min_count = min(map(len, file_names_))
            #file_names = file_names_0[:min_count] + file_names_1[:min_count] # получаем равные по количеству больные-здоровые выборку
            file_names = []
            for i in range(len(odors)):
                file_names += file_names_[i][:min_count]
        elif train_or_test=='test':
            #min_count = len(file_names_0)
            min_count = len(file_names_[0])
            #file_names = file_names_0 + file_names_1 # берем все файлы
            file_names = []
            for i in range(len(odors)):
                file_names += file_names_[i]
        #print('Found:    %d files - 0class,  %d files - 1class' %(len(file_names_0), len(file_names_1)))
        #print('Selected: '+('%d files - 0class,  %d files - 1class' %(min_count, min_count) if train_or_test=='train' else 'all'))
        #print('Ignored:  '+('%d files - 0class,  %d files - 1class' %(len(file_names_0)-min_count, len(file_names_1)-min_count) if train_or_test=='train' else 'none'))

    if verbose:
        print('file\tcateg_id\texamp_count')
    log = 'file\tcateg_id\texamp_count\n'
    file_xy = {}

    if saveload and os.path.exists(dir+os.sep+'signal_'+take_signal+'.pkl'):
        with open(dir+os.sep+'signal_'+take_signal+'.pkl', 'rb') as f:
            file_xy = pickle.load(f)
        sample_size = file_xy.values()[0][0].shape[2]
        #num_classes = max(2,file_xy.values()[0][1].shape[1])
    else:
        for i, full_file_name in enumerate(file_names):
            # загрузим данные
            file_name = full_file_name.split(os.sep)[-1]
            if task=='onco':
                category = full_file_name.split(os.sep)[-2]
                # в название файла добавляем счетчик
                if category == '0': # сначала идут "здоровые"
                    file_name = '%02d_%s' %(((i+1)-0*min_count)*num_categs-2, file_name) #'%02d_%s' %(2*(i+1)-1, file_name) # нечетный - здоровый, начинаем с 1 (i+1)
                if category == '1': # потом идут lung "больные", поэтому может вычитать min_count чтобы получить чередование файлов четные-нечетные
                    file_name = '%02d_%s' %(((i+1)-1*min_count)*num_categs-1, file_name) #'%02d_%s' %(2*(i+1-min_count), file_name) # # четный - больной
                if category == '2': # потом идут stom "больные", поэтому может вычитать min_count чтобы получить чередование файлов четные-нечетные
                    file_name = '%02d_%s' %(((i+1)-2*min_count)*num_categs-0, file_name) #'%02d_%s' %(2*(i+1-min_count), file_name) # # четный - больной

            file_xy[file_name] = [None,None,None,None,None,None]
            eeg, bio, stimul, num_breaths = load_file_data(full_file_name, num_channels=num_channels, take_signal=take_signal, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes, add_breath=add_breath, unite_categs=unite_categs, relabel=relabel, gen_len=gen_len, gen_num=gen_num)
            file_xy[file_name][0] = eeg.transpose((0,2,1))
            file_xy[file_name][1] = stimul
            file_xy[file_name][2] = num_breaths
            file_xy[file_name][3] = bio[0]
            file_xy[file_name][4] = bio[1] # ecg
            file_xy[file_name][5] = bio[2].transpose((0, 2, 1)) if take_spikes else None # spikes

            if task=='onco':
                category = full_file_name.split(os.sep)[-2]
                if num_categs > 2:
                    # stimul всегда возвращаеся как массив из [0 1 0] - единицы всегда во второй колонке, поэтому поменяем местами колонки в соотвествии с категорией
                    if category == '0': file_xy[file_name][1] = file_xy[file_name][1][:, [1, 0, 2]]
                    if category == '2': file_xy[file_name][1] = file_xy[file_name][1][:, [0, 2, 1]]
                else:
                    if category == '0': file_xy[file_name][1] = file_xy[file_name][1]*0 # устанавливаем класс 'здоровые'-0, т.к. по умолчанию все 'больные'-1
        if saveload:
            with open(dir+os.sep+'signal_'+take_signal+'.pkl', 'wb') as f:
                pickle.dump(file_xy, f, -1)
        sample_size = list(file_xy.values())[0][0].shape[2] # python3
        #num_classes = max(2,file_xy.values()[0][1].shape[1])

    # посчитаем количество примеров каждого класса
    for file_name in sorted(file_xy):
        """
        def count(log, tab, subj, day=''):
            # tab = '\t' if not noday else ''
            categs_id_line = ''
            examps_count_line = ''
            tot_num_values = 0
            for idx in range(num_categs):
                categ = categs[idx]
                num_values = np.sum(data_xy[subj][day][series][1] == (idx + 1))  # np.sum(data_xy[subj][series][1] == (idx+1)) if noday else
                categs_id_line += '%s\t' % categ
                examps_count_line += '%d\t' % num_values
                tot_num_values += num_values
            log += tab + '\tCategs:\t%s\tTotal\n' % categs_id_line
            log += tab + '\tCount:\t%s\t%d\n' % (examps_count_line, tot_num_values)
            return log
        """
        num_examps = Counter(file_xy[file_name][1])#np.sum(file_xy[file_name][1], axis=0, dtype=np.int32)
        categs_id = ''
        examps_count = ''
        for categ in sorted(num_examps):
            categs_id += '{} '.format(categ)
            examps_count += '{} '.format(num_examps[categ])
        """
        numbers = '('
        if num_categs > 2:
            for n in num_examps:
                numbers += '{} '.format(n)
        else:
            numbers += '{} '.format(len(file_xy[file_name][1])-num_examps[0])
            numbers += '{} '.format(num_examps[0])
        numbers += ')'
        """
        if task=='onco':
            file_name_ = file_name[3:] # [3:] чтобы отрезать счетчик
        else:
            file_name_ = file_name
        line = '%s\t(%s)\t%d(%s)' %(file_name_, categs_id.strip(), file_xy[file_name][0].shape[0], examps_count.strip())
        if verbose:
            print(line)
        log += line + '\n'

    return file_xy, sample_size, num_channels, log, odors

# загрузка данных нескольких крыс
def load_rats(task='odor', take_signal='inhale', take_air=False, add_breath=False, gen_len=0, gen_num=0, saveload=False, select_rats=None, file_ext='dat'):
    dir_mask = '*odor*' if task == 'odor' else '*onco*'
    _, rats_dirs = get_rat_file_names(dir_mask, file_ext)

    # делаем текущей очередную директорию и load_rat_files
    rats_data = {} # dict( rat_name: dict(rat_dir/day:data) )
    rats_logs = {}
    for rat_name in rats_dirs: # здесь rat_name это ключ типа '15a'
        if select_rats and (select_rats.count(rat_name) == 0): # добавим "-", чтобы "select_rats.count(rat_name)" различало 12а и 2а
            continue
        #if task == 'onco' and rats_dirs[rat_name][3:7] != 'onco':
        #    continue
        rats_data[rat_name] = {}
        rats_logs[rat_name] = ''
        for rat_dir in rats_dirs[rat_name]:
            os.chdir(rat_dir)#print(os.getcwd())
            if task == 'odor':
                rat_day = rat_dir[2:11]#3:12
            elif task == 'onco':
                rat_day = rat_dir[2:6]
            print('rat - %s, day - %s' %(rat_name, rat_day))
            xy, sample_size, num_channels, log, odors = load_rat_files(task=task, take_signal=take_signal, take_air=take_air, add_breath=add_breath, gen_len=gen_len, gen_num=gen_num, saveload=saveload, select_files=None, file_ext=file_ext)
            rats_logs[rat_name] += log
            print('')
            rats_data[rat_name][rat_day] = xy
            os.chdir(os.pardir)
    return rats_data, sample_size, num_channels, rats_logs, rats_dirs

def load_data(task='odor', dir='.', subjects='ignore', days='ignore', series='all', categs=[1,2], piece=2, take_signal='stimul', take_air=False, take_ecg=False, take_spikes=False, det_breath=1, cut_breath=[-400,100], add_breath=False, unite_categs=[], relabel=None, file_ext='txt', saveload=False, log_dir='', verbose=True):
    # делаем длину кратной стимулу
    len_piece = piece * SAMPLE_RATE
    #if take_ecg:
    #    pass

    if task == 'odor' or task == 'onco':
        global INCLUDE_STIMUL_VALUES
        INCLUDE_STIMUL_VALUES = categs[:]
        #take_signal = task_specs['take_signal'] #'stimul'
        #take_air = task_specs['take_air']
        #det_breath = task_specs['det_breath']
        # if det_breath == 1:
        #     set_breath_detect(detect=1, length=512, forward=512, back=0)
        # elif det_breath == 2:
        #     set_breath_detect(detect=2, length=512, forward=100, back=412) #length=512, forward=100, back=412
        set_breath_detect(detect=det_breath, length=cut_breath[1]-cut_breath[0], forward=cut_breath[1], back=abs(cut_breath[0])) #length=512, forward=100, back=412
        file_xy, sample_size, num_channels, log, categs = load_rat_files(dir=dir, select_files=series, task=task, take_signal=take_signal, take_air=take_air, take_ecg=take_ecg, take_spikes=take_spikes, add_breath=add_breath, unite_categs=unite_categs, relabel=relabel, gen_len=0, gen_num=0, saveload=saveload, file_ext=file_ext, verbose=verbose)
        #log = print_log('Классы %s' % (str(categs)[1:-1]), log, verbose)

        data_xy = {'': {'': file_xy}}
        # сборка всех файлов в один массив
        # xy = [None, None, None]  # x, y, вдохи
        # for file_name in sorted(file_xy):
        #     xy[0] = file_xy[file_name][0] if xy[0] is None else np.vstack((xy[0], file_xy[file_name][0]))
        #     xy[1] = file_xy[file_name][1] if xy[1] is None else np.vstack((xy[1], file_xy[file_name][1]))
        #     xy[2] = file_xy[file_name][2] if xy[2] is None else np.hstack((xy[2], file_xy[file_name][2]))  # вдохи
        #data_xy = xy
    elif task == 'stress':
        global NUM_ALL_CHANNELS
        global ADD_STIM_CHANNEL
        NUM_ALL_CHANNELS = 10 if take_ecg else 9
        ADD_STIM_CHANNEL = True

        num_eeg_channels = 8
        eeg_ch_names = ['F1', 'Fz', 'FC1', 'FCz', 'C1', 'Cz', 'CP1', 'CPz']
        other_ch_names = ['STIMUL'] + ['ECG'] if take_ecg else []
        ch_names = eeg_ch_names + other_ch_names
        montage = mne.channels.read_montage('standard_1020', ch_names)
        ch_type = ['eeg'] * len(eeg_ch_names) + ['stim']

        if series == 'all':
            series = [subj for subj in os.listdir(dir) if not ('.' in subj)]
        data_xy = {'':{'':{}}}
        for data_name in series:
            #data_xy[''][''][data_name] = [None,None]
            file_name = glob(os.path.join(dir,data_name,'*.dat')) # здесь только один файл
            eeg, _, stimul = load_file(file_name[0])
            # меняем каналы местами и удаляем пустые
            if os.path.exists(os.path.join(dir,data_name,'channels.txt')):
                channel_idxs = np.loadtxt(os.path.join(dir,data_name,'channels.txt'), dtype=np.int8, delimiter=' ')
                eeg = eeg[:,channel_idxs]
                num_eeg_channels = len(channel_idxs)
                eeg_ch_names = eeg_ch_names[:num_eeg_channels]
                ch_names = eeg_ch_names + other_ch_names
                ch_type = ['eeg'] * num_eeg_channels + ['stim']
            # в канале стимула меняем 0 на 3
            stimul[stimul==0] = 3
            # создаем raw объект
            eeg_ = 1e-6 * np.array(eeg).T
            eeg_and_stim = np.vstack((eeg_, stimul))
            info = mne.create_info(ch_names, sfreq=SAMPLE_RATE, ch_types=ch_type, montage=montage)
            raw = mne.io.RawArray(eeg_and_stim, info, verbose=False)
            events = mne.find_events(raw, consecutive=True, stim_channel='STIMUL', verbose=False) #shortest_event=0,
            # измеряем стрессовый сегмент (он последний) и отрезаем такой же по длине нестрессовый непосредственно до подачи тока
            #begin_stress, end_stress = events[-1,0], eeg.shape[0]
            begin_stress, end_stress = events[-1, 0]+10*SAMPLE_RATE, events[-1,0]+70*SAMPLE_RATE  # 1 минута с 10 секунды по 70
            len_signal = end_stress-begin_stress
            for i, prev_categ, categ in events[:]:#,2
                if categ==1:
                    end_normal = i#events[i,0]
                    begin_normal = end_normal-len_signal
                    break
            n = len_signal // len_piece
            b = n * len_piece
            # нестресс + стресс
            normal = eeg[begin_normal:begin_normal+b, :].reshape(-1,len_piece,num_eeg_channels)
            stress = eeg[begin_stress:begin_stress+b, :].reshape(-1,len_piece,num_eeg_channels)
            x = np.vstack((normal, stress))
            y = np.concatenate((np.zeros(normal.shape[0],dtype=np.int8), np.ones(stress.shape[0],dtype=np.int8)))
            data_xy[''][''][data_name] = [x,y]
        categs = [0,1]
        log = count_examples(data_xy, categs, verbose=verbose)

    elif task == 'fitness':
        global DECIMATE, LEN_STIMUL_SECS, IDXS_EEG_CHANNELS, NUM_EEG_CHANNELS, IDX_BREATH_CHANNEL, IDX_ECG_CHANNEL
        assert take_ecg, 'Error!\nParameter "take_ecg" must be True!'
        log = print_log('Subj name\tPiece count', '', verbose)
        subjs = subjects#[subj for subj in os.listdir(dir) if not ('.' in subj)]
        data_xy = {}#{'':{'':{}}}
        subj_num_pieces = []
        for subj_name in subjs:
            #data_xy[''][''][data_name] = [None,None]
            # читаем настройки крысы
            _, settings = load_settings(os.path.join(dir, subj_name, 'settings'))
            DECIMATE = settings["data"]["orig_sample_rate"]  // settings["data"]["final_sample_rate"]
            LEN_STIMUL_SECS = settings["signal"]["len_piece_secs"] # 5
            IDXS_EEG_CHANNELS = settings["channel"]["eeg_idxs"]
            NUM_EEG_CHANNELS = len(IDXS_EEG_CHANNELS)
            IDX_BREATH_CHANNEL = settings["channel"]["breath_idx"]
            IDX_ECG_CHANNEL = settings["channel"]["ecg_idx"]
            # загружаем файл, читаем название файла в котором есть три минуты фона вначале
            file_name = os.path.join(dir, subj_name, settings["data"]["train_names"][0]) #берем первый файл, часто единственный
            eeg, (breath, ecg), stimul = load_file(file_name, take_ecg)
            # находим положение первого стимула и вырезаем всё до него в Х, разрезая на куски
            end_background = np.where(stimul != AIR_LABEL)[0][0]
            assert end_background > len_piece, 'Error!\nToo short background segment before stimuls!'
            num_pieces = end_background // len_piece
            subj_num_pieces.append(num_pieces)
            log = print_log('%s\t%d' % (subj_name, num_pieces), log, verbose)
            idx_begin = 0
            idx_end = num_pieces*len_piece
            eeg, breath, ecg = eeg[idx_begin:idx_end].reshape((num_pieces,len_piece,NUM_EEG_CHANNELS)), breath[idx_begin:idx_end].reshape((num_pieces,len_piece)), ecg[idx_begin:idx_end].reshape((num_pieces,len_piece))
            # находим Y, из папки .logs читаем из подпапки все эксперименты, берем из последнего эксперимента файл *batch_splits.log и оттуда читаем точность
            file_name = glob(os.path.join(dir, '.'+log_dir, '*', '*batch_splits.log'))[-1]
            with open(file_name,'r') as file:
                found_subj = False
                while not found_subj:
                    line = file.readline()
                    if subj_name in line:
                        found_subj = True
                        found_acc = False
                        while not found_acc:
                            line = file.readline()
                            if 'accuracy' in line:
                                found_acc = True
                                acc = float(line.split(': ')[-1])
            acc = [acc] * num_pieces
            data_xy[subj_name] = ((eeg, breath, ecg), acc)

        # сборка
        min_num_pieces = 1#min(subj_num_pieces)
        eegs, breaths, ecgs, accs = None, None, None, []
        for subj in sorted(data_xy):
            ((eeg, breath, ecg), acc) = data_xy[subj]
            eeg, breath, ecg, acc = eeg[:min_num_pieces], breath[:min_num_pieces], ecg[:min_num_pieces], acc[:min_num_pieces]
            eegs = eeg if eegs is None else np.vstack((eegs, eeg))
            breaths = breath if breaths is None else np.vstack((breaths, breath))
            ecgs = ecg if ecgs is None else np.vstack((ecgs, ecg))
            accs.extend(acc)# = acc if accs is None else np.concatenate((accs, acc))
        eegs = np.transpose(eegs, (0,2,1))
        accs = np.array(accs)
        xy = ((eegs, breaths, ecgs), accs)

        categs = None
        log = print_log('Loaded %d pieces with %d pieces from each of %d subjects \n' %(len(accs), min_num_pieces, len(subjects)), log, verbose)
        return (min_num_pieces, xy), log, categs

    return data_xy, log, categs #sample_size, num_channels,

def load_one_day(task='odor', interval=5, decimate=1, cv_factor=2, odors=[1,2], take_air=False, cut_signal='inhale', det_breath=1, diff=False, file_ext='txt'):
    global INCLUDE_STIMUL_VALUES
    global DECIMATE
    global LEN_STIMUL
    INCLUDE_STIMUL_VALUES = odors[:]
    DECIMATE = decimate
    if task=='onco':
        LEN_STIMUL = interval*SAMPLE_RATE
    if det_breath==1:
        set_breath_detect(detect=1, length=512, forward=512, back=0)
    elif det_breath==2:
        set_breath_detect(detect=2, length=512, forward=100, back=412)
    file_xy, sample_size, num_channels, log = load_rat_files(task=task, mode=cut_signal, take_air=take_air, add_breath=False, gen_len=0, gen_num=0, saveload=True, file_ext=file_ext)

    rat_name = 'rat1'
    yield [rat_name] # при первом вызове возвращаем список пациентов

    if task=='onco':
        odors = [0]+odors

    # сборка всех файлов в один массив
    xy = [None,None,None] # x, y, вдохи
    cv_split = [0]
    for file_name in sorted(file_xy):
        xy[0] = file_xy[file_name][0] if xy[0] is None else np.vstack((xy[0], file_xy[file_name][0]))
        xy[1] = file_xy[file_name][1] if xy[1] is None else np.vstack((xy[1], file_xy[file_name][1]))
        xy[2] = file_xy[file_name][2] if xy[2] is None else np.hstack((xy[2], file_xy[file_name][2])) # вдохи
        if task=='odor':
            cv_split.append(xy[0].shape[0])
        elif task=='onco':
            if (int(file_name.split('_')[0]) % cv_factor) == 0: # N fold - при сортировке пара нечетный-четный номер файла составляет валидационную выборку
                cv_split.append(xy[0].shape[0])
    #return cv_split, (xy,sample_size,num_channels,log), odors
    yield (rat_name, cv_split, (xy, sample_size, num_channels, log), odors)

def load_many_days(rats=[], task='odor', interval=5, decimate=1, cv_factor=2, odors=[1,2], take_air=False, cut_signal='inhale', det_breath=1, diff=False, file_ext='txt'):
    global INCLUDE_STIMUL_VALUES
    global DECIMATE
    global LEN_STIMUL
    global MIN_NUM_DAYS
    global MIN_NUM_SERIES
    global MAX_NUM_SERIES
    INCLUDE_STIMUL_VALUES = odors[:]
    DECIMATE = decimate
    if task=='onco':
        LEN_STIMUL = interval*SAMPLE_RATE
        MIN_NUM_DAYS = 1
        MIN_NUM_SERIES = 4
        MAX_NUM_SERIES = 100
    if det_breath==1:
        set_breath_detect(detect=1, length=512, forward=512, back=0)
    elif det_breath==2:
        set_breath_detect(detect=2, length=512, forward=100, back=412)
    rats_data, sample_size, num_channels, rats_logs, rats_dirs = load_rats(task=task, mode=cut_signal, take_air=take_air, add_breath=False, gen_len=0, gen_num=0, saveload=True, select_rats=rats, file_ext=file_ext)

    if task=='onco':
        odors = [0]+odors

    # сборка всех файлов в один массив
    for rat_name in sorted(rats_data):
        xy = [None,None,None] # x, y, вдохи
        cv_split = [0]
        for rat_day in sorted(rats_data[rat_name]):
            for file_name in sorted(rats_data[rat_name][rat_day]):
                xy[0] = rats_data[rat_name][rat_day][file_name][0] if xy[0] is None else np.vstack((xy[0], rats_data[rat_name][rat_day][file_name][0]))
                xy[1] = rats_data[rat_name][rat_day][file_name][1] if xy[1] is None else np.vstack((xy[1], rats_data[rat_name][rat_day][file_name][1]))
                xy[2] = rats_data[rat_name][rat_day][file_name][2] if xy[2] is None else np.hstack((xy[2], rats_data[rat_name][rat_day][file_name][2])) # вдохи
                if task=='onco': # кроссвалидация между файлами папки onco, здесь всегда единственный rat_day - это 'onco', т.к. имя папки onco-15a
                    if (int(file_name.split('_')[0]) % cv_factor) == 0: # N fold - при сортировке пара нечетный-четный номер файла составляет валидационную выборку
                        cv_split.append(xy[0].shape[0])
            if task=='odor': # кроссвалидация между днями одной крысы
                cv_split.append(xy[0].shape[0])
        print('rat %s: %d-fold cross-validation' %(rat_name, len(cv_split)-1))
        if task=='onco':
            os.chdir(rats_dirs[rat_name][0]) # сделаем текущей директорией папку этой крысы, чтобы сохранить туда картинки и др. файлы, [0] т.к. там list, но для onco c единственным элементом
        #go_over_segments(cv=cv, cv_split=cv_split, loaded_data=(xy,sample_size,num_channels,rats_logs[rat_name]), odors=odors, fixed_odor=fixed_odor, take_air=take_air, two_freq_bands=two_freq_bands, feats=feats, study_classes=study_classes, cut_signal=cut_signal, det_breath=det_breath, diff=diff)
        yield (cv_split, (xy, sample_size, num_channels,rats_logs[rat_name]), odors)
        if task=='onco':
            os.chdir(os.pardir) # восстановим текущую директорию

def creat_mne_raw_object(file_name, task='', channel_idxs=None, bin_breath=False, interval=None, encode_categ='', invert_breath=False, take_air=False, first_inhale=False, cut=False):#, read_events=True):
    """Create a mne raw instance from csv/txt file."""
    # get chanel names
    orig_eeg_ch_names = ['F1','Fz','FC1','FCz','C1','Cz','CP1','CPz']
    other_ch_names = ['Breath','STIMUL']
    if ADD_STIM_CHANNEL:
        eeg_ch_names = orig_eeg_ch_names[:NUM_ALL_CHANNELS-1]
        ch_names = eeg_ch_names + other_ch_names # количество каналов (NUM_ALL_CHANNELS) увеличивается на 1 (NUM_ALL_CHANNELS+1)
    else:
        eeg_ch_names = orig_eeg_ch_names[:NUM_ALL_CHANNELS-2]
        ch_names = eeg_ch_names + other_ch_names # количество каналов не изменяется (NUM_ALL_CHANNELS)
    # read EEG standard montage from mne
    montage = mne.channels.read_montage('standard_1020', ch_names)
    ch_type = ['eeg']*len(eeg_ch_names) + ['bio'] + ['stim']
    eeg, breath, stimul = load_file(file_name)
    # извлекаем данные под стимулами
    if interval: # для onco
        # синтезируем новый канал-стимул по количеству интервалов
        interval = interval*SAMPLE_RATE # 1000

        def gen_stimul(stimul, interval):
            # пролонгированные стимулы как для запахов на 1 меньше длины interval, дополняя одним 0
            stimul_len = len(stimul)//interval * interval
            interval = interval - 2 # вставка нуля будет вначало и конец фрагмента
            fragm_count = stimul_len//interval
            stimul_len_ = fragm_count * interval
            stimul = np.ones(stimul_len_)
            # вставка в конец
            fragm_idxs = np.arange(0, stimul_len, interval)[1:]
            stimul =  np.insert(stimul, fragm_idxs, 0)
            # вставка в начало
            fragm_idxs = np.arange(0, stimul_len, interval+1)[:-1]
            stimul = np.insert(stimul, fragm_idxs, 0)
            return stimul

        if task=='stress':
            # меняем каналы местами и удаляем пустые
            if channel_idxs is not None:
                eeg = eeg[:, channel_idxs]
            eeg_ch_names = eeg_ch_names[:len(channel_idxs)]
            ch_names = eeg_ch_names + other_ch_names  # количество каналов не изменяется (NUM_ALL_CHANNELS)
            ch_type = ['eeg'] * len(eeg_ch_names) + ['bio'] + ['stim']
            # в канале стимула меняем 0 на 3
            stimul[stimul == 0] = 3
            # создаем raw объект
            eeg_ = 1e-6 * np.array(eeg).T
            eeg_and_stim = np.vstack((eeg_, breath, stimul))
            info = mne.create_info(ch_names[:len(channel_idxs)+2], sfreq=SAMPLE_RATE, ch_types=ch_type, montage=montage)
            raw = mne.io.RawArray(eeg_and_stim, info, verbose=False)
            events = mne.find_events(raw, consecutive=True, stim_channel='STIMUL', verbose=False)  # shortest_event=0,
            # измеряем стрессовый сегмент (он последний) и отрезаем такой же по длине нестрессовый непосредственно до подачи тока
            #begin_stress, end_stress = events[-1,0], eeg.shape[0]
            begin_stress, end_stress = events[-1,0]+10*SAMPLE_RATE, events[-1,0]+70*SAMPLE_RATE  #
            len_signal = end_stress - begin_stress
            for i, prev_categ, categ in events[:]:  # ,2
                if categ == 1:
                    end_normal = i  # events[i,0]
                    begin_normal = end_normal - len_signal
                    break
            n = len_signal // interval
            b = n * interval
            # нестресс + стресс
            eeg_normal = eeg[begin_normal:begin_normal + b, :]
            breath_normal = np.zeros(eeg_normal.shape[0])
            stimul_normal = gen_stimul(stimul[begin_normal:begin_normal + b], interval) # метка нестресса 1

            eeg_stress = eeg[begin_stress:begin_stress + b, :]
            breath_stress = np.zeros(eeg_stress.shape[0])
            stimul_stress = gen_stimul(stimul[begin_stress:begin_stress + b], interval)*2 # метка стресса 2

            eeg = np.vstack((eeg_normal, eeg_stress))
            breath = np.concatenate((breath_normal, breath_stress))
            stimul = np.concatenate((stimul_normal, stimul_stress))
        else:
            stimul = gen_stimul(stimul, interval)

        # пролонгированные стимулы как для запахов
        #stimul_values = np.arange(1,len(stimul)/interval+1)
        #stimul =  np.repeat(stimul_values, interval)

        # импульсные метки стимулов, 1 - начало стимула, остальные 0
        #stimul_len = len(stimul)//interval * interval
        #stimul = np.zeros(stimul_len)
        #stimul_idxs = np.arange(0, stimul_len, interval)
        #stimul[stimul_idxs] = 1


        sample_size = interval #eeg, breath, stimul, sample_size = data.extract_stimuls(eeg, breath, stimul, take_air=take_air)
        # подрезаем eeg если он не кратен 1000
        #if (eeg.shape[0] % data.SAMPLE_RATE) != 0:
        if eeg.shape[0] != stimul.shape[0]:
            eeg = eeg[:stimul.shape[0]]
            breath = breath[:stimul.shape[0]]
    else:
        eeg, breath, stimul, sample_size = extract_stimuls(eeg, breath, stimul, take_air=take_air, cut=cut)

    if encode_categ=='dir2categ': # для ERS сделаем для здоровых людей stimul=1, для больных - stimul=2
        categ = int(file_name.split(os.sep)[0]) + 1 # переводим в 1 и 2 вместо 0 и 1
        stimul = stimul * categ
        #print('categ', categ)
    elif encode_categ=='fileindex2categ': # читаем номер файла после последнего символа _ в качестве категории
        categ = int(file_name.split('.')[0].split('_')[-1]) + 1 # [0]-отрезаем расширение, +1 переводим в 1 2 3 вместо 0 1 2
        stimul = stimul * categ
    elif encode_categ=='filetimeindex2categ': # читаем время файла и номер файла в качестве категории
        categ = int(file_name.split('.')[0].split('_')[-2] + file_name.split('.')[0].split('_')[-1]) # [0]-отрезаем расширение, + соединяем время и индекс
        stimul = stimul * categ

    # инвертируем дыхание
    if invert_breath:
        breath = -1*breath
    # бинаризуем дыхание
    if bin_breath:
        set_breath_detect(detect=2, length=500, forward=100, back=400)#data.set_breath_detect(detect=1, length=512, forward=512, back=0)
        breath_bin = bin_breath(breath, sample_size, cut=cut, uncut_stimul=stimul)
        #stimul_code = np.reshape(stimul, (len(stimul) // sample_size, sample_size))[:,0]
        # count number of inhales in each stimul and left only first inhale, if specified
        num_breaths, breath_bin = get_num_breaths_per_stimul(breath_bin, sample_size, first_inhale, cut, stimul)
        stimul = stimul*breath_bin
        #print('%s\t\t%s' % (file_name, zip(stimul_code.tolist(), num_breaths.tolist())))
    else:
        # добавляем нуль в начало и конец, чтобы MNE могла найти все стимулы
        #print(eeg.shape,stimul.shape,breath.shape)
        eeg = np.vstack((np.zeros((1,eeg.shape[1])),eeg,np.zeros((1,eeg.shape[1]))))
        stimul = np.hstack((np.zeros(1),stimul,np.zeros(1)))
        breath = np.hstack((np.zeros(1),breath,np.zeros(1)))
        #print(eeg.shape,stimul.shape,breath.shape)
    eeg = 1e-6*np.array(eeg).T
    eeg_and_stim = np.vstack((eeg, 1e-6*breath, stimul))
    #eeg = eeg[breath == 1]
    #stimul = stimul[breath == 1]
    #sample_size = breath.BREATH_LEN
    # create and populate MNE info structure
    info = mne.create_info(ch_names, sfreq=1000.0, ch_types=ch_type, montage=montage)
    #info['filename'] = file_name
    # create raw object
    raw = mne.io.RawArray(eeg_and_stim, info, verbose=False)
    #if interval: # для onco
    #    return raw, stimul_values.tolist()
    #elif bin_breath:
    if bin_breath:
        return raw, num_breaths
    else:
        return raw, None


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    #global ALTER_STIM_CHANNEL
    SAMPLE_RATE = 1000
    LEN_STIMUL = 5000
    LEN_STIMUL_SECS = 5
    DECIMATE = 10  # 5#1
    AIR_LABEL = 64
    NUM_EEG_CHANNELS = 13
    NUM_ALL_CHANNELS = 15
    ADD_STIM_CHANNEL = False
    ALTER_STIM_CHANNEL = False
    IDXS_EEG_CHANNELS = slice(0, NUM_EEG_CHANNELS)
    IDX_BREATH_CHANNEL = 13
    IDX_ECG_CHANNEL = 14

    ALTER_STIM_CHANNEL = True
    #IDXS_EEG_CHANNELS = slice(0, NUM_EEG_CHANNELS)
    #IDX_BREATH_CHANNEL = 3
    eeg, (breath, ecg, spikes), stimul = load_file('d:\\data\Rat\\2019\\2019_12_03-n33-5odor-18chan-10hz-low\\03.12.2019.N33.1.plx', take_ecg=True, take_spikes=True)# d:\\data\Rat\\2019\\2019_05_31-n26-3odor-10chan-10hz-low\\31.05.19.N26.4.plx
    #eeg, (breath, ecg, spikes), stimul = load_file('d:\\data\\Rat\\2019\\2019_11_27-n34-5odor-18chan-10hz-low\\27.11.19.N34.2.plx', take_ecg=True, take_spikes=True)# d:\\data\Rat\\2019\\2019_05_31-n26-3odor-10chan-10hz-low\\31.05.19.N26.4.plx
    #eeg, (breath, ecg, spikes), stimul = load_file('d:\\data\\Rat\\2019\\2019_11_27-n34-5odor-18chan-10hz-low\\27.11.19.N34.3.plx', take_ecg=True, take_spikes=True)# d:\\data\Rat\\2019\\2019_05_31-n26-3odor-10chan-10hz-low\\31.05.19.N26.4.plx
    #eeg, (breath, ecg, spikes), stimul = load_file('d:\\data\\Rat\\2019\\2019_11_27-n34-5odor-18chan-10hz-low\\27.11.19.N34.4.plx', take_ecg=True, take_spikes=True)# d:\\data\Rat\\2019\\2019_05_31-n26-3odor-10chan-10hz-low\\31.05.19.N26.4.plx
    plt.plot(stimul)
    plt.show()


